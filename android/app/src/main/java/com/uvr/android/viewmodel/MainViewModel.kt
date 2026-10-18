package com.uvr.android.viewmodel

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.uvr.android.audio.AudioProcessor
import com.uvr.android.audio.DecodedAudio
import com.uvr.android.audio.OutputFormat
import com.uvr.android.audio.OutputStem
import com.uvr.android.audio.SeparationService
import com.uvr.android.audio.SilenceTrimmer
import com.uvr.android.ml.ArchitectureType
import com.uvr.android.ml.ModelInfo
import com.uvr.android.ml.ModelManager
import com.uvr.android.ml.SeparationParams
import com.uvr.android.ml.SeparationResult
import com.uvr.android.ml.Separator
import com.uvr.android.ml.mdx.MdxModelMeta
import com.uvr.android.utils.FileUtils
import com.uvr.android.utils.MediaFile
import com.uvr.android.utils.SavedSettings
import com.uvr.android.utils.SettingsStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.coroutineContext

/**
 * ViewModel for the main separation screen.
 *
 * Audio is kept on disk throughout — no full-length FloatArray is ever held in
 * the Java heap. The decoded temp file lives until another file is selected or
 * the selection is cleared.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val audioProcessor = AudioProcessor(application)

    // -------------------------------------------------------------------------
    // UI State
    // -------------------------------------------------------------------------

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _models = MutableStateFlow<List<ModelInfo>>(emptyList())
    val models: StateFlow<List<ModelInfo>> = _models.asStateFlow()

    private val _selectedModel = MutableStateFlow<ModelInfo?>(null)
    val selectedModel: StateFlow<ModelInfo?> = _selectedModel.asStateFlow()

    private val _waveform = MutableStateFlow<FloatArray?>(null)
    val waveform: StateFlow<FloatArray?> = _waveform.asStateFlow()

    private val _params = MutableStateFlow(SeparationParams())
    val params: StateFlow<SeparationParams> = _params.asStateFlow()

    private val _outputFormat = MutableStateFlow(OutputFormat.WAV)
    val outputFormat: StateFlow<OutputFormat> = _outputFormat.asStateFlow()

    private val _importing = MutableStateFlow(false)
    val importing: StateFlow<Boolean> = _importing.asStateFlow()

    /** Desktop UVR's "Trim Silence": drop silent stretches (0.5 s or longer) from the stems. */
    private val _trimSilence = MutableStateFlow(false)
    val trimSilence: StateFlow<Boolean> = _trimSilence.asStateFlow()

    /** For a video input: the stem saved as its audio, [ALL_STEMS], or [AUDIO_ONLY]. Null = the first option. */
    private val _videoStem = MutableStateFlow<String?>(null)
    val videoStem: StateFlow<String?> = _videoStem.asStateFlow()

    /** One-off messages for a snackbar (import/delete results). */
    private val _messages = MutableSharedFlow<String>(extraBufferCapacity = 4)
    val messages: SharedFlow<String> = _messages.asSharedFlow()

    // File-backed decoded audio — never holds raw PCM in the Java heap
    private var decodedAudio: DecodedAudio? = null
    /** The selected folder's files, decoded one at a time as they are separated; null when a file is selected. */
    private var folderFiles: List<MediaFile>? = null
    private var loadJob: Job? = null
    private var separationJob: Job? = null
    /** The progress notification's text, so it is only re-posted when it changes. */
    @Volatile private var lastNotification: String? = null

    private val settingsStore = SettingsStore(application)
    private var saved = SavedSettings()

    init {
        refreshModels()
        // Restore last session's choices once loaded (models are already listed)
        viewModelScope.launch {
            saved = runCatching { settingsStore.load() }.getOrDefault(SavedSettings())
            saved.outputFormat
                ?.let { name -> OutputFormat.values().firstOrNull { it.name == name } }
                ?.let { _outputFormat.value = it }
            saved.trimSilence?.let { _trimSilence.value = it }
            val model = saved.modelKey?.let { key -> _models.value.firstOrNull { it.key == key && it.isSupported } }
            if (model != null) applyModel(model)
            else _selectedModel.value?.let { applyModel(it) }   // re-apply saved VR params
        }
    }

    // -------------------------------------------------------------------------
    // Actions
    // -------------------------------------------------------------------------

    fun refreshModels() {
        ModelManager.scanModels()
        publishModels()
    }

    private fun publishModels() {
        _models.value = ModelManager.availableModels.toList()
        val current = _selectedModel.value
        if (current == null || _models.value.none { it.filePath == current.filePath && it.fileName == current.fileName }) {
            _selectedModel.value = null
            _models.value.firstOrNull { it.isSupported }?.let { applyModel(it) }
        }
    }

    /** Copy a model file the user picked into the app and select it if usable. */
    fun importModel(uri: Uri) {
        _importing.value = true
        viewModelScope.launch {
            try {
                val model = withContext(Dispatchers.IO) { ModelManager.importModel(uri) }
                publishModels()
                if (model.isSupported) {
                    selectModel(model)
                    _messages.tryEmit("Imported ${model.displayName}")
                } else {
                    _messages.tryEmit("Imported ${model.displayName}, but it can't be used: ${model.unsupportedReason}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Import failed", e)
                _messages.tryEmit("Import failed: ${e.message}")
            } finally {
                _importing.value = false
            }
        }
    }

    fun deleteModel(model: ModelInfo) {
        viewModelScope.launch {
            val deleted = withContext(Dispatchers.IO) { ModelManager.deleteModel(model) }
            publishModels()
            _messages.tryEmit(if (deleted) "Deleted ${model.displayName}" else "Couldn't delete ${model.displayName}")
        }
    }

    /** User picked [model]: use it and remember it for next launch. */
    fun selectModel(model: ModelInfo) {
        applyModel(model)
        viewModelScope.launch { settingsStore.saveModel(model.key) }
    }

    private fun applyModel(model: ModelInfo) {
        _selectedModel.value = model
        var params = defaultParamsFor(model)
        if (model.architectureType == ArchitectureType.VR_ARCH) {
            saved.vrWindowSize?.let { params = params.copy(windowSize = it) }
            saved.vrAggression?.let { params = params.copy(aggressiveness = it) }
            saved.vrTta?.let { params = params.copy(tta = it) }
            saved.vrUseGpu?.let { params = params.copy(useGpu = it) }
        }
        if (model.architectureType == ArchitectureType.DEMUCS) {
            saved.demucsOverlap?.let { params = params.copy(demucsOverlap = it) }
            saved.demucsShifts?.let { params = params.copy(demucsShifts = it) }
        }
        if (model.architectureType == ArchitectureType.MDX_NET) {
            saved.mdxOverlap?.let { params = params.copy(mdxOverlap = it.takeIf { o -> o >= 0f }) }
            saved.mdxDenoise?.let { params = params.copy(mdxDenoise = it) }
            saved.mdxFastGpu?.let { params = params.copy(mdxFastGpu = it) }
        }
        _params.value = params
    }

    /** Identifies a model across launches. */
    private val ModelInfo.key get() = "${source.name}:${filePath ?: fileName}"

    /**
     * Load [uri] as the input file. Returns false (and does nothing) while a
     * separation is running, since that job still reads the current decoded file.
     */
    fun selectAudioFile(uri: Uri): Boolean {
        if (_uiState.value.status is ProcessingStatus.Separating) {
            Log.w(TAG, "Ignoring $uri — separation in progress")
            return false
        }
        dropSelection()
        val fileName = FileUtils.displayName(getApplication(), uri)
        _uiState.value = _uiState.value.copy(
            selectedFileName = fileName,
            durationSeconds = 0f,
            isAudioReady = false,
            isVideo = false,
            isFolder = false,
            folderFileCount = 0,
            status = ProcessingStatus.Loading
        )
        loadJob = viewModelScope.launch {
            try {
                val decoded = audioProcessor.decode(uri, keepVideo = FileUtils.looksLikeVideo(getApplication(), uri, fileName))
                decodedAudio = decoded
                _waveform.value = audioProcessor.generateWaveformPreview(decoded)
                val durationSec = decoded.totalSamples.toFloat() / decoded.sampleRate / decoded.channels
                _uiState.value = _uiState.value.copy(
                    status = ProcessingStatus.Ready,
                    durationSeconds = durationSec,
                    isAudioReady = true,
                    isVideo = decoded.videoFile != null
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load: ${e.message}", e)
                // FFmpeg's reason (logged above) means little to users
                _uiState.value = _uiState.value.copy(
                    status = ProcessingStatus.Error(
                        "Couldn't read this file — it may be damaged, or have no sound.")
                )
            }
        }
        return true
    }

    /**
     * Use the audio and video files in the folder [treeUri] (from the system folder picker,
     * subfolders included) as the input; they are separated one after another. Returns false
     * (and does nothing) while a separation is running.
     */
    fun selectFolder(treeUri: Uri): Boolean {
        if (_uiState.value.status is ProcessingStatus.Separating) {
            Log.w(TAG, "Ignoring $treeUri — separation in progress")
            return false
        }
        dropSelection()
        val app = getApplication<Application>()
        _uiState.value = _uiState.value.copy(
            selectedFileName = FileUtils.folderName(app, treeUri),
            durationSeconds = 0f,
            isAudioReady = false,
            isVideo = false,
            isFolder = true,
            folderFileCount = 0,
            status = ProcessingStatus.Loading
        )
        loadJob = viewModelScope.launch {
            try {
                val files = withContext(Dispatchers.IO) { FileUtils.listMediaFiles(app, treeUri) }
                if (files.isEmpty()) {
                    _uiState.value = _uiState.value.copy(
                        status = ProcessingStatus.Error("No audio or video files in this folder.")
                    )
                    return@launch
                }
                folderFiles = files
                _uiState.value = _uiState.value.copy(
                    status = ProcessingStatus.Ready,
                    folderFileCount = files.size,
                    isAudioReady = true,
                    isVideo = files.any { it.isVideo }
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Failed to list $treeUri: ${e.message}", e)
                _uiState.value = _uiState.value.copy(
                    status = ProcessingStatus.Error("Couldn't read this folder.")
                )
            }
        }
        return true
    }

    /** Forget the current input file or folder and delete its decoded temp files. */
    private fun dropSelection() {
        loadJob?.cancel()
        decodedAudio?.delete()
        decodedAudio = null
        folderFiles = null
        _waveform.value = null
    }

    /**
     * Separate the loaded file, or each file of the selected folder in turn. Stems are
     * encoded, then published to Music/UVR ([outputDir] is only used on Android 9 and
     * below). The input stays selected, so it can be run again with another model or setting.
     * In a folder, a file that fails is reported and skipped.
     * Returns false if there is nothing to separate.
     */
    fun startSeparation(outputDir: File): Boolean {
        val model = _selectedModel.value?.takeIf { it.isSupported } ?: return false
        val folder = folderFiles
        val audio = decodedAudio
        if (folder == null && audio == null) return false
        if (_uiState.value.status is ProcessingStatus.Separating) return false
        val params = _params.value
        val run = RunSettings(model, params, _outputFormat.value, effectiveVideoStem(model, params),
            _trimSilence.value, outputDir)
        val fileName = _uiState.value.selectedFileName

        _uiState.value = _uiState.value.copy(
            status = ProcessingStatus.Separating(0f, fileCount = folder?.size ?: 1),
            lastOutputDir = outputDir
        )

        val app = getApplication<Application>()
        val previous = separationJob
        separationJob = viewModelScope.launch {
            // A cancelled run stops at its next model window; let it finish cleaning
            // up (and release the keep-alive service) before this one starts.
            previous?.cancelAndJoin()
            SeparationService.keepAlive(app)
            lastNotification = null
            try {
                _uiState.value = _uiState.value.copy(status =
                    if (folder == null) ProcessingStatus.Done(listOf(FileResult(fileName, separateOne(audio!!, fileName, run))))
                    else separateFolder(folder, run))
            } catch (e: CancellationException) {
                throw e   // cancelSeparation() already reset the UI
            } catch (e: Exception) {
                Log.e(TAG, "Separation failed: ${e.message}", e)
                _uiState.value = _uiState.value.copy(
                    status = ProcessingStatus.Error("Separation failed: ${e.message}")
                )
            } finally {
                SeparationService.release(app)
            }
        }
        return true
    }

    /** Decode and separate each of [files]; Done with every file's result, or Error if none worked. */
    private suspend fun separateFolder(files: List<MediaFile>, run: RunSettings): ProcessingStatus {
        val results = files.mapIndexed { i, file ->
            coroutineContext.ensureActive()   // cancelSeparation() has already reset the UI
            _uiState.value = _uiState.value.copy(status = ProcessingStatus.Separating(
                0f, "Reading file…", fileIndex = i, fileCount = files.size, fileName = file.name,
                overall = i.toFloat() / files.size))
            var audio: DecodedAudio? = null
            try {
                audio = audioProcessor.decode(file.uri, keepVideo = file.isVideo)
                FileResult(file.name, separateOne(audio, file.name, run))
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Separating ${file.name} failed: ${e.message}", e)
                FileResult(file.name, emptyList(), error = e.message ?: e.javaClass.simpleName)
            } finally {
                audio?.delete()
            }
        }
        val failed = results.filter { it.error != null }
        val first = failed.firstOrNull()
        return if (first != null && failed.size == results.size) {
            ProcessingStatus.Error(
                if (results.size == 1) "Couldn't separate ${first.fileName}: ${first.error}"
                else "None of the ${results.size} files could be separated. ${first.fileName}: ${first.error}")
        } else {
            ProcessingStatus.Done(results)
        }
    }

    /** Separate [audio] (from the file [fileName]) and save its stems, plus its videos for a video. */
    private suspend fun separateOne(audio: DecodedAudio, fileName: String, run: RunSettings): List<Pair<String, OutputStem>> {
        val app = getApplication<Application>()
        val baseName = FileUtils.sanitizeBaseName(fileName)
        var result: SeparationResult? = null
        try {
            val separator = Separator.create(run.model, run.params)
            showProgress(0f)
            result = separator.separate(audio, app.cacheDir) { progress -> showProgress(progress) }
            if (run.trimSilence) {
                showProgress(1f, "Trimming silence…", notify = false)
                withContext(Dispatchers.IO) {
                    result.stems.forEach { SilenceTrimmer.trimInPlace(it.file, result.sampleRate) }
                }
            }
            val outputs = result.stems.map { stem ->
                stem.name to encodeAndPublish(stem.file,
                    "${baseName}_(${stem.name}).${run.format.extension}", run.format, result.sampleRate, run.outputDir)
            }
            val videos = audio.videoFile?.let { saveVideos(it, result, run.videoChoice, baseName, run.outputDir) }.orEmpty()
            return outputs + videos
        } finally {
            result?.stems?.forEach { it.file.delete() }
        }
    }

    /**
     * Show the running file's [progress] in [stage] (null: separating) on screen and, if
     * [notify], in the notification. Ignored once the run is cancelled.
     */
    private fun showProgress(progress: Float, stage: String? = null, notify: Boolean = true) {
        val current = _uiState.value.status as? ProcessingStatus.Separating ?: return
        // Only the separation itself moves a folder's overall progress, so it never goes back
        val overall = if (stage == null) (current.fileIndex + progress) / current.fileCount else current.overall
        val status = current.copy(progress = progress, stage = stage, overall = overall)
        _uiState.value = _uiState.value.copy(status = status)
        if (!notify) return
        val percent = (progress * 100).toInt()
        val text = (if (status.fileCount > 1) "File ${status.fileIndex + 1} of ${status.fileCount} · " else "") +
            "${stage ?: "Separating…"} $percent%"
        if (text != lastNotification) {
            lastNotification = text
            SeparationService.updateProgress(getApplication(), text,
                if (status.fileCount > 1) (overall * 100).toInt() else percent)
        }
    }

    /**
     * Save [video] once per chosen stem, with that stem as its audio, to Movies/UVR.
     * A video that can't be saved is reported and skipped — the stems are already saved.
     */
    private suspend fun saveVideos(
        video: File, result: SeparationResult, choice: String, baseName: String, fallbackDir: File
    ): List<Pair<String, OutputStem>> {
        if (choice == AUDIO_ONLY) return emptyList()
        val stems = if (choice == ALL_STEMS) result.stems else result.stems.filter { it.name.equals(choice, ignoreCase = true) }
        if (stems.isEmpty()) {
            _messages.tryEmit("No \"$choice\" stem was made, so no video was saved")
            return emptyList()
        }
        val app = getApplication<Application>()
        return stems.mapNotNull { stem ->
            val stage = "Saving ${stem.name} video…"
            val muxed = File(app.cacheDir, "uvr_video_${System.nanoTime()}.mp4")
            try {
                val saved = audioProcessor.muxVideo(video, stem.file, muxed, result.sampleRate) { progress ->
                    showProgress(progress, stage)
                }
                "${stem.name} (Video)" to audioProcessor.publish(saved, "${baseName}_(${stem.name}).${saved.extension}", fallbackDir)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Saving the ${stem.name} video failed", e)
                _messages.tryEmit("Couldn't save the ${stem.name} video: ${e.message}")
                null
            } finally {
                muxed.delete()
                File(app.cacheDir, "${muxed.nameWithoutExtension}.mkv").delete()
            }
        }
    }

    /** The video choice that applies with [model]: the user's pick if still offered, else the first option. */
    private fun effectiveVideoStem(model: ModelInfo, params: SeparationParams): String {
        val options = videoStemOptions(model, params)
        return _videoStem.value?.takeIf { it in options } ?: options.first()
    }

    fun setVideoStem(stem: String) {
        _videoStem.value = stem
    }

    private suspend fun encodeAndPublish(
        raw: File, displayName: String, format: OutputFormat, sampleRate: Int, fallbackDir: File
    ): OutputStem {
        val app = getApplication<Application>()
        val encoded = File(app.cacheDir, "uvr_out_${System.nanoTime()}.${format.extension}")
        try {
            audioProcessor.encode(raw, encoded, format, sampleRate)
            return audioProcessor.publish(encoded, displayName, fallbackDir)
        } finally {
            encoded.delete()
        }
    }

    /** A file is decoded, or a folder's files are listed, ready to separate. */
    private val hasInput get() = decodedAudio != null || folderFiles != null

    /** Stop the running separation; the loaded file or folder stays selected. */
    fun cancelSeparation() {
        separationJob?.cancel()
        _uiState.value = _uiState.value.copy(
            status = if (hasInput) ProcessingStatus.Ready else ProcessingStatus.Idle
        )
    }

    /** Dismiss a load/separation error shown on the home screen. */
    fun dismissError() {
        if (_uiState.value.status is ProcessingStatus.Error) {
            _uiState.value = _uiState.value.copy(
                status = if (hasInput) ProcessingStatus.Ready else ProcessingStatus.Idle
            )
        }
    }

    fun updateParams(update: (SeparationParams) -> SeparationParams) {
        val params = update(_params.value)
        _params.value = params
        if (_selectedModel.value?.architectureType == ArchitectureType.VR_ARCH) {
            saved = saved.copy(
                vrWindowSize = params.windowSize, vrAggression = params.aggressiveness,
                vrTta = params.tta, vrUseGpu = params.useGpu
            )
            viewModelScope.launch {
                settingsStore.saveVrParams(params.windowSize, params.aggressiveness, params.tta, params.useGpu)
            }
        }
        if (_selectedModel.value?.architectureType == ArchitectureType.DEMUCS) {
            saved = saved.copy(demucsOverlap = params.demucsOverlap, demucsShifts = params.demucsShifts)
            viewModelScope.launch { settingsStore.saveDemucsParams(params.demucsOverlap, params.demucsShifts) }
        }
        if (_selectedModel.value?.architectureType == ArchitectureType.MDX_NET) {
            saved = saved.copy(mdxOverlap = params.mdxOverlap ?: -1f, mdxDenoise = params.mdxDenoise, mdxFastGpu = params.mdxFastGpu)
            viewModelScope.launch { settingsStore.saveMdxParams(params.mdxOverlap, params.mdxDenoise, params.mdxFastGpu) }
        }
    }

    fun setOutputFormat(format: OutputFormat) {
        _outputFormat.value = format
        viewModelScope.launch { settingsStore.saveOutputFormat(format.name) }
    }

    fun setTrimSilence(on: Boolean) {
        _trimSilence.value = on
        viewModelScope.launch { settingsStore.saveTrimSilence(on) }
    }

    fun clearSelection() {
        dropSelection()
        _uiState.value  = UiState()
    }

    override fun onCleared() {
        decodedAudio?.delete()
        super.onCleared()
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private companion object {
        const val TAG = "MainViewModel"
    }

    /** A run's settings, fixed when it starts. */
    private class RunSettings(
        val model: ModelInfo,
        val params: SeparationParams,
        val format: OutputFormat,
        val videoChoice: String,
        val trimSilence: Boolean,
        val outputDir: File
    )

    private fun defaultParamsFor(model: ModelInfo): SeparationParams {
        val base = when (model.architectureType) {
            // Desktop UVR defaults: Overlap "Default", Denoise off
            ArchitectureType.MDX_NET -> SeparationParams(mdxOverlap = null, mdxDenoise = false)
            // Desktop UVR defaults: All Stems, Overlap 0.25, Shifts 2
            ArchitectureType.DEMUCS  -> SeparationParams(demucsOverlap = 0.25f, demucsShifts = 2)
            // Desktop UVR defaults: Window Size 512, Aggression Setting 5
            ArchitectureType.VR_ARCH -> SeparationParams(windowSize = 512, aggressiveness = 0.05f)
        }
        // The model decides which stem is primary (the SP-UVR models output Instrumental)
        return base.copy(
            primaryStemName   = model.primaryStem   ?: base.primaryStemName,
            secondaryStemName = model.secondaryStem ?: base.secondaryStemName
        )
    }
}

/** Video choice: save a video with every stem. */
const val ALL_STEMS = "All Stems"

/** Video choice: save only the audio stems. */
const val AUDIO_ONLY = "Audio Only (No Video)"

/**
 * What a video input can be saved with, given the model and its settings: each stem it
 * makes, then [ALL_STEMS] and [AUDIO_ONLY]. The first entry is the default.
 */
fun videoStemOptions(model: ModelInfo?, params: SeparationParams): List<String> {
    val stems = when {
        model?.architectureType != ArchitectureType.DEMUCS -> listOf(params.primaryStemName, params.secondaryStemName)
        params.demucsStem != null -> listOf(params.demucsStem, MdxModelMeta.secondaryStemOf(params.demucsStem))
        else -> listOf("Vocals", "Drums", "Bass", "Other")
    }
    return stems.distinct() + ALL_STEMS + AUDIO_ONLY
}

// -------------------------------------------------------------------------
// UI State data classes
// -------------------------------------------------------------------------

data class UiState(
    val selectedFileName: String = "",
    val durationSeconds: Float   = 0f,
    val status: ProcessingStatus = ProcessingStatus.Idle,
    val lastOutputDir: File?     = null,
    /** The selected file is decoded and can be separated. */
    val isAudioReady: Boolean    = false,
    /** The selected file is a video (for a folder: some of its files may be), which can be saved again with a stem as its audio. */
    val isVideo: Boolean         = false,
    /** A folder is selected ([selectedFileName] is its name); its files are separated one after another. */
    val isFolder: Boolean        = false,
    /** How many audio and video files the selected folder has (0 while they are being listed). */
    val folderFileCount: Int     = 0
)

sealed class ProcessingStatus {
    object Idle      : ProcessingStatus()
    object Loading   : ProcessingStatus()
    object Ready     : ProcessingStatus()
    /**
     * [progress] of file [fileIndex] (of [fileCount], named [fileName] in a folder) in [stage]: what runs
     * besides the separation itself (e.g. saving a video); null while separating. [overall]: of the whole run.
     */
    data class Separating(
        val progress: Float,
        val stage: String? = null,
        val fileIndex: Int = 0,
        val fileCount: Int = 1,
        val fileName: String? = null,
        val overall: Float = 0f
    ) : ProcessingStatus()
    /** Each input file's outputs, in order. */
    data class Done(val files: List<FileResult>) : ProcessingStatus()
    data class Error(val message: String)                                   : ProcessingStatus()
}

/** What one input file produced: (stem name, published file) in the model's output order, or why it failed. */
data class FileResult(
    val fileName: String,
    val stems: List<Pair<String, OutputStem>>,
    val error: String? = null
)
