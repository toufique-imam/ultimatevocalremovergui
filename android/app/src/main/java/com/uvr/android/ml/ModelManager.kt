package com.uvr.android.ml

import android.content.Context
import android.net.Uri
import android.util.Log
import com.uvr.android.ml.demucs.DemucsModelMeta
import com.uvr.android.ml.mdx.MdxModelMeta
import com.uvr.android.ml.vr.VrModelMeta
import com.uvr.android.utils.FileUtils
import java.io.File
import java.io.FileInputStream

/**
 * Singleton that tracks available ONNX and PTL models on device.
 *
 * Model loading strategy (checked in order):
 *  1. Bundled models: assets/models/  (shipped inside APK — .onnx or .ptl)
 *  2. Internal storage: filesDir/models/  (models imported from the Models screen)
 *  3. External storage: getExternalFilesDir("models")  (e.g. `adb push`)
 */
object ModelManager {

    private const val TAG = "ModelManager"
    private const val MODELS_DIR = "models"

    /** Optional files beside a model with its base name (extension → what they are): other runtimes' versions. */
    private val COMPANIONS = mapOf("tflite" to "GPU", "mnn" to "MNN")

    private lateinit var appContext: Context

    /** All models discovered at startup or after [scanModels]. */
    val availableModels: MutableList<ModelInfo> = mutableListOf()

    fun initialize(context: Context) {
        appContext = context.applicationContext
        scanModels()
    }

    fun getModelsDir(): File {
        val dir = File(appContext.filesDir, MODELS_DIR)
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    fun getExternalModelsDir(): File? = appContext.getExternalFilesDir(MODELS_DIR)

    /**
     * A file stored beside [modelInfo] with the same base name and [extension]
     * (e.g. the GPU `.tflite` exported next to a VR `.ptl`), or null if absent.
     * Asset companions are cached to disk like [resolveModelFile].
     */
    fun resolveCompanionFile(modelInfo: ModelInfo, extension: String): File? {
        val name = modelInfo.fileName.substringBeforeLast('.') + ".$extension"
        return when (modelInfo.source) {
            ModelSource.FILE -> File(modelInfo.filePath!!).resolveSibling(name).takeIf { it.isFile }
            ModelSource.ASSET -> {
                val exists = runCatching { appContext.assets.list(MODELS_DIR)?.contains(name) == true }
                    .getOrDefault(false)
                if (exists) cacheAsset(name, assetLength(name)) else null
            }
        }
    }

    /** Scan all three locations for .onnx and .ptl model files. */
    fun scanModels() {
        availableModels.clear()

        // 1. Bundled assets — must use AssetManager, not File.listFiles()
        try {
            val assetFiles = appContext.assets.list(MODELS_DIR) ?: emptyArray()
            for (fileName in assetFiles) {
                val fmt = ModelFormat.fromFileName(fileName) ?: continue
                val size = assetLength(fileName)
                val meta = if (fmt == ModelFormat.PTL) readAssetMeta(fileName) else null
                val mdx = if (fmt == ModelFormat.ONNX) mdxMeta("asset:$fileName", size) {
                    appContext.assets.open("$MODELS_DIR/$fileName")
                } else null
                val demucs = if (fmt == ModelFormat.DEMUCS) runCatching {
                    DemucsModelMeta.parse(appContext.assets.open("$MODELS_DIR/$fileName").bufferedReader().readText())
                }.getOrNull() else null
                val missing = demucs?.models?.map { it.file }?.filter { it !in assetFiles }.orEmpty()
                availableModels.add(
                    ModelInfo.create(fileName, fmt, ModelSource.ASSET, null, size, meta, mdx, demucs, missing)
                )
                Log.d(TAG, "Found asset model: $fileName ($fmt)")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to list assets/$MODELS_DIR: ${e.message}")
        }

        // 2. Internal files/models/ and 3. external storage
        for (dir in listOfNotNull(getModelsDir(), getExternalModelsDir())) {
            dir.listFiles()?.sortedBy { it.name }?.forEach { file ->
                val fmt = ModelFormat.fromFileName(file.name) ?: return@forEach
                val meta = if (fmt == ModelFormat.PTL) VrModelMeta.readFromPtl(file) else null
                val demucs = if (fmt == ModelFormat.DEMUCS) runCatching {
                    DemucsModelMeta.parse(file.readText())
                }.getOrNull() else null
                // A Demucs sub-model .onnx belongs to its manifest, not a model of its own
                if (fmt == ModelFormat.ONNX && isDemucsPart(file)) return@forEach
                val mdx = if (fmt == ModelFormat.ONNX) mdxMeta(file.absolutePath, file.length()) {
                    file.inputStream()
                } else null
                val missing = demucs?.models?.map { it.file }?.filter { !file.resolveSibling(it).isFile }.orEmpty()
                val size = file.length() + (demucs?.models?.sumOf { file.resolveSibling(it.file).length() } ?: 0L)
                availableModels.add(
                    ModelInfo.create(file.name, fmt, ModelSource.FILE, file.absolutePath, size, meta, mdx, demucs, missing)
                )
                Log.d(TAG, "Found model: ${file.absolutePath} ($fmt)")
            }
        }

        Log.i(TAG, "Total models found: ${availableModels.size} " +
                "(${availableModels.count { it.format == ModelFormat.ONNX }} ONNX, " +
                "${availableModels.count { it.format == ModelFormat.PTL }} PTL, " +
                "${availableModels.count { it.format == ModelFormat.DEMUCS }} Demucs)")
    }

    /** True if a `.demucs` manifest in [file]'s folder lists it as a sub-model. */
    private fun isDemucsPart(file: File): Boolean =
        file.parentFile?.listFiles { f -> f.name.endsWith(".${DemucsModelMeta.EXTENSION}") }.orEmpty().any { m ->
            runCatching { DemucsModelMeta.parse(m.readText()).models.any { it.file == file.name } }.getOrDefault(false)
        }

    /** Another file stored next to [modelInfo] (e.g. a Demucs sub-model), cached to disk for assets. */
    fun resolveSiblingFile(modelInfo: ModelInfo, name: String): File? = when (modelInfo.source) {
        ModelSource.FILE -> File(modelInfo.filePath!!).resolveSibling(name).takeIf { it.isFile }
        ModelSource.ASSET -> runCatching { cacheAsset(name, assetLength(name)) }.getOrNull()
    }

    /**
     * UVR's MDX-Net settings for an .onnx, by model hash. Hashing reads 10 MB,
     * so hashes are remembered per file (path + size + mtime / app install).
     */
    private fun mdxMeta(key: String, size: Long, open: () -> java.io.InputStream): MdxModelMeta? = runCatching {
        val stamp = if (key.startsWith("asset:")) apkUpdateTime() else File(key).lastModified()
        val cacheKey = "$key|$size|$stamp"
        val prefs = appContext.getSharedPreferences("model_hashes", Context.MODE_PRIVATE)
        val hash = prefs.getString(cacheKey, null)
            ?: MdxModelMeta.modelHash(size, open).also { prefs.edit().putString(cacheKey, it).apply() }
        MdxModelMeta.forHash(appContext, hash)
    }.onFailure { Log.w(TAG, "MDX lookup failed for $key: ${it.message}") }.getOrNull()

    private fun apkUpdateTime(): Long =
        appContext.packageManager.getPackageInfo(appContext.packageName, 0).lastUpdateTime

    /**
     * Copy a model file picked by the user into internal storage and rescan.
     * Blocking — call off the main thread. Returns the imported model.
     */
    fun importModel(uri: Uri): ModelInfo {
        val name = FileUtils.displayName(appContext, uri)
        val companionKind = COMPANIONS.entries.firstOrNull { name.endsWith(".${it.key}", ignoreCase = true) }?.value
        val isCompanion = companionKind != null
        require(ModelFormat.fromFileName(name) != null || isCompanion) {
            "“$name” isn't a model file — choose a .ptl, .onnx or .demucs file"
        }
        val dest = File(getModelsDir(), name)
        val tmp = File(getModelsDir(), "$name.part")
        try {
            appContext.contentResolver.openInputStream(uri)?.use { input ->
                tmp.outputStream().use { input.copyTo(it) }
            } ?: throw IllegalStateException("Couldn't open “$name”")
            if (!tmp.renameTo(dest)) throw IllegalStateException("Couldn't save “$name”")
        } finally {
            tmp.delete()
        }
        scanModels()
        // A .tflite (GPU) or .mnn (MNN CPU) is another version of the .ptl (VR) or .onnx (MDX-Net) with the
        // same name, not a model of its own
        val base = dest.name.substringBeforeLast('.')
        val owners = if (isCompanion) listOf(ModelFormat.PTL, ModelFormat.ONNX).map { "$base.${it.extension}" } else listOf(dest.name)
        return owners.firstNotNullOfOrNull { owner -> availableModels.firstOrNull { it.filePath == File(getModelsDir(), owner).absolutePath } }
            // A Demucs sub-model: its manifest is the model
            ?: availableModels.firstOrNull { m -> m.demucsMeta?.models?.any { it.file == name } == true }
            ?: throw IllegalStateException(
                if (isCompanion) "saved $name ($companionKind version) — import ${owners.joinToString(" or ")} to use it"
                else "saved $name — import the .demucs file it belongs to"
            )
    }

    /** Delete an imported/pushed model file. Bundled (asset) models can't be deleted. */
    fun deleteModel(model: ModelInfo): Boolean {
        if (model.source != ModelSource.FILE) return false
        val deleted = File(model.filePath!!).delete()
        COMPANIONS.keys.forEach { resolveCompanionFile(model, it)?.delete() }
        model.demucsMeta?.models?.forEach { File(model.filePath).resolveSibling(it.file).delete() }
        scanModels()
        return deleted
    }

    /** Embedded VR metadata of a bundled .ptl (stored uncompressed, see aaptOptions.noCompress). */
    private fun readAssetMeta(fileName: String): VrModelMeta? = runCatching {
        appContext.assets.openFd("$MODELS_DIR/$fileName").use { fd ->
            FileInputStream(fd.fileDescriptor).use { stream ->
                VrModelMeta.readFromPtl(stream.channel, fd.startOffset, fd.length)
            }
        }
    }.getOrNull()

    private fun assetLength(fileName: String): Long = runCatching {
        appContext.assets.openFd("$MODELS_DIR/$fileName").use { it.length }
    }.getOrDefault(0L)

    /**
     * Resolve a model to an absolute file path suitable for [Module.load()].
     *
     * Works for both ONNX and PTL:
     *  - FILE source → returns [filePath] directly (already on filesystem)
     *  - ASSET source → copies the asset to `cacheDir/models/<fileName>`, reusing
     *    that copy while it still matches the asset. It is replaced when its size
     *    differs or it predates the installed APK — an app update can ship a
     *    re-exported model under the same name.
     */
    fun resolveModelFile(modelInfo: ModelInfo): File? {
        return try {
            when (modelInfo.source) {
                ModelSource.FILE -> File(modelInfo.filePath!!)
                ModelSource.ASSET -> cacheAsset(modelInfo.fileName, modelInfo.fileSizeBytes)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to resolve model file: ${modelInfo.fileName}", e)
            null
        }
    }

    /** Copy asset models/[fileName] to `cacheDir/models/`, reusing a copy that still matches it. */
    private fun cacheAsset(fileName: String, size: Long): File {
        val cacheFile = File(appContext.cacheDir, "$MODELS_DIR/$fileName")
        val apkUpdated = appContext.packageManager
            .getPackageInfo(appContext.packageName, 0).lastUpdateTime
        val fresh = cacheFile.exists() &&
            cacheFile.length() == size &&
            cacheFile.lastModified() >= apkUpdated
        if (!fresh) {
            cacheFile.parentFile?.mkdirs()
            val tmp = File(cacheFile.path + ".part")
            appContext.assets
                .open("$MODELS_DIR/$fileName")
                .use { input -> tmp.outputStream().use { input.copyTo(it) } }
            if (!tmp.renameTo(cacheFile)) throw IllegalStateException("Couldn't cache $fileName")
            Log.i(TAG, "Cached asset model to: ${cacheFile.absolutePath}")
        }
        return cacheFile
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

enum class ModelSource { ASSET, FILE }

/** The on-disk format of a model file. */
enum class ModelFormat(val extension: String, val label: String) {
    ONNX("onnx", "ONNX"),
    PTL("ptl", "PTL"),
    /** Demucs v4 manifest listing its ONNX sub-models. */
    DEMUCS(DemucsModelMeta.EXTENSION, "ONNX");

    companion object {
        /** Returns the format for [fileName] based on extension, or null if unsupported. */
        fun fromFileName(fileName: String): ModelFormat? =
            values().firstOrNull { fileName.endsWith(".${it.extension}", ignoreCase = true) }
    }
}

/**
 * Metadata for a single model file.
 * Format-agnostic — works for both .onnx and .ptl files.
 */
data class ModelInfo(
    val fileName: String,
    val displayName: String,
    val architectureType: ArchitectureType,
    val format: ModelFormat,
    val source: ModelSource,
    val filePath: String? = null,   // non-null when source == FILE
    val fileSizeBytes: Long = 0L,
    // Stems the model outputs, from the metadata embedded by convert.ipynb (VR .ptl)
    val primaryStem: String? = null,
    val secondaryStem: String? = null,
    /** UVR's settings for an MDX-Net .onnx (found by model hash). */
    val mdxMeta: MdxModelMeta? = null,
    /** A Demucs v4 manifest's contents. */
    val demucsMeta: DemucsModelMeta? = null,
    /** Why this model can't be used yet, or null if it can. */
    val unsupportedReason: String? = null
) {
    val isSupported: Boolean get() = unsupportedReason == null

    companion object {
        fun create(
            fileName: String, format: ModelFormat, source: ModelSource,
            filePath: String?, size: Long, meta: VrModelMeta?, mdx: MdxModelMeta? = null,
            demucs: DemucsModelMeta? = null, missingFiles: List<String> = emptyList()
        ): ModelInfo {
            val arch = when {
                meta != null -> ArchitectureType.VR_ARCH
                mdx != null -> ArchitectureType.MDX_NET
                format == ModelFormat.DEMUCS -> ArchitectureType.DEMUCS
                else -> detectArchitecture(fileName)
            }
            val reason = when {
                format == ModelFormat.DEMUCS -> when {
                    demucs == null -> "Not a Demucs v4 manifest — re-export it with convert.ipynb"
                    missingFiles.isNotEmpty() -> "Missing ${missingFiles.joinToString()} — import it next to the .demucs file"
                    else -> null
                }
                format == ModelFormat.ONNX -> if (mdx != null) null
                    else "Not a UVR MDX-Net model (its hash isn't in UVR's model list)"
                meta == null ->
                    "No UVR settings inside — re-export it with convert.ipynb"
                meta.isVr51 ->
                    "VR 5.1 models aren't supported yet"
                else -> null
            }
            return ModelInfo(
                fileName = fileName,
                // Desktop UVR's name, e.g. "10_SP-UVR-2B-32000-1" (the converter adds "_vr")
                displayName = fileName.substringBeforeLast('.').removeSuffix("_vr"),
                architectureType = arch,
                format = format,
                source = source,
                filePath = filePath,
                fileSizeBytes = size,
                primaryStem = meta?.primaryStem ?: mdx?.primaryStem,
                secondaryStem = meta?.secondaryStem ?: mdx?.secondaryStem,
                mdxMeta = mdx,
                demucsMeta = demucs,
                unsupportedReason = reason
            )
        }

        /**
         * Heuristic architecture detection from model filename.
         * Mirrors the logic in UVR's constants.py MODEL_TYPE checks.
         */
        fun detectArchitecture(fileName: String): ArchitectureType {
            val lower = fileName.lowercase()
            return when {
                lower.contains("mdx") || lower.contains("htdemucs") -> ArchitectureType.MDX_NET
                lower.contains("demucs") -> ArchitectureType.DEMUCS
                else -> ArchitectureType.VR_ARCH
            }
        }
    }
}

enum class ArchitectureType(val label: String) {
    VR_ARCH("VR Arch"),    // Original VR Architecture (Spleeter-style U-Net)
    MDX_NET("MDX-Net"),    // MDX-Net / TFC-TDF
    DEMUCS("Demucs")       // Facebook Demucs (hybrid transformer)
}
