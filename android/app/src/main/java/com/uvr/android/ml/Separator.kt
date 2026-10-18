package com.uvr.android.ml

import com.uvr.android.audio.DecodedAudio
import com.uvr.android.utils.BenchmarkUtils
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.system.measureTimeMillis

/**
 * Abstract base for audio separator implementations.
 *
 * The pipeline is fully streaming: audio stays on disk and separators read/write
 * chunks rather than loading the whole PCM into the Java heap.
 */
abstract class Separator(
    protected val model: ModelInfo,
    protected val params: SeparationParams
) {
    protected val tag = "Separator[${model.architectureType}/${model.format}]"

    /**
     * Run the full separation pipeline.
     *
     * @param audio    Decoded audio (file-backed — never a full FloatArray)
     * @param cacheDir Directory for intermediate temp files produced during separation
     * @param onProgress Progress callback in [0, 1]
     */
    suspend fun separate(
        audio: DecodedAudio,
        cacheDir: File,
        onProgress: (Float) -> Unit = {}
    ): SeparationResult = withContext(Dispatchers.Default) {
        val startTime = System.currentTimeMillis()
        Log.i(tag, "Starting separation — ${audio.totalSamples / audio.channels / audio.sampleRate}s, model: ${model.displayName}")
        onProgress(0.05f)

        val engine = buildEngine()
        System.gc()   // encourage reclaim of model ByteArray before separation begins
        onProgress(0.15f)

        try {
            for (name in engine.inputNames) {
                val shape = engine.expectedInputShape(name)
                Log.i(tag, "Input '$name' declared shape: ${shape?.toList() ?: "unknown (PTL)"}")
            }

            var result: SeparationResult
            val inferenceTime = measureTimeMillis {
                result = runSeparation(audio, engine, cacheDir, onProgress)
            }
            onProgress(1.0f)

            val totalTime = System.currentTimeMillis() - startTime
            val inputDurationSec = audio.totalSamples.toDouble() / audio.channels / audio.sampleRate
            val metrics = BenchmarkUtils.PerformanceMetrics(
                modelName        = model.displayName,
                totalTimeMs      = totalTime,
                inferenceTimeMs  = inferenceTime,
                processingTimeMs = totalTime - inferenceTime,
                inputDurationSec = inputDurationSec,
                speedFactor      = inputDurationSec / (totalTime / 1000.0)
            )
            metrics.log()

            result.copy(metrics = metrics)
        } finally {
            engine.close()
        }
    }

    protected abstract suspend fun runSeparation(
        audio: DecodedAudio,
        engine: InferenceEngine,
        cacheDir: File,
        onProgress: (Float) -> Unit
    ): SeparationResult

    protected open fun buildEngine(): InferenceEngine {
        return when (model.format) {
            ModelFormat.ONNX -> {
                val file = ModelManager.resolveModelFile(model)
                    ?: throw IllegalStateException("Could not resolve ONNX model: ${model.fileName}")
                OnnxInferenceEngine(file)
            }
            ModelFormat.PTL -> {
                if (!PtlInferenceEngine.isAvailable) throw PtlNotAvailableException()
                val file = ModelManager.resolveModelFile(model)
                    ?: throw IllegalStateException("Could not resolve PTL model: ${model.fileName}")
                PtlInferenceEngine(file.absolutePath)
            }
            ModelFormat.DEMUCS -> error("Demucs separators open their own sessions")
        }
    }

    companion object {
        fun create(model: ModelInfo, params: SeparationParams): Separator = when (model.architectureType) {
            ArchitectureType.VR_ARCH -> VRSeparator(model, params)
            ArchitectureType.MDX_NET -> MDXSeparator(model, params)
            ArchitectureType.DEMUCS  -> DemucsSeparator(model, params)
        }
    }
}

data class SeparationParams(
    val primaryStemName: String   = "Vocals",
    val secondaryStemName: String = "Instrumental",
    val overlap: Float            = 0.25f,
    val segmentSize: Int          = 256,
    val windowSize: Int           = SpectrogramUtils.DEFAULT_WINDOW_SIZE,
    val nFft: Int                 = SpectrogramUtils.DEFAULT_N_FFT,
    val hopLength: Int            = SpectrogramUtils.DEFAULT_HOP_LENGTH,
    val sampleRate: Int           = SpectrogramUtils.DEFAULT_SR,
    val aggressiveness: Float     = 0.05f,
    /**
     * VR: desktop UVR's "TTA" — average with a half-window-shifted pass (more accurate, 2× model time).
     * Off by default, as on desktop.
     */
    val tta: Boolean              = false,
    /** MDX-Net: UVR "Overlap" — null = "Default" (chunk − n_fft), else 0.25 … 0.99. */
    val mdxOverlap: Float?        = null,
    /** MDX-Net: UVR "Denoise Output" — average with the model run on −X (more accurate, 2× model time). */
    val mdxDenoise: Boolean       = false,
    /** Demucs: UVR "Overlap" between segments (0.25 … 0.99). */
    val demucsOverlap: Float      = 0.25f,
    /** Demucs: UVR "Shifts" — passes on the input shifted by up to 0.5 s, averaged (more accurate, N× time). */
    val demucsShifts: Int         = 2,
    /** Demucs: null = every stem (UVR "All Stems"), else that stem plus "No <stem>". */
    val demucsStem: String?       = null,
    /** VR: run the network on the GPU (FP32, verified against the CPU model) when a .tflite is available. */
    val useGpu: Boolean           = true,
    /**
     * MDX-Net "Fast": the GPU in half precision (FP16) when a .tflite is available and it is faster
     * than the CPU. Close to desktop UVR but not identical (audible to a careful listener on loud
     * vocals), so off by default: "Exact" runs ONNX Runtime on the CPU like desktop.
     */
    val mdxFastGpu: Boolean       = false,
    val postProcess: Boolean      = false,
    val invertSpectrogram: Boolean = false
)

/** One output stem: a raw interleaved f32le stereo PCM file. */
data class StemFile(val name: String, val file: File)

/**
 * Separation result — primary/secondary for VR and MDX-Net, every source for
 * Demucs. Delete the files after encoding to free space.
 */
data class SeparationResult(
    val stems: List<StemFile>,
    val sampleRate: Int,
    val metrics: BenchmarkUtils.PerformanceMetrics? = null
)
