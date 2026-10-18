package com.uvr.android.ml

import android.util.Log
import com.uvr.android.audio.DecodedAudio
import com.uvr.android.ml.mdx.MdxPipeline
import com.uvr.android.ml.vr.MaskModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import java.io.File
import kotlin.coroutines.coroutineContext

/**
 * MDX-Net separator — desktop UVR's SeperateMDX ([MdxPipeline]) streaming from
 * the decoded 44.1 kHz PCM file to two raw stem files, with the model on
 * ONNX Runtime (CPU, FP32) or on the GPU.
 */
class MDXSeparator(model: ModelInfo, params: SeparationParams) : Separator(model, params) {

    /**
     * "Exact" ([SeparationParams.mdxFastGpu] off): ONNX Runtime on the CPU, as desktop UVR.
     * "Fast": GPU (.tflite beside the .onnx, FP16) when available and — checked on the first
     * non-silent chunk — close enough to and faster than ONNX Runtime on CPU; otherwise the CPU.
     * FP32 on the GPU is no faster than the CPU for MDX-Net, so the check is by level
     * ([MIN_GPU_SNR_DB]) instead of the per-value FP32 tolerance VR uses.
     */
    override fun buildEngine(): InferenceEngine {
        val file = ModelManager.resolveModelFile(model)
            ?: throw IllegalStateException("Could not resolve model file: ${model.fileName}")
        val cpu = { OnnxInferenceEngine(file, lowMemory = true) }
        if (!params.mdxFastGpu) return VerifiedEngine(null, cpu, tag = tag)
        val meta = model.mdxMeta
        val tflite = ModelManager.resolveCompanionFile(model, "tflite")
        if (tflite == null || meta == null) {
            Log.i(tag, "No .tflite for ${model.fileName} — CPU")
            return VerifiedEngine(null, cpu, tag = tag)
        }
        val key = "${tflite.absolutePath}:${tflite.length()}:${tflite.lastModified()}"
        if (VerifiedEngine.decision(key) == false) {
            Log.i(tag, "GPU was slower or inaccurate for this model earlier — CPU")
            return VerifiedEngine(null, cpu, tag = tag)
        }
        val gpu = runCatching { TfliteGpuEngine(tflite, fp16 = true) }
            .onFailure { Log.w(tag, "GPU engine unavailable, using CPU: ${it.message}") }
            .getOrNull()
            ?.takeIf { engine ->
                val shape = longArrayOf(1, 4, meta.dimF.toLong(), meta.dimT.toLong())
                engine.accepts(shape).also {
                    if (!it) { Log.i(tag, "GPU model is built for ${engine.inputShape.toList()}, not $shape — CPU"); engine.close() }
                }
            }
        return VerifiedEngine(gpu, cpu, decisionKey = key, minSnrDb = MIN_GPU_SNR_DB, minLevel = MIN_CHECK_LEVEL, tag = tag)
    }

    override suspend fun runSeparation(
        audio: DecodedAudio,
        engine: InferenceEngine,
        cacheDir: File,
        onProgress: (Float) -> Unit
    ): SeparationResult {
        require(audio.channels == 2 && audio.sampleRate == 44100) {
            "MDX-Net expects 44.1 kHz stereo input (got ${audio.sampleRate} Hz × ${audio.channels})"
        }
        val meta = model.mdxMeta
            ?: throw IllegalStateException("${model.fileName} isn't a known UVR MDX-Net model")
        val pipeline = MdxPipeline(
            nFft = meta.nFft,
            dimF = meta.dimF,
            dimT = meta.dimT,
            compensate = meta.compensate,
            overlap = params.mdxOverlap,
            denoise = params.mdxDenoise,
            log = { Log.i(tag, it) }
        )

        val job = coroutineContext[Job]
        val primaryOut = File.createTempFile("uvr_mdx_primary", ".raw", cacheDir)
        val secondaryOut = File.createTempFile("uvr_mdx_secondary", ".raw", cacheDir)
        try {
            RawStereoSource(audio.pcmFile).use { source ->
                RawStereoSink(primaryOut).use { prim ->
                    RawStereoSink(secondaryOut).use { sec ->
                        var calls = 0
                        var modelMs = 0L
                        val timedModel = MaskModel { input, shape ->
                            val t0 = System.nanoTime()
                            engine.runBatch(input, shape).also {
                                modelMs += (System.nanoTime() - t0) / 1_000_000
                                calls++
                                Log.d(tag, "model call $calls: avg ${modelMs / calls} ms")
                            }
                        }
                        val t0 = System.nanoTime()
                        pipeline.run(source, timedModel, prim, sec) { p ->
                            if (job?.isActive == false) throw CancellationException("Separation cancelled")
                            onProgress(0.15f + 0.85f * p)
                        }
                        Log.i(tag, "MDX done: $calls model calls, model $modelMs ms of " +
                                "${(System.nanoTime() - t0) / 1_000_000} ms total")
                    }
                }
            }
        } catch (e: Throwable) {
            primaryOut.delete()
            secondaryOut.delete()
            throw e
        }

        return SeparationResult(
            stems = listOf(
                StemFile(params.primaryStemName, primaryOut),
                StemFile(params.secondaryStemName, secondaryOut)
            ),
            sampleRate = 44100
        )
    }

    companion object {
        /**
         * Lowest GPU-vs-CPU output SNR (dB) on the checked chunk for the GPU to be used. FP16 on an
         * Adreno 650 gives ~41 dB per chunk and moves SDR by under 0.01 dB (MUSDB18, 10 songs), as
         * error 30 dB below the signal adds only ~0.04 dB to a separation that is 10–15 dB from the
         * truth; a broken driver lands near 0 dB.
         */
        const val MIN_GPU_SNR_DB = 30.0

        /**
         * Chunks whose CPU output RMS is below this (≈ −60 dBFS audio: |STFT| ≈ 54 × RMS for the
         * 7680-sample Hann window) aren't used for the check — FP16 error is relatively large in
         * near-silence and says nothing about the music.
         */
        const val MIN_CHECK_LEVEL = 0.05
    }
}
