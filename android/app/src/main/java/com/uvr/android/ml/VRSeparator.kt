package com.uvr.android.ml

import android.util.Log
import com.uvr.android.audio.DecodedAudio
import com.uvr.android.ml.vr.MaskModel
import com.uvr.android.ml.vr.StereoSink
import com.uvr.android.ml.vr.StereoSource
import com.uvr.android.ml.vr.VrModelMeta
import com.uvr.android.ml.vr.VrPipeline
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import java.io.File
import kotlin.coroutines.coroutineContext

/**
 * VR Architecture separator — desktop UVR's SeperateVR pipeline ([VrPipeline])
 * streaming from the decoded PCM file to two raw stem files.
 *
 * Heap use is bounded by the pipeline's segment size (~12 s of audio),
 * so input length is limited only by storage.
 */
class VRSeparator(model: ModelInfo, params: SeparationParams) : Separator(model, params) {

    /**
     * GPU (.tflite beside the .ptl, FP32) when available, built for this window
     * size, and — checked on the first window — as accurate as and faster than
     * the .ptl on CPU; otherwise the .ptl alone.
     */
    override fun buildEngine(): InferenceEngine {
        val cpu = { super.buildEngine() }
        if (!params.useGpu) return VerifiedEngine(null, cpu, tag = tag)
        val tflite = ModelManager.resolveCompanionFile(model, "tflite")
            ?: return VerifiedEngine(null, cpu, tag = tag).also { Log.i(tag, "No .tflite for ${model.fileName} — CPU") }
        val key = "${tflite.absolutePath}:${tflite.length()}:${tflite.lastModified()}:${params.windowSize}"
        if (VerifiedEngine.decision(key) == false) {
            Log.i(tag, "GPU was slower or inaccurate for this model earlier — CPU")
            return VerifiedEngine(null, cpu, tag = tag)
        }
        val meta = ModelManager.resolveModelFile(model)?.let { VrModelMeta.readFromPtl(it) }
        val gpu = runCatching { TfliteGpuEngine(tflite) }
            .onFailure { Log.w(tag, "GPU engine unavailable, using CPU: ${it.message}") }
            .getOrNull()
            ?.takeIf { engine ->
                val shape = longArrayOf(1, 2, (meta?.params?.bins ?: -1) + 1L, params.windowSize.toLong())
                engine.accepts(shape).also {
                    if (!it) { Log.i(tag, "GPU model is built for ${engine.inputShape.toList()}, not $shape — CPU"); engine.close() }
                }
            }
        return VerifiedEngine(gpu, cpu, decisionKey = key, tag = tag)
    }

    override suspend fun runSeparation(
        audio: DecodedAudio,
        engine: InferenceEngine,
        cacheDir: File,
        onProgress: (Float) -> Unit
    ): SeparationResult {
        require(audio.channels == 2) { "VR separation expects stereo input" }
        val modelFile = ModelManager.resolveModelFile(model)
            ?: throw IllegalStateException("Could not resolve model file: ${model.fileName}")
        val meta = VrModelMeta.readFromPtl(modelFile)
            ?: throw IllegalStateException(
                "${model.fileName} has no uvr_meta.json — re-export it with convert.ipynb")

        val pipeline = VrPipeline(
            meta = meta,
            inputSampleRate = audio.sampleRate,
            windowSize = params.windowSize,
            aggression = params.aggressiveness,
            tta = params.tta,
            log = { Log.i(tag, it) }
        )
        Log.i(tag, "VR params=${meta.paramName} bands=${meta.params.bands.size} " +
                "primary=${meta.primaryStem} window=${params.windowSize} aggression=${params.aggressiveness} tta=${params.tta}")

        val job = coroutineContext[Job]
        val primaryOut = File.createTempFile("uvr_vr_primary", ".raw", cacheDir)
        val secondaryOut = File.createTempFile("uvr_vr_secondary", ".raw", cacheDir)
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
                                if (++calls % 10 == 0) Log.d(tag, "model window $calls: avg ${modelMs / calls} ms " +
                                        "(${(engine as? VerifiedEngine)?.backend ?: "cpu"})")
                            }
                        }
                        val t0 = System.nanoTime()
                        pipeline.run(source, timedModel, prim, sec) { p ->
                            if (job?.isActive == false) throw CancellationException("Separation cancelled")
                            onProgress(0.15f + 0.85f * p)
                        }
                        val totalMs = (System.nanoTime() - t0) / 1_000_000
                        Log.i(tag, "VR done: $calls windows on ${(engine as? VerifiedEngine)?.backend ?: "cpu"}, " +
                                "model ${modelMs} ms of ${totalMs} ms total")
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
            sampleRate = pipeline.outputSampleRate
        )
    }
}
