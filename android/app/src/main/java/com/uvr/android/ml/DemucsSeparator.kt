package com.uvr.android.ml

import android.util.Log
import com.uvr.android.audio.DecodedAudio
import com.uvr.android.ml.demucs.DemucsCore
import com.uvr.android.ml.demucs.DemucsModelMeta
import com.uvr.android.ml.demucs.DemucsRunner
import com.uvr.android.ml.vr.StereoSink
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import java.io.File
import kotlin.coroutines.coroutineContext

/**
 * Demucs v4 separator — desktop UVR's SeperateDemucs ([DemucsRunner]) streaming
 * from the decoded 44.1 kHz PCM file. Each sub-model's ONNX core runs on ONNX
 * Runtime (CPU, FP32) and is loaded only while it runs.
 *
 * Outputs every source (UVR "All Stems"), or with [SeparationParams.demucsStem]
 * that stem plus "No <stem>" = mix − stem, as desktop does.
 */
class DemucsSeparator(model: ModelInfo, params: SeparationParams) : Separator(model, params) {

    /** Sessions are opened per sub-model in [runSeparation]; nothing to preload. */
    override fun buildEngine(): InferenceEngine = object : InferenceEngine {
        override fun runBatch(data: FloatArray, shape: LongArray) = error("unused")
        override val inputNames = emptySet<String>()
        override val outputNames = emptySet<String>()
        override fun expectedInputShape(inputName: String): LongArray? = null
        override fun expectedInputRank(inputName: String) = -1
        override fun close() {}
    }

    override suspend fun runSeparation(
        audio: DecodedAudio,
        engine: InferenceEngine,
        cacheDir: File,
        onProgress: (Float) -> Unit
    ): SeparationResult {
        val meta = model.demucsMeta ?: throw IllegalStateException("${model.fileName} isn't a Demucs v4 model")
        require(audio.channels == 2 && audio.sampleRate == meta.sampleRate) {
            "Demucs expects ${meta.sampleRate} Hz stereo input (got ${audio.sampleRate} Hz × ${audio.channels})"
        }
        val subFiles = meta.models.map {
            ModelManager.resolveSiblingFile(model, it.file) ?: throw IllegalStateException("Missing ${it.file}")
        }
        val stems = meta.uvrStems
        val chosen = params.demucsStem?.takeIf { it in stems }
        Log.i(tag, "Demucs ${meta.name}: stems=$stems chosen=$chosen overlap=${params.demucsOverlap} shifts=${params.demucsShifts}")

        val job = coroutineContext[Job]
        val outFiles = stems.map { File.createTempFile("uvr_demucs_${it}_", ".raw", cacheDir) }
        var open: OnnxInferenceEngine? = null
        try {
            RawStereoSource(audio.pcmFile).use { source ->
                val sinks = outFiles.map { RawStereoSink(it) }
                try {
                    var modelMs = 0L
                    var calls = 0
                    val t0 = System.nanoTime()
                    DemucsRunner.separate(
                        meta = meta,
                        source = source,
                        cores = { i ->
                            open?.close()
                            val session = OnnxInferenceEngine(subFiles[i], lowMemory = true).also { open = it }
                            DemucsCore { mix, mag ->
                                val t = System.nanoTime()
                                val out = session.run(mapOf(
                                    "mix" to (mix to longArrayOf(1, 2, meta.segmentSamples.toLong())),
                                    "mag" to (mag to longArrayOf(1, 4, meta.nFft / 2L, mag.size / (4L * meta.nFft / 2)))
                                ))
                                modelMs += (System.nanoTime() - t) / 1_000_000
                                Log.d(tag, "model call ${++calls}: avg ${modelMs / calls} ms")
                                out[0] to out[1]
                            }
                        },
                        overlap = params.demucsOverlap,
                        shifts = params.demucsShifts,
                        sinks = sinks,
                        tempDir = cacheDir,
                        log = { Log.i(tag, it) }
                    ) { p ->
                        if (job?.isActive == false) throw CancellationException("Separation cancelled")
                        onProgress(0.15f + 0.85f * p)
                    }
                    Log.i(tag, "Demucs done: $calls model calls, model $modelMs ms of " +
                            "${(System.nanoTime() - t0) / 1_000_000} ms total")
                } finally {
                    sinks.forEach { it.close() }
                }
            }
            if (chosen == null) {
                return SeparationResult(stems.zip(outFiles).map { (n, f) -> StemFile(n, f) }, meta.sampleRate)
            }
            // One stem + its complement: secondary = mix − stem (separate.py secondary_save)
            val primary = outFiles[stems.indexOf(chosen)]
            val secondary = File.createTempFile("uvr_demucs_no_", ".raw", cacheDir)
            subtract(audio.pcmFile, primary, secondary)
            outFiles.filter { it != primary }.forEach { it.delete() }
            val noName = com.uvr.android.ml.mdx.MdxModelMeta.secondaryStemOf(chosen)
            return SeparationResult(listOf(StemFile(chosen, primary), StemFile(noName, secondary)), meta.sampleRate)
        } catch (e: Throwable) {
            outFiles.forEach { it.delete() }
            throw e
        } finally {
            open?.close()
        }
    }

    private fun subtract(mixFile: File, stemFile: File, out: File) {
        RawStereoSource(mixFile).use { mix ->
            RawStereoSource(stemFile).use { stem ->
                RawStereoSink(out).use { sink: StereoSink ->
                    val block = 1 shl 16
                    val ml = FloatArray(block); val mr = FloatArray(block)
                    val sl = FloatArray(block); val sr = FloatArray(block)
                    var pos = 0L
                    while (pos < mix.length) {
                        val n = minOf(block.toLong(), mix.length - pos).toInt()
                        mix.read(pos, ml, mr); stem.read(pos, sl, sr)
                        for (i in 0 until n) { ml[i] = -sl[i] + ml[i]; mr[i] = -sr[i] + mr[i] }
                        sink.write(ml, mr, n)
                        pos += n
                    }
                }
            }
        }
    }
}
