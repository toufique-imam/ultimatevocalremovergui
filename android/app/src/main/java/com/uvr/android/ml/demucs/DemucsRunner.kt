package com.uvr.android.ml.demucs

import com.uvr.android.ml.vr.StereoSink
import com.uvr.android.ml.vr.StereoSource
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The whole of desktop UVR's demix_demucs for a (bag of) Demucs v4 model(s):
 * normalise the mix, run every sub-model ([DemucsPipeline]), combine the bag
 * (Σ weight·estimate / Σ weight per source), de-normalise, and write the stems
 * in UVR's order ([DemucsModelMeta.uvrStems]) to [sinks].
 *
 * A single-model bag streams straight to the sinks. For several models each one's
 * weighted stems go to temp files in [tempDir] first, then a final pass combines them.
 */
object DemucsRunner {

    fun separate(
        meta: DemucsModelMeta,
        source: StereoSource,
        cores: (Int) -> DemucsCore,
        overlap: Float,
        shifts: Int,
        sinks: List<StereoSink>,
        tempDir: File,
        log: (String) -> Unit = {},
        onProgress: (Float) -> Unit = {}
    ) {
        require(sinks.size == meta.sources.size) { "Need ${meta.sources.size} sinks" }
        val (mean, std) = DemucsPipeline.normalization(source)
        log("Demucs ${meta.name}: ${meta.models.size} model(s), ref mean=$mean std=$std")
        val normalized = NormalizedSource(source, mean, std)
        val nStems = meta.sources.size
        val totals = FloatArray(nStems) { k -> meta.models.sumOf { it.weights[k].toDouble() }.toFloat() }
        val pipeline = DemucsPipeline(meta, overlap, shifts, log)

        /** estimates for model-order stems → (÷ totals) × std + mean → UVR-order sinks. */
        fun finish(est: Array<Array<FloatArray>>, count: Int) {
            for (i in 0 until nStems) {
                val k = meta.uvrToModelIndex(i)
                val l = est[k][0]; val r = est[k][1]
                for (j in 0 until count) {
                    l[j] = l[j] / totals[k] * std + mean
                    r[j] = r[j] / totals[k] * std + mean
                }
                sinks[i].write(l, r, count)
            }
        }

        if (meta.models.size == 1) {
            val w = meta.models[0].weights
            pipeline.run(normalized, cores(0), { stems, count ->
                for (k in 0 until nStems) for (c in 0 until 2) {
                    val a = stems[k][c]
                    for (j in 0 until count) a[j] *= w[k]
                }
                finish(stems, count)
            }, onProgress)
            return
        }

        // Bag: each model's weighted, non-zero stems to temp files
        val files = Array(meta.models.size) { arrayOfNulls<File>(nStems) }
        try {
            for ((m, sub) in meta.models.withIndex()) {
                val outs = (0 until nStems).map { k ->
                    if (sub.weights[k] == 0f) null
                    else File.createTempFile("uvr_demucs_m${m}_s$k", ".raw", tempDir).also { files[m][k] = it }
                        .let { DataSink(it) }
                }
                try {
                    pipeline.run(normalized, cores(m), { stems, count ->
                        for (k in 0 until nStems) outs[k]?.let { o ->
                            val a = stems[k]
                            for (c in 0 until 2) for (j in 0 until count) a[c][j] *= sub.weights[k]
                            o.write(a[0], a[1], count)
                        }
                    }) { p -> onProgress((m + p) / meta.models.size) }
                } finally {
                    outs.forEach { it?.close() }
                }
            }
            // estimates = Σ_m weighted_m, in model order (desktop adds all models; zero-weight terms add 0)
            val block = 1 shl 16
            val readers = files.map { row -> row.map { f -> f?.let { DataSource(it) } } }
            try {
                var remaining = source.length
                while (remaining > 0) {
                    val count = minOf(block.toLong(), remaining).toInt()
                    val est = Array(nStems) { Array(2) { FloatArray(count) } }
                    for (row in readers) for (k in 0 until nStems) row[k]?.readInto(est[k], count)
                    finish(est, count)
                    remaining -= count
                }
            } finally {
                readers.flatten().forEach { it?.close() }
            }
        } finally {
            files.flatten().forEach { it?.delete() }
        }
    }

    /** (x − mean) / std inside the song, 0 outside (desktop pads after normalising). */
    private class NormalizedSource(
        private val raw: StereoSource, private val mean: Float, private val std: Float
    ) : StereoSource {
        override val length get() = raw.length
        override fun read(start: Long, left: FloatArray, right: FloatArray) {
            raw.read(start, left, right)
            for (i in left.indices) {
                val s = start + i
                if (s in 0 until length) {
                    left[i] = (left[i] - mean) / std
                    right[i] = (right[i] - mean) / std
                } else {
                    left[i] = 0f; right[i] = 0f
                }
            }
        }
    }

    /** Interleaved stereo f32 temp file writer. */
    private class DataSink(file: File) : AutoCloseable {
        private val out = BufferedOutputStream(FileOutputStream(file), 1 shl 16)
        private val bb = ByteBuffer.allocate(8 * 4096).order(ByteOrder.LITTLE_ENDIAN)
        fun write(l: FloatArray, r: FloatArray, count: Int) {
            for (i in 0 until count) {
                if (!bb.hasRemaining()) flush()
                bb.putFloat(l[i]).putFloat(r[i])
            }
            flush()
        }
        private fun flush() { out.write(bb.array(), 0, bb.position()); bb.clear() }
        override fun close() = out.close()
    }

    /** Sequential reader for [DataSink] files, adding into the destination. */
    private class DataSource(file: File) : AutoCloseable {
        private val input = DataInputStream(BufferedInputStream(FileInputStream(file), 1 shl 16))
        private val bytes = ByteArray(8 * 4096)
        fun readInto(dst: Array<FloatArray>, count: Int) {
            var i = 0
            while (i < count) {
                val n = minOf(4096, count - i)
                input.readFully(bytes, 0, n * 8)
                val bb = ByteBuffer.wrap(bytes, 0, n * 8).order(ByteOrder.LITTLE_ENDIAN)
                for (j in 0 until n) { dst[0][i + j] += bb.float; dst[1][i + j] += bb.float }
                i += n
            }
        }
        override fun close() = input.close()
    }
}
