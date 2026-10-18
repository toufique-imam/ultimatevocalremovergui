package com.uvr.android.ml.demucs

import com.uvr.android.ml.vr.StereoSink
import com.uvr.android.ml.vr.StereoSource
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.log10
import kotlin.math.sin

/**
 * Compares [DemucsRunner] against desktop UVR (separate.SeperateDemucs.demix_demucs
 * + demucs apply_model + HTDemucs's own STFT/iSTFT) with a fake network core and
 * fixed shift offsets. Each fixture dir holds model.demucs, demucs.json, input.f32
 * and the desktop's <Stem>.f32 files. Point UVR_DEMUCS_FIXTURES at the parent
 * directory; skipped when unset.
 */
class DemucsFixtureTest {

    /** Same as the generator's fake_core. */
    private fun fakeCore(stems: Int, bins: Int, frames: Int, tl: Int) = DemucsCore { mix, mag ->
        val spec = FloatArray(stems * 4 * bins * frames)
        for (s in 0 until stems) for (ch in 0 until 4) for (f in 0 until bins) for (t in 0 until frames) {
            val m = mag[(ch * bins + f) * frames + t]
            val g = 1f / (1f + exp(-(0.5f * m + 0.1f * s - 0.0003f * f + 0.1f * sin(0.07f * t))))
            spec[((s * 4 + ch) * bins + f) * frames + t] = m * g
        }
        val wave = FloatArray(stems * 2 * tl)
        for (s in 0 until stems) for (c in 0 until 2) for (n in 0 until tl) {
            wave[(s * 2 + c) * tl + n] = mix[c * tl + n] * (0.2f + 0.1f * s) * (1f + 0.1f * sin(0.001f * n))
        }
        spec to wave
    }

    @Test
    fun matchesDesktopWithFakeCore() {
        val root = System.getenv("UVR_DEMUCS_FIXTURES")
        assumeTrue("UVR_DEMUCS_FIXTURES not set", root != null)
        val dirs = File(root!!).listFiles()!!.filter { File(it, "model.demucs").exists() }.sorted()
        assertTrue("no fixtures in $root", dirs.isNotEmpty())
        val failures = mutableListOf<String>()
        for (dir in dirs) failures += check(dir)
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    private fun check(dir: File): List<String> {
        val meta = DemucsModelMeta.parse(File(dir, "model.demucs").readText())
        val opts = File(dir, "demucs.json").readText()
        val shifts = Regex("\"shifts\": (\\d+)").find(opts)!!.groupValues[1].toInt()
        val overlap = Regex("\"overlap\": ([0-9.]+)").find(opts)!!.groupValues[1].toFloat()
        val input = readF32(File(dir, "input.f32"))
        val source = object : StereoSource {
            override val length = input.size / 2L
            override fun read(start: Long, left: FloatArray, right: FloatArray) {
                for (i in left.indices) {
                    val s = start + i
                    left[i] = if (s in 0 until length) input[(2 * s).toInt()] else 0f
                    right[i] = if (s in 0 until length) input[(2 * s + 1).toInt()] else 0f
                }
            }
        }
        val frames = (meta.segmentSamples + meta.hop - 1) / meta.hop
        val core = fakeCore(meta.sources.size, meta.nFft / 2, frames, meta.segmentSamples)
        val sinks = meta.uvrStems.map { CollectingSink() }
        val t0 = System.currentTimeMillis()
        DemucsRunner.separate(meta, source, { core }, overlap, shifts, sinks, createTempDir())
        val ms = System.currentTimeMillis() - t0

        val failures = mutableListOf<String>()
        for ((i, name) in meta.uvrStems.withIndex()) {
            val ref = readF32(File(dir, "$name.f32"))
            val out = sinks[i].interleaved()
            if (out.size != ref.size) { failures += "${dir.name} $name length ${out.size} ≠ ${ref.size}"; continue }
            var err = 0.0; var sig = 0.0; var maxAbs = 0.0
            for (j in ref.indices) {
                val e = (out[j] - ref[j]).toDouble()
                err += e * e; sig += ref[j].toDouble() * ref[j]
                maxAbs = maxOf(maxAbs, kotlin.math.abs(e))
            }
            val snr = 10 * log10(sig / maxOf(err, 1e-30))
            println("%-24s %-7s SNR=%.1f dB  max|err|=%.2e  (%d ms)".format(dir.name, name, snr, maxAbs, ms))
            if (!(snr > 60)) failures += "${dir.name} $name SNR $snr"
        }
        return failures
    }

    private class CollectingSink : StereoSink {
        private val data = ArrayList<FloatArray>()
        override fun write(left: FloatArray, right: FloatArray, count: Int) {
            data += FloatArray(2 * count) { if (it % 2 == 0) left[it / 2] else right[it / 2] }
        }
        fun interleaved(): FloatArray {
            val out = FloatArray(data.sumOf { it.size })
            var o = 0
            for (d in data) { d.copyInto(out, o); o += d.size }
            return out
        }
    }

    private fun readF32(f: File): FloatArray {
        val bb = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bb.remaining() / 4) { bb.float }
    }
}
