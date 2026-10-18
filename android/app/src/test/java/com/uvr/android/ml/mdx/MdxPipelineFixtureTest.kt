package com.uvr.android.ml.mdx

import com.uvr.android.ml.vr.MaskModel
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
 * Compares [MdxPipeline] against desktop UVR (separate.SeperateMDX.demix) run
 * with a deterministic fake model. Each fixture dir holds mdx.json, input.f32
 * (44.1 kHz stereo interleaved) and the desktop's primary.f32 / secondary.f32.
 * Point UVR_MDX_FIXTURES at the parent directory; skipped when unset.
 */
class MdxPipelineFixtureTest {

    /** Same function as the fixture generator's fake_model: x·σ(2x + 0.3 − 0.001f + 0.01c + 0.2 sin(0.05t)). */
    private val fakeModel = MaskModel { input, shape ->
        val ch = shape[1].toInt(); val bins = shape[2].toInt(); val frames = shape[3].toInt()
        FloatArray(input.size) { i ->
            val c = i / (bins * frames)
            val f = (i / frames) % bins
            val t = i % frames
            val x = input[i]
            x * (1f / (1f + exp(-(2f * x + 0.3f - 0.001f * f + 0.01f * c + 0.2f * sin(0.05f * t)))))
        }.also { check(ch == 4) }
    }

    @Test
    fun matchesDesktopWithFakeModel() {
        val root = System.getenv("UVR_MDX_FIXTURES")
        assumeTrue("UVR_MDX_FIXTURES not set", root != null)
        val dirs = File(root!!).listFiles()!!.filter { File(it, "mdx.json").exists() }.sorted()
        assertTrue("no fixtures in $root", dirs.isNotEmpty())
        for (dir in dirs) check(dir)
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    private val failures = mutableListOf<String>()

    private fun check(dir: File) {
        val json = File(dir, "mdx.json").readText()
        fun num(key: String) = Regex("\"$key\": ([0-9.]+)").find(json)!!.groupValues[1]
        val overlap = Regex("\"overlap\": \"([^\"]+)\"").find(json)!!.groupValues[1]
        val pipeline = MdxPipeline(
            nFft = num("n_fft").toInt(),
            dimF = num("dim_f").toInt(),
            dimT = num("dim_t").toInt(),
            compensate = num("compensate").toFloat(),
            overlap = if (overlap == "default") null else overlap.toFloat(),
            denoise = json.contains("\"denoise\": true")
        )
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
        val prim = CollectingSink()
        val sec = CollectingSink()
        val t0 = System.currentTimeMillis()
        pipeline.run(source, fakeModel, prim, sec)
        val ms = System.currentTimeMillis() - t0

        for ((name, sink) in listOf("primary" to prim, "secondary" to sec)) {
            val ref = readF32(File(dir, "$name.f32"))
            val out = sink.interleaved()
            assertTrue("${dir.name} $name length ${out.size} ≠ ${ref.size}", out.size == ref.size)
            var err = 0.0; var sig = 0.0; var maxAbs = 0.0
            for (i in ref.indices) {
                val e = (out[i] - ref[i]).toDouble()
                err += e * e; sig += ref[i].toDouble() * ref[i]
                maxAbs = maxOf(maxAbs, kotlin.math.abs(e))
            }
            val snr = 10 * log10(sig / maxOf(err, 1e-30))
            val firstBad = ref.indices.firstOrNull { kotlin.math.abs(out[it] - ref[it]) > 1e-3 }
            if (firstBad != null) println("  first |err|>1e-3 at frame ${firstBad / 2}: out=${out[firstBad]} ref=${ref[firstBad]}")
            println("%-34s %-9s SNR=%.1f dB  max|err|=%.2e  (%d ms)".format(dir.name, name, snr, maxAbs, ms))
            if (snr <= 60) failures += "${dir.name} $name SNR $snr"
        }
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
