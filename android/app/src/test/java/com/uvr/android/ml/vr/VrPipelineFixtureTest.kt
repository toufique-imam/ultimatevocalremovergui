package com.uvr.android.ml.vr

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
 * Compares [VrPipeline] against desktop UVR (separate.SeperateVR) outputs.
 *
 * Fixtures are generated outside the repo by running the desktop code with a
 * deterministic pointwise mask in place of the network; each fixture dir holds
 * uvr_meta.json, options.json ({"tta": bool}), input.f32 (44.1 kHz stereo
 * interleaved) and the desktop's primary.f32 / secondary.f32. Point UVR_VR_FIXTURES at the parent directory;
 * the test is skipped when it is not set.
 */
class VrPipelineFixtureTest {

    /**
     * Same function as the fixture generator's FakeModel.mask. It depends on the
     * frame's position t inside the window, so a wrong window grid shows up.
     */
    private val fakeModel = MaskModel { input, shape ->
        val rows = shape[2].toInt()
        val frames = shape[3].toInt()
        FloatArray(input.size) { i ->
            val c = i / (rows * frames)
            val r = (i / frames) % rows
            val t = i % frames
            1f / (1f + exp(-(10f * input[i] - 0.5f - 0.3f * c + 0.001f * r + 0.5f * sin(0.05f * t))))
        }
    }

    @Test
    fun matchesDesktopWithFakeMask() {
        val root = System.getenv("UVR_VR_FIXTURES")
        assumeTrue("UVR_VR_FIXTURES not set", root != null)
        val dirs = File(root!!).listFiles()!!.filter { File(it, "uvr_meta.json").exists() }.sorted()
        assertTrue("no fixtures in $root", dirs.isNotEmpty())
        for (dir in dirs) {
            for (windowsPerSegment in intArrayOf(4, 1)) check(dir, windowsPerSegment)
        }
    }

    private fun check(dir: File, windowsPerSegment: Int) {
        val meta = VrModelMeta.parse(File(dir, "uvr_meta.json").readText())
        val tta = File(dir, "options.json").takeIf { it.exists() }
            ?.readText()?.contains("\"tta\": true") ?: false
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
        VrPipeline(meta, 44100, windowsPerSegment = windowsPerSegment, tta = tta)
            .run(source, fakeModel, prim, sec)
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
            println("%-24s seg=%d %-9s SNR=%.1f dB  max|err|=%.2e  (%d ms)"
                .format(dir.name, windowsPerSegment, name, snr, maxAbs, ms))
            assertTrue("${dir.name} $name SNR $snr dB too low", snr > 60)
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
