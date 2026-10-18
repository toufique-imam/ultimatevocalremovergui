package com.uvr.android.audio

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Compares [SilenceTrimmer] against desktop UVR's spec_utils.trim_silence. Each fixture dir
 * holds input.f32 (interleaved stereo), case.json ({"sr": …, "ranges": [[start, end], …] or
 * null}) and, when the desktop cut something, its output as expected.f32.
 * Point UVR_TRIM_FIXTURES at the parent directory; skipped when unset.
 */
class SilenceTrimmerFixtureTest {

    @Test
    fun matchesDesktop() {
        val root = System.getenv("UVR_TRIM_FIXTURES")
        assumeTrue("UVR_TRIM_FIXTURES not set", root != null)
        val dirs = File(root!!).listFiles()!!.filter { File(it, "case.json").exists() }.sorted()
        assertTrue("no fixtures in $root", dirs.isNotEmpty())
        val failures = dirs.mapNotNull { check(it) }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    private fun check(dir: File): String? {
        val json = File(dir, "case.json").readText()
        val sampleRate = Regex("\"sr\": (\\d+)").find(json)!!.groupValues[1].toInt()
        val expectedRanges = if (json.contains("\"ranges\": null")) null
            else Regex("\\[(\\d+), (\\d+)\\]").findAll(json).map { it.groupValues[1].toLong() until it.groupValues[2].toLong() }.toList()

        val work = File.createTempFile("trim", ".f32").also { File(dir, "input.f32").copyTo(it, overwrite = true) }
        try {
            val ranges = SilenceTrimmer.trimInPlace(work, sampleRate)
            if (ranges != expectedRanges) return "${dir.name}: ranges $ranges, desktop $expectedRanges"
            val expected = if (expectedRanges == null) File(dir, "input.f32") else File(dir, "expected.f32")
            val got = read(work)
            val want = read(expected)
            if (got.size != want.size) return "${dir.name}: ${got.size} samples, desktop ${want.size}"
            val maxErr = got.indices.maxOfOrNull { abs(got[it] - want[it]) } ?: 0f
            return if (maxErr > 1e-6f) "${dir.name}: max error $maxErr" else null
        } finally {
            work.delete()
        }
    }

    private fun read(file: File): FloatArray {
        val bb = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(bb.remaining()).also { bb.get(it) }
    }
}
