package com.uvr.android.audio

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.log10
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Desktop UVR's "Trim Silence" (lib_v5/spec_utils.py silence_keep_ranges / trim_silence) for
 * raw interleaved f32le files: drops stretches that stay [THRESHOLD_DB] below the loudest
 * 20 ms (or under −80 dBFS) for at least [MIN_SILENCE] s, keeping [KEEP_SILENCE] s on each
 * side of a cut and fading the joins over 5 ms. Streams through the file, so memory use
 * doesn't grow with the song's length.
 */
object SilenceTrimmer {
    private const val THRESHOLD_DB = -40.0
    private const val MIN_SILENCE = 0.5
    private const val KEEP_SILENCE = 0.15
    private const val FRAME_SEC = 0.02

    /**
     * Rewrite [pcm] without its silent stretches. Returns the kept (start, end) sample-frame
     * ranges, or null when nothing would be cut (the file is left as it is).
     */
    fun trimInPlace(pcm: File, sampleRate: Int, channels: Int = 2): List<LongRange>? {
        val ranges = keepRanges(pcm, sampleRate, channels) ?: return null
        val trimmed = File(pcm.parentFile, "uvr_trim_${System.nanoTime()}.f32")
        try {
            writeRanges(pcm, trimmed, ranges, sampleRate, channels)
            if (!pcm.delete() || !trimmed.renameTo(pcm)) throw java.io.IOException("Could not replace ${pcm.name}")
        } finally {
            trimmed.delete()
        }
        return ranges
    }

    /** The sample-frame ranges [start, end) Trim Silence keeps, or null when nothing would be cut. */
    fun keepRanges(pcm: File, sampleRate: Int, channels: Int = 2): List<LongRange>? {
        val totalFrames = pcm.length() / (4L * channels)
        val frameLen = maxOf(1, (sampleRate * FRAME_SEC).toInt())
        val power = framePower(pcm, totalFrames, frameLen, channels)
        if (power.isEmpty()) return null

        val levelDb = DoubleArray(power.size) { 10 * log10(power[it] + 1e-20) }
        val floor = maxOf(levelDb.max() + THRESHOLD_DB, -80.0)
        val isSound = BooleanArray(power.size) { levelDb[it] > floor }
        if (isSound.all { it } || isSound.none { it }) return null

        val minFrames = (MIN_SILENCE / FRAME_SEC).roundToInt()
        val keepFrames = (KEEP_SILENCE / FRAME_SEC).roundToInt()
        val n = isSound.size
        val keep = BooleanArray(n) { true }

        // Silent runs [start, end) in analysis frames
        var i = 0
        while (i < n) {
            if (isSound[i]) { i++; continue }
            val start = i
            while (i < n && !isSound[i]) i++
            val end = i
            val dropStart = if (start > 0) start + keepFrames else 0
            val dropEnd = if (end < n) end - keepFrames else n
            if (end - start >= minFrames && dropEnd > dropStart) keep.fill(false, dropStart, dropEnd)
        }
        if (keep.all { it }) return null

        val ranges = mutableListOf<LongRange>()
        i = 0
        while (i < n) {
            if (!keep[i]) { i++; continue }
            val start = i
            while (i < n && keep[i]) i++
            ranges += (start.toLong() * frameLen) until minOf(i.toLong() * frameLen, totalFrames)
        }
        return ranges
    }

    /** Mean square of every [frameLen]-frame block (all channels), then of the shorter last block. */
    private fun framePower(pcm: File, totalFrames: Long, frameLen: Int, channels: Int): DoubleArray {
        val fullBlocks = (totalFrames / frameLen).toInt()
        val hasTail = totalFrames > fullBlocks.toLong() * frameLen
        val power = DoubleArray(fullBlocks + if (hasTail) 1 else 0)
        val block = ByteArray(frameLen * channels * 4)
        val floats = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        BufferedInputStream(FileInputStream(pcm), 1 shl 16).use { input ->
            for (b in power.indices) {
                val frames = if (b < fullBlocks) frameLen else (totalFrames - fullBlocks.toLong() * frameLen).toInt()
                val values = frames * channels
                readFully(input, block, values * 4)
                var sum = 0.0
                for (k in 0 until values) {
                    val v = floats.get(k).toDouble()
                    sum += v * v
                }
                power[b] = sum / values
            }
        }
        return power
    }

    /** Copy [ranges] of [input] to [output], fading each cut in and out over 5 ms. */
    private fun writeRanges(input: File, output: File, ranges: List<LongRange>, sampleRate: Int, channels: Int) {
        val totalFrames = input.length() / (4L * channels)
        val fadeLen = (sampleRate * 0.005).toInt()
        // numpy.linspace(0, 1, fadeLen)
        val fade = FloatArray(fadeLen) { if (fadeLen > 1) (it.toDouble() / (fadeLen - 1)).toFloat() else 0f }
        val chunkFrames = 1 shl 14
        val bytes = ByteArray(chunkFrames * channels * 4)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        RandomAccessFile(input, "r").use { raf ->
            BufferedOutputStream(FileOutputStream(output), 1 shl 16).use { out ->
                for (range in ranges) {
                    val start = range.first
                    val length = range.last + 1 - start
                    val n = min(fadeLen.toLong(), length / 2).toInt()
                    val fadeIn = start > 0
                    val fadeOut = range.last + 1 < totalFrames
                    raf.seek(start * channels * 4)
                    var done = 0L
                    while (done < length) {
                        val frames = min(chunkFrames.toLong(), length - done).toInt()
                        raf.readFully(bytes, 0, frames * channels * 4)
                        for (f in 0 until frames) {
                            val pos = done + f
                            var gain = 1f
                            if (fadeIn && pos < n) gain *= fade[pos.toInt()]
                            if (fadeOut && pos >= length - n) gain *= fade[(length - 1 - pos).toInt()]
                            if (gain != 1f) {
                                for (c in 0 until channels) {
                                    val at = (f * channels + c) * 4
                                    buffer.putFloat(at, buffer.getFloat(at) * gain)
                                }
                            }
                        }
                        out.write(bytes, 0, frames * channels * 4)
                        done += frames
                    }
                }
            }
        }
    }

    private fun readFully(input: java.io.InputStream, buf: ByteArray, count: Int) {
        var read = 0
        while (read < count) {
            val r = input.read(buf, read, count - read)
            if (r < 0) throw java.io.EOFException()
            read += r
        }
    }
}
