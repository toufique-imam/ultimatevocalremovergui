package com.uvr.android.ml

import com.uvr.android.ml.vr.StereoSink
import com.uvr.android.ml.vr.StereoSource
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Random access to an interleaved stereo f32le file. */
internal class RawStereoSource(file: File) : StereoSource, AutoCloseable {
    private val raf = RandomAccessFile(file, "r")
    override val length: Long = raf.length() / 8
    private var buf = ByteArray(0)

    override fun read(start: Long, left: FloatArray, right: FloatArray) {
        left.fill(0f); right.fill(0f)
        val from = maxOf(start, 0L)
        val to = minOf(start + left.size, length)
        if (to <= from) return
        val bytes = ((to - from) * 8).toInt()
        if (buf.size < bytes) buf = ByteArray(bytes)
        raf.seek(from * 8)
        raf.readFully(buf, 0, bytes)
        val bb = ByteBuffer.wrap(buf, 0, bytes).order(ByteOrder.LITTLE_ENDIAN)
        val o = (from - start).toInt()
        for (i in 0 until (to - from).toInt()) {
            left[o + i] = bb.float
            right[o + i] = bb.float
        }
    }

    override fun close() = raf.close()
}

/** Appends interleaved stereo f32le frames to a file. */
internal class RawStereoSink(file: File) : StereoSink, AutoCloseable {
    private val out = BufferedOutputStream(FileOutputStream(file), 1 shl 16)
    private val bb = ByteBuffer.allocate(8 * 4096).order(ByteOrder.LITTLE_ENDIAN)

    override fun write(left: FloatArray, right: FloatArray, count: Int) {
        for (i in 0 until count) {
            if (!bb.hasRemaining()) drain()
            bb.putFloat(left[i]).putFloat(right[i])
        }
        drain()
    }

    private fun drain() {
        out.write(bb.array(), 0, bb.position())
        bb.clear()
    }

    override fun close() = out.close()
}
