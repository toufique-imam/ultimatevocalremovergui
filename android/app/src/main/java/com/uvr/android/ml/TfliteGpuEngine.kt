package com.uvr.android.ml

import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors

/**
 * LiteRT (TensorFlow Lite) GPU engine for the VR and MDX-Net nets exported by convert.ipynb.
 *
 * VR runs in full FP32 (precision loss disallowed): FP16 is ~2× faster but moves
 * the mask by up to 0.13, while FP32 stays within ~1e-5 of the .pth — closer than
 * the PTL/XNNPACK CPU path. On the S20 FE (Adreno 650) a 512-frame window takes
 * ~1.1 s vs ~3.1 s for the .ptl on CPU. MDX-Net is only faster than the CPU in
 * FP16 ([fp16]), so its result is checked by level rather than per value.
 *
 * The .tflite graph is NHWC with a static [1, rows, frames, channels] shape; callers
 * use the pipeline's NCHW [1, channels, rows, frames] layout, converted here.
 * All interpreter calls run on one dedicated thread, which the GPU delegate needs.
 */
class TfliteGpuEngine(modelFile: File, private val fp16: Boolean = false) : InferenceEngine {

    private val thread = Executors.newSingleThreadExecutor { r -> Thread(r, "tflite-gpu") }
    private lateinit var delegate: GpuDelegate
    private lateinit var interpreter: Interpreter
    private lateinit var inBuf: ByteBuffer
    private lateinit var outBuf: ByteBuffer

    /** Model input as NHWC [1, rows, frames, channels]. */
    val inputShape: IntArray

    init {
        inputShape = onThread {
            delegate = GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(fp16))
            try {
                interpreter = Interpreter(mapFile(modelFile), Interpreter.Options().addDelegate(delegate))
            } catch (e: Throwable) {
                delegate.close()
                throw e
            }
            val shape = interpreter.getInputTensor(0).shape()
            val bytes = shape.fold(4) { acc, d -> acc * d }
            inBuf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
            outBuf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
            shape
        }
        Log.i(TAG, "GPU (${if (fp16) "FP16" else "FP32"}) engine ready: ${modelFile.name} input=${inputShape.toList()}")
    }

    /** True if this engine can run NCHW windows of [shape] = [1, channels, rows, frames]. */
    fun accepts(shape: LongArray): Boolean =
        shape.size == 4 && shape[0] == 1L &&
            shape[1].toInt() == inputShape[3] && shape[2].toInt() == inputShape[1] && shape[3].toInt() == inputShape[2]

    override fun runBatch(data: FloatArray, shape: LongArray): FloatArray {
        require(accepts(shape)) { "GPU model expects NHWC ${inputShape.toList()}, got NCHW ${shape.toList()}" }
        val channels = shape[1].toInt()
        val plane = shape[2].toInt() * shape[3].toInt()
        return onThread {
            inBuf.rewind()
            val input: FloatBuffer = inBuf.asFloatBuffer()
            val nhwc = FloatArray(data.size)
            for (c in 0 until channels) {
                val base = c * plane
                for (i in 0 until plane) nhwc[i * channels + c] = data[base + i]
            }
            input.put(nhwc)
            outBuf.rewind()
            interpreter.run(inBuf, outBuf)
            outBuf.rewind()
            outBuf.asFloatBuffer().get(nhwc)
            val out = FloatArray(data.size)
            for (c in 0 until channels) {
                val base = c * plane
                for (i in 0 until plane) out[base + i] = nhwc[i * channels + c]
            }
            out
        }
    }

    override val inputNames: Set<String> get() = setOf("input")
    override val outputNames: Set<String> get() = setOf("output")
    override fun expectedInputShape(inputName: String): LongArray =
        longArrayOf(1, inputShape[3].toLong(), inputShape[1].toLong(), inputShape[2].toLong())
    override fun expectedInputRank(inputName: String): Int = 4

    override fun close() {
        runCatching {
            onThread {
                if (::interpreter.isInitialized) interpreter.close()
                if (::delegate.isInitialized) delegate.close()
            }
        }
        thread.shutdown()
    }

    private fun <T> onThread(block: () -> T): T = try {
        thread.submit(Callable { block() }).get()
    } catch (e: ExecutionException) {
        throw e.cause ?: e
    }

    private fun mapFile(file: File): ByteBuffer = FileInputStream(file).use { stream ->
        stream.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    companion object {
        private const val TAG = "TfliteGpuEngine"
    }
}
