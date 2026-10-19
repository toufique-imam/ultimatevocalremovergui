package com.uvr.android.ml

import android.util.Log
import java.io.File

/**
 * MNN on the CPU in FP32 for the `.mnn` exported next to a model by mnn_export.py — the same
 * math as ONNX Runtime, faster on some phones: Kim_Vocal_2 takes 29 s per chunk instead of 37 on a
 * Galaxy M12 (8×Cortex-A55), but ~1.35× longer on a Snapdragon 680 (A73/A53). Callers check it
 * against ONNX Runtime per phone ([VerifiedEngine]).
 *
 * MNN's two inexact shortcuts are off: Winograd 3x3 convolutions (session hint, see uvr_mnn.cpp)
 * and Strassen matrix products (patched out of the build, see patch_mnn.cmake). What remains is
 * FP32 rounding: ~118 dB SNR against ONNX Runtime on Kim_Vocal_2, vs ~80 dB with MNN's defaults.
 * The session keeps the model's input shape (see uvr_mnn.cpp).
 *
 * Threads = [CpuInfo.inferenceThreads], as for ONNX Runtime. Not thread-safe: one caller at a time.
 */
class MnnCpuEngine(modelFile: File, threads: Int = CpuInfo.inferenceThreads()) : InferenceEngine {

    private var handle: Long

    /** The model's input shape (NCHW), as MNN read it from the file. */
    val inputShape: IntArray

    init {
        check(isAvailable) { "MNN isn't available on this device" }
        handle = nativeCreate(modelFile.absolutePath, threads)
        inputShape = nativeInputShape(handle)
        Log.i(TAG, "MNN CPU engine ($threads threads): ${modelFile.name} input=${inputShape.toList()}")
    }

    override fun runBatch(data: FloatArray, shape: LongArray): FloatArray {
        check(handle != 0L) { "engine closed" }
        return nativeRun(handle, data, IntArray(shape.size) { shape[it].toInt() })
    }

    override val inputNames: Set<String> get() = setOf("input")
    override val outputNames: Set<String> get() = setOf("output")
    override fun expectedInputShape(inputName: String): LongArray = LongArray(inputShape.size) { inputShape[it].toLong() }
    override fun expectedInputRank(inputName: String): Int = inputShape.size

    override fun close() {
        if (handle != 0L) nativeRelease(handle)
        handle = 0L
    }

    companion object {
        private const val TAG = "MnnCpuEngine"

        /** False where the native library is missing or built without MNN (non-arm64 ABIs). */
        val isAvailable: Boolean by lazy {
            runCatching { System.loadLibrary("uvr_mnn"); !nativeIsStub() }
                .onFailure { Log.w(TAG, "MNN unavailable: ${it.message}") }
                .getOrDefault(false)
        }

        @JvmStatic private external fun nativeIsStub(): Boolean
        @JvmStatic private external fun nativeCreate(path: String, threads: Int): Long
        @JvmStatic private external fun nativeInputShape(handle: Long): IntArray
        @JvmStatic private external fun nativeRun(handle: Long, data: FloatArray, shape: IntArray): FloatArray
        @JvmStatic private external fun nativeRelease(handle: Long)
    }
}
