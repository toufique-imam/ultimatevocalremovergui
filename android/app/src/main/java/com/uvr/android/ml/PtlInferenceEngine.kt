package com.uvr.android.ml

import android.util.Log
import org.pytorch.IValue
//import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.nio.FloatBuffer

/**
 * PyTorch inference engine supporting two TorchScript serialisation formats:
 *
 *  1. **Lite / Mobile** (`bytecode.pkl` inside the zip) — produced by:
 *       model._save_for_lite_interpreter("model.ptl")
 *     Loaded via [LiteModuleLoader.load].
 *
 *  2. **Full TorchScript** (`data.pkl` inside the zip) — produced by:
 *       torch.jit.save(model, "model.ptl")   or   model.save("model.pt")
 *     Loaded via [Module.load].
 *
 * The engine detects which format is present at load time and picks the right
 * loader automatically. Both are unified behind [InferenceEngine] so
 * separators never need to care about the distinction.
 *
 * To convert a full TorchScript model to the Lite format (smaller + faster
 * on-device), run this Python snippet once:
 *
 *   from torch.utils.mobile_optimizer import optimize_for_mobile
 *   scripted = torch.jit.load("model.ptl")
 *   lite = optimize_for_mobile(scripted)
 *   lite._save_for_lite_interpreter("model_lite.ptl")
 */
/** Thrown when the PyTorch Lite native library is missing from the device. */
class PtlNotAvailableException : RuntimeException(
    "PyTorch Lite native library (libpytorch_jni_lite.so) is not available on this device. " +
    "Convert your model to ONNX format: torch.onnx.export(model, dummy_input, 'model.onnx')"
)

class PtlInferenceEngine(private val modelFilePath: String) : InferenceEngine {

    private sealed class Backend {
        class Lite(val module: org.pytorch.Module) : Backend()   // LiteModuleLoader result
        class Full(val module: Module) : Backend()               // Module.load result
    }

    private val backend: Backend

    // One direct buffer reused for every call: allocating a fresh native buffer per
    // model window piles up off-heap memory until GC runs, which long songs hit.
    private var inputBuffer: FloatBuffer? = null

    init {
        if (!isAvailable) throw PtlNotAvailableException()
        backend = loadModule(modelFilePath)
    }

    init {
        val kind = when (backend) {
            is Backend.Lite -> "Lite (bytecode.pkl)"
            is Backend.Full -> "Full TorchScript (data.pkl)"
        }
        Log.i(TAG, "Loaded [$kind] from: $modelFilePath")
    }

    override fun runBatch(data: FloatArray, shape: LongArray): FloatArray {
        runCatching { forward(data, shape) }
            .onSuccess { return it }
            .onFailure { e ->
                Log.w(TAG, "forward failed with shape ${shape.toList()}: ${e.message}. " +
                        "Trying unsqueezed shapes…")
            }

        for (extraDims in 1..2) {
            val adapted = unsqueeze(shape, shape.size + extraDims)
            Log.d(TAG, "Retry with shape ${adapted.toList()}")
            runCatching { forward(data, adapted) }
                .onSuccess {
                    Log.i(TAG, "Succeeded with adapted shape ${adapted.toList()}")
                    return it
                }
        }

        // Final attempt — surface the real error
        return forward(data, shape)
    }

    override fun expectedInputShape(inputName: String): LongArray? = null
    override fun expectedInputRank(inputName: String): Int = -1
    override val inputNames: Set<String> get() = setOf("input")
    override val outputNames: Set<String> get() = setOf("output")

    override fun close() {
        when (backend) {
            is Backend.Lite -> backend.module.destroy()
            is Backend.Full -> backend.module.destroy()
        }
        Log.d(TAG, "PTL module destroyed")
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private fun forward(data: FloatArray, shape: LongArray): FloatArray {
        val buffer = inputBuffer?.takeIf { it.capacity() == data.size }
            ?: Tensor.allocateFloatBuffer(data.size).also { inputBuffer = it }
        buffer.rewind()
        buffer.put(data)
        buffer.rewind()
        val inputTensor = Tensor.fromBlob(buffer, shape)
        val outputIValue = when (backend) {
            is Backend.Lite -> backend.module.forward(IValue.from(inputTensor))
            is Backend.Full -> backend.module.forward(IValue.from(inputTensor))
        }
        return flattenIValue(outputIValue)
    }

    private fun flattenIValue(value: IValue): FloatArray = when {
        value.isTensor -> value.toTensor().dataAsFloatArray

        value.isTuple -> {
            val elements = value.toTuple()
            if (elements.size == 1) flattenIValue(elements[0])
            else elements.filter { it.isTensor }
                .flatMap { it.toTensor().dataAsFloatArray.toList() }
                .toFloatArray()
        }

        value.isList -> value.toList()
            .filter { it.isTensor }
            .flatMap { it.toTensor().dataAsFloatArray.toList() }
            .toFloatArray()

        else -> throw IllegalStateException(
            "Unsupported PTL output type. Expected Tensor or Tuple<Tensor>."
        )
    }

    private fun unsqueeze(shape: LongArray, targetRank: Int): LongArray {
        if (shape.size >= targetRank) return shape
        val result = LongArray(targetRank)
        val insertCount = targetRank - shape.size
        for (i in 0 until shape.size - 1) result[i] = shape[i]
        for (i in shape.size - 1 until shape.size - 1 + insertCount) result[i] = 1L
        result[targetRank - 1] = shape.last()
        return result
    }

    companion object {
        private const val TAG = "PtlInferenceEngine"

        /**
         * True if the PyTorch Lite native library loaded successfully.
         * Evaluated once lazily; if false, attempting to construct a
         * [PtlInferenceEngine] will throw [PtlNotAvailableException] with a
         * clear message instead of crashing the process with UnsatisfiedLinkError.
         */
        val isAvailable: Boolean by lazy {
            try {
                // pytorch_android_lite 1.x uses libpytorch_jni_lite.so
                System.loadLibrary("pytorch_jni_lite")
                Log.i(TAG, "PyTorch Lite native library loaded (pytorch_jni_lite)")
                true
            } catch (lite: UnsatisfiedLinkError) {
                try {
                    // Fallback: 2.x merged into libpytorch_jni.so
                    System.loadLibrary("pytorch_jni")
                    Log.i(TAG, "PyTorch Lite native library loaded (pytorch_jni)")
                    true
                } catch (full: UnsatisfiedLinkError) {
                    Log.w(TAG, "PyTorch Lite native library not found on this device. " +
                            "PTL models will not be available. " +
                            "Convert to ONNX with: " +
                            "torch.onnx.export(model, dummy, 'model.onnx')")
                    false
                }
            }
        }

        /**
         * Detects TorchScript format and picks the right loader:
         * - Lite format (bytecode.pkl) → [LiteModuleLoader]
         * - Full TorchScript (data.pkl) → [Module.load]
         */
        private fun loadModule(path: String): Backend {
            return Backend.Full(Module.load(path))

//            return try {
//                Backend.Lite(LiteModuleLoader.load(path))
//            } catch (e: Exception) {
//                val msg = e.message ?: ""
//                if (msg.contains("bytecode.pkl") || msg.contains("file not found")) {
//                    Log.w(TAG, "Not a Lite model (no bytecode.pkl), " +
//                            "falling back to full TorchScript loader: $path")
//                } else {
//                    throw e
//                }
//            }
        }
    }
}
