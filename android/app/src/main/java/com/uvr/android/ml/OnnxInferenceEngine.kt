package com.uvr.android.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * ONNX Runtime on the CPU in FP32 — the same execution desktop UVR uses for
 * MDX-Net, so outputs match it to float rounding.
 *
 * NNAPI is deliberately not used: drivers may run it in reduced precision.
 * Threads = [CpuInfo.inferenceThreads]: the performance cores; on big.LITTLE chips
 * adding the little cores makes ORT slower (measured on the S20 FE).
 */
class OnnxInferenceEngine(
    modelFile: File,
    /**
     * Free every activation right after use (no arena, no memory-pattern
     * planning). Needed for htdemucs, whose peak otherwise exceeds 4 GB and gets
     * the app killed, and for MDX-Net on phones under 4 GB (Kim_Vocal_2 peaks at
     * 2.4 GB with the arena, 1.6 GB without on the S20 FE); speed is unchanged.
     */
    lowMemory: Boolean = false
) : InferenceEngine {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private var inputBuffer: FloatBuffer? = null

    init {
        val threads = OrtTuning.threads ?: CpuInfo.inferenceThreads()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(threads)
            setInterOpNumThreads(1)
            if (lowMemory) {
                setCPUArenaAllocator(false)
                setMemoryPatternOptimization(false)
            }
            if (!OrtTuning.allowSpinning) addConfigEntry("session.intra_op.allow_spinning", "0")
            if (OrtTuning.xnnpack) {
                // XNNPACK runs its own thread pool; ops it can't take stay on the CPU provider
                setIntraOpNumThreads(1)
                addXnnpack(mapOf("intra_op_num_threads" to "$threads"))
            }
        }
        session = env.createSession(modelFile.absolutePath, opts)
        inputName = session.inputNames.first()
        Log.i(TAG, "ONNX CPU session ($threads threads${if (OrtTuning.xnnpack) ", XNNPACK" else ""}" +
                "${if (!OrtTuning.allowSpinning) ", no spinning" else ""}): ${modelFile.name} inputs=${session.inputNames}")
    }

    override fun runBatch(data: FloatArray, shape: LongArray): FloatArray {
        val buffer = inputBuffer?.takeIf { it.capacity() == data.size }
            ?: ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
                .also { inputBuffer = it }
        buffer.rewind()
        buffer.put(data)
        buffer.rewind()
        OnnxTensor.createTensor(env, buffer, shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val out = (result[0] as OnnxTensor).floatBuffer
                return FloatArray(out.remaining()).also { out.get(it) }
            }
        }
    }

    /** Multi-input run: name → (data, shape); returns every output flattened, in output order. */
    fun run(inputs: Map<String, Pair<FloatArray, LongArray>>): List<FloatArray> {
        val tensors = inputs.mapValues { (_, v) -> OnnxTensor.createTensor(env, FloatBuffer.wrap(v.first), v.second) }
        try {
            session.run(tensors).use { result ->
                return (0 until result.size()).map { i ->
                    val out = (result[i] as OnnxTensor).floatBuffer
                    FloatArray(out.remaining()).also { out.get(it) }
                }
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    override val inputNames: Set<String> get() = session.inputNames
    override val outputNames: Set<String> get() = session.outputNames
    override fun expectedInputShape(inputName: String): LongArray? =
        (session.inputInfo[inputName]?.info as? ai.onnxruntime.TensorInfo)?.shape
    override fun expectedInputRank(inputName: String): Int = expectedInputShape(inputName)?.size ?: -1

    override fun close() = session.close()

    private companion object {
        const val TAG = "OnnxInferenceEngine"
    }
}

/**
 * ONNX Runtime CPU settings for new sessions. The defaults are what the app uses;
 * on-device benchmarks (androidTest) change them to compare devices' best settings.
 */
object OrtTuning {
    /** Intra-op threads; null = [CpuInfo.inferenceThreads]. */
    @Volatile var threads: Int? = null
    /** Let idle worker threads spin between ops (ORT's default). */
    @Volatile var allowSpinning: Boolean = true
    /** Run supported ops on the XNNPACK execution provider. */
    @Volatile var xnnpack: Boolean = false
}

/** CPU topology helpers. */
object CpuInfo {
    /**
     * ONNX Runtime threads: the performance cores, but two fewer on chips whose cores are all
     * alike, where the last two only contend with the app's own threads. On an 8×A55 Exynos 850,
     * 6 threads ran MDX-Net 8 % and Demucs 17–20 % faster than 8 (4 was slower than both).
     */
    fun inferenceThreads(): Int {
        val performance = performanceCores()
        val all = Runtime.getRuntime().availableProcessors()
        return if (performance == all && all > 4) all - 2 else performance
    }

    /**
     * Cores whose max frequency is within 70% of the fastest core's: 4 on a
     * 1+3+4 Snapdragon 865, all 8 on an 8×A55 Exynos 850.
     */
    fun performanceCores(): Int {
        val freqs = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { cpu ->
            runCatching {
                File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq").readText().trim().toLong()
            }.getOrNull()
        }
        if (freqs.isEmpty()) return maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
        val top = freqs.max()
        return freqs.count { it >= top * 0.7 }.coerceAtLeast(1)
    }
}
