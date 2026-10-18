package com.uvr.android.bench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.uvr.android.ml.PtlInferenceEngine
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs

/**
 * On-device inference benchmark. Push to <externalFilesDir>/bench:
 *   in.bin, ref.bin (f32le [1,2,769,512]), model.ptl, model.onnx
 * Run: adb shell am instrument -w -e class com.uvr.android.bench.EngineBenchmark \
 *        com.uvr.android.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class EngineBenchmark {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val dir = File(ctx.getExternalFilesDir(null), "bench")
    private val args = InstrumentationRegistry.getArguments()
    // NCHW; default is a VR window
    private val shape = (args.getString("shape") ?: "1,2,769,512").split(",").map { it.trim().toLong() }.toLongArray()
    private val reps = args.getString("reps")?.toInt() ?: 3

    private fun readF32(f: File): FloatArray {
        val bb = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(bb.remaining()).also { bb.get(it) }
    }

    private fun report(name: String, ref: FloatArray, run: () -> FloatArray) {
        val first = run()
        var err = 0f
        for (i in ref.indices) err = maxOf(err, abs(first[i] - ref[i]))
        val t = LongArray(reps) {
            val t0 = System.nanoTime(); run(); (System.nanoTime() - t0) / 1_000_000
        }
        val msg = "BENCH $name: ms=${t.toList()} min=${t.min()} maxErr=$err"
        Log.i("UVRBench", msg)
        InstrumentationRegistry.getInstrumentation().sendStatus(0, android.os.Bundle().apply { putString("stream", msg + "\n") })
    }

    @Test
    fun benchmark() {
        val input = readF32(File(dir, args.getString("in") ?: "in.bin"))
        val ref = readF32(File(dir, args.getString("ref") ?: "ref.bin"))
        val which = (args.getString("which") ?: "ptl,ort4,ort8,xnn4,nnapi").split(",")

        if ("ptl" in which) {
            PtlInferenceEngine(File(dir, args.getString("ptl") ?: "model.ptl").path).use { e ->
                report("ptl", ref) { e.runBatch(input, shape) }
            }
        }
        fun tfl(name: String, cfg: org.tensorflow.lite.Interpreter.Options.() -> Unit) {
            if (name !in which) return
            try {
                val opts = org.tensorflow.lite.Interpreter.Options().apply(cfg)
                org.tensorflow.lite.Interpreter(File(dir, args.getString("tflite") ?: "model.tflite"), opts).use { it ->
                    // NHWC [1, H, W, C]
                    val C = shape[1].toInt(); val H = shape[2].toInt(); val W = shape[3].toInt()
                    val inBuf = ByteBuffer.allocateDirect(input.size * 4).order(ByteOrder.nativeOrder())
                    val inF = inBuf.asFloatBuffer()
                    for (h in 0 until H) for (w in 0 until W) for (c in 0 until C) inF.put(input[(c * H + h) * W + w])
                    val outBuf = ByteBuffer.allocateDirect(input.size * 4).order(ByteOrder.nativeOrder())
                    report(name, ref) {
                        inBuf.rewind(); outBuf.rewind()
                        it.run(inBuf, outBuf)
                        outBuf.rewind()
                        val of = outBuf.asFloatBuffer()
                        val out = FloatArray(input.size)
                        for (h in 0 until H) for (w in 0 until W) for (c in 0 until C) out[(c * H + h) * W + w] = of.get()
                        out
                    }
                }
            } catch (e: Throwable) {
                Log.e("UVRBench", "BENCH $name failed", e)
                InstrumentationRegistry.getInstrumentation().sendStatus(0, android.os.Bundle().apply { putString("stream", "BENCH $name failed: $e\n") })
            }
        }
        tfl("tfl4") { setNumThreads(4) }
        tfl("tfl2") { setNumThreads(2) }
        tfl("tfl3") { setNumThreads(3) }
        tfl("tfl8") { setNumThreads(8) }
        tfl("gpu32") {
            addDelegate(org.tensorflow.lite.gpu.GpuDelegate(
                org.tensorflow.lite.gpu.GpuDelegate.Options().setPrecisionLossAllowed(false)))
        }
        tfl("gpu16") {
            addDelegate(org.tensorflow.lite.gpu.GpuDelegate(
                org.tensorflow.lite.gpu.GpuDelegate.Options().setPrecisionLossAllowed(true)))
        }

        val env = OrtEnvironment.getEnvironment()
        val onnx = File(dir, args.getString("onnx") ?: "model.onnx").path
        fun ort(name: String, cfg: OrtSession.SessionOptions.() -> Unit) {
            if (name !in which) return
            try {
                val opts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    cfg()
                }
                env.createSession(onnx, opts).use { s ->
                    val buf = ByteBuffer.allocateDirect(input.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
                    buf.put(input).rewind()
                    report(name, ref) {
                        OnnxTensor.createTensor(env, buf, shape).use { t ->
                            s.run(mapOf("input" to t)).use { r ->
                                val fb = (r[0] as OnnxTensor).floatBuffer
                                FloatArray(fb.remaining()).also { fb.get(it) }
                            }
                        }
                    }
                }
            } catch (e: Throwable) {
                Log.e("UVRBench", "BENCH $name failed", e)
                InstrumentationRegistry.getInstrumentation().sendStatus(0, android.os.Bundle().apply { putString("stream", "BENCH $name failed: $e\n") })
            }
        }
        ort("ort1") { setIntraOpNumThreads(1) }
        ort("ort4") { setIntraOpNumThreads(4) }
        ort("ort8") { setIntraOpNumThreads(8) }
        ort("ort4spin0") { setIntraOpNumThreads(4); addConfigEntry("session.intra_op.allow_spinning", "0") }
        ort("xnn8") { setIntraOpNumThreads(1); addXnnpack(mapOf("intra_op_num_threads" to "8")) }
        ort("xnn4") { setIntraOpNumThreads(1); addXnnpack(mapOf("intra_op_num_threads" to "4")) }
        ort("ort3") { setIntraOpNumThreads(3) }
        ort("ort5") { setIntraOpNumThreads(5) }
        ort("nnapi") {
            addNnapi(java.util.EnumSet.of(ai.onnxruntime.providers.NNAPIFlags.CPU_DISABLED))
        }
    }
}
