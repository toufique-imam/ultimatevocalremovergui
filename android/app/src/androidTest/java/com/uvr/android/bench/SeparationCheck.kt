package com.uvr.android.bench

import android.os.Bundle
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.uvr.android.audio.DecodedAudio
import com.uvr.android.ml.ModelManager
import com.uvr.android.ml.OrtTuning
import com.uvr.android.ml.SeparationParams
import com.uvr.android.ml.Separator
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.log10

/**
 * One separation on device, timed, and compared with a desktop UVR output.
 * Push <externalFilesDir>/bench/clip.f32 (44.1 kHz stereo f32le) and the model
 * (e.g. to <externalFilesDir>/models) — after one run with -e setup 1, which creates those folders — then:
 *   adb shell am instrument -w -e class com.uvr.android.bench.SeparationCheck \
 *     -e model Kim_Vocal_2.onnx [-e ref mdx_kim2_primary.f32] [-e denoise 1] [-e overlap 0.25] \
 *     [-e configs t8,t4,x8,t8] [-e out clip01_] \
 *     com.uvr.android.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class SeparationCheck {
    private val inst = InstrumentationRegistry.getInstrumentation()
    private val ctx = inst.targetContext
    private val args = InstrumentationRegistry.getArguments()

    private fun say(msg: String) {
        Log.i("UVRBench", msg)
        inst.sendStatus(0, Bundle().apply { putString("stream", msg + "\n") })
    }

    @Test
    fun separate() = runBlocking {
        val bench = File(ctx.getExternalFilesDir(null), "bench")
        // The app makes its own folders: on some Android 15 builds (OPPO ColorOS) it can't read
        // files in a folder that adb created. Run with -e setup 1 before the first push.
        File(bench, "out").mkdirs()
        ctx.getExternalFilesDir("models")
        if (args.getString("setup") == "1") {
            say("SETUP ${bench.path}")
            return@runBlocking
        }
        val clip = File(bench, args.getString("clip") ?: "clip.f32")
        check(clip.length() > 0) { "$clip is missing or unreadable" }
        val audio = DecodedAudio(clip, 44100, 2, clip.length() / 4)
        ModelManager.scanModels()
        val name = args.getString("model")!!
        val model = ModelManager.availableModels.first { it.fileName == name }
        check(model.isSupported) { "${model.fileName}: ${model.unsupportedReason}" }
        val params = SeparationParams(
            mdxOverlap = args.getString("overlap")?.toFloat(),
            mdxDenoise = args.getString("denoise") == "1",
            tta = args.getString("tta") == "1",
            useGpu = args.getString("gpu") != "0",
            // Benchmarks measure the GPU path unless told otherwise; the app defaults to "Exact"
            mdxFastGpu = args.getString("gpu") != "0",
            demucsShifts = args.getString("shifts")?.toInt() ?: 2,
            demucsOverlap = args.getString("doverlap")?.toFloat() ?: 0.25f,
        ).let { p ->
            // Stem order and names come from the model, as in the app (SP-UVR VR models output Instrumental first)
            p.copy(primaryStemName = model.primaryStem ?: p.primaryStemName,
                   secondaryStemName = model.secondaryStem ?: p.secondaryStemName)
        }
        val seconds = audio.totalSamples / 2.0 / 44100
        // -e configs t8,t4,x8,n4: ONNX Runtime with N threads (t), XNNPACK with N threads (x),
        // N threads without spinning (n); each separates the clip once. Default: the app's settings.
        val configs = args.getString("configs")?.split(",")?.map { it.trim() } ?: listOf("")
        var result: com.uvr.android.ml.SeparationResult? = null
        for (config in configs) {
            result?.stems?.forEach { it.file.delete() }
            OrtTuning.threads = config.drop(1).toIntOrNull()
            OrtTuning.xnnpack = config.startsWith("x")
            OrtTuning.allowSpinning = !config.startsWith("n")
            val t0 = System.currentTimeMillis()
            result = Separator.create(model, params).separate(audio, ctx.cacheDir)
            val ms = System.currentTimeMillis() - t0
            say("SEP ${model.fileName} (${model.architectureType}) [${config.ifEmpty { "app" }}]: $ms ms for %.1f s audio (%.2fx realtime)".format(
                seconds, seconds * 1000 / ms))
            val status = File("/proc/self/status").readLines()
            fun field(name: String) = status.firstOrNull { it.startsWith(name) }?.substringAfter(':')?.trim()
            say("  process RSS after this run: ${field("VmRSS")}, peak: ${field("VmHWM")}")
        }
        OrtTuning.threads = null; OrtTuning.xnnpack = false; OrtTuning.allowSpinning = true
        val output = result!!
        // -e ref FILE compares the first stem; -e refs PREFIX compares every stem with PREFIX<Stem>.f32
        val refs = args.getString("refs")
        for ((i, stem) in output.stems.withIndex()) {
            val ref = if (refs != null) File(bench, "$refs${stem.name}.f32")
                else if (i == 0) args.getString("ref")?.let { File(bench, it) } else null
            if (ref?.exists() == true) say("  ${stem.name} vs desktop SNR=%.1f dB".format(snrDb(ref, stem.file)))
        }
        // -e out PREFIX keeps every stem as bench/out/PREFIX<Stem>.f32 (raw interleaved stereo at the rate printed)
        val out = args.getString("out")
        for (stem in output.stems) {
            if (out != null) {
                val dest = File(bench, "out/$out${stem.name}.f32").also { it.parentFile!!.mkdirs() }
                stem.file.copyTo(dest, overwrite = true)
                say("OUT ${dest.name} ${output.sampleRate}")
            }
            stem.file.delete()
        }
        Unit
    }

    private fun snrDb(ref: File, out: File): Double {
        val x = read(ref); val y = read(out)
        check(x.size == y.size) { "length ${y.size} ≠ desktop ${x.size}" }
        var s = 0.0; var e = 0.0
        for (i in x.indices) { s += x[i] * x[i].toDouble(); val d = x[i] - y[i]; e += d * d.toDouble() }
        return 10 * log10(s / maxOf(e, 1e-30))
    }

    private fun read(f: File): FloatArray = RandomAccessFile(f, "r").use { raf ->
        val bytes = ByteArray(raf.length().toInt()); raf.readFully(bytes)
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        FloatArray(fb.remaining()).also { fb.get(it) }
    }
}
