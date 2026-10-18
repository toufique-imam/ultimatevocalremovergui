package com.uvr.android.bench

import android.os.Bundle
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.uvr.android.audio.DecodedAudio
import com.uvr.android.ml.ModelManager
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
 * End-to-end VR separation timing/accuracy on device. Push a 44.1 kHz stereo
 * f32le file to <externalFilesDir>/bench/clip.f32, then:
 *   adb shell am instrument -w -e class com.uvr.android.bench.VrSeparationBenchmark \
 *     [-e model 12_SP-UVR-3B-44100_vr.ptl] com.uvr.android.test/androidx.test.runner.AndroidJUnitRunner
 * Runs CPU (no TTA), GPU (no TTA), GPU + TTA and compares stems with the CPU run.
 */
@RunWith(AndroidJUnit4::class)
class VrSeparationBenchmark {
    private val inst = InstrumentationRegistry.getInstrumentation()
    private val ctx = inst.targetContext

    private fun say(msg: String) {
        Log.i("UVRBench", msg)
        inst.sendStatus(0, Bundle().apply { putString("stream", msg + "\n") })
    }

    @Test
    fun separate() = runBlocking {
        val clip = File(ctx.getExternalFilesDir(null), "bench/clip.f32")
        val audio = DecodedAudio(clip, 44100, 2, clip.length() / 4)
        val name = InstrumentationRegistry.getArguments().getString("model") ?: "12_SP-UVR-3B-44100_vr.ptl"
        val model = ModelManager.availableModels.first { it.fileName == name }
        val seconds = audio.totalSamples / 2.0 / 44100
        val configs = listOf(
            "cpu" to SeparationParams(useGpu = false, tta = false),
            "gpu" to SeparationParams(useGpu = true, tta = false),
            "gpu+tta" to SeparationParams(useGpu = true, tta = true),
        )
        var reference: File? = null
        for ((label, params) in configs) {
            val t0 = System.currentTimeMillis()
            val result = Separator.create(model, params).separate(audio, ctx.cacheDir)
            val ms = System.currentTimeMillis() - t0
            val snr = reference?.let { snrDb(it, result.stems[0].file) }
            say("SEP $label: ${ms} ms for %.1f s audio (%.2fx realtime)%s".format(
                seconds, seconds * 1000 / ms, snr?.let { " primary vs cpu SNR=%.1f dB".format(it) } ?: ""))
            if (reference == null) reference = result.stems[0].file else result.stems[0].file.delete()
            result.stems[1].file.delete()
        }
        reference?.delete()
        Unit
    }

    private fun snrDb(a: File, b: File): Double {
        val x = read(a); val y = read(b)
        var s = 0.0; var e = 0.0
        for (i in 0 until minOf(x.size, y.size)) { s += x[i] * x[i].toDouble(); val d = x[i] - y[i]; e += d * d.toDouble() }
        return 10 * log10(s / maxOf(e, 1e-30))
    }

    private fun read(f: File): FloatArray = RandomAccessFile(f, "r").use { raf ->
        val bytes = ByteArray(raf.length().toInt()); raf.readFully(bytes)
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        FloatArray(fb.remaining()).also { fb.get(it) }
    }
}
