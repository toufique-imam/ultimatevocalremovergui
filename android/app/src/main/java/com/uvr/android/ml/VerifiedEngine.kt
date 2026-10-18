package com.uvr.android.ml

import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import android.util.Log
import androidx.core.content.pm.PackageInfoCompat
import kotlin.math.abs

/**
 * Runs a candidate fast engine (e.g. GPU), but only if it is both accurate and
 * actually faster than the reference engine (the CPU model) on this device.
 *
 * The first window is run on both and timed. The fast engine is kept only if
 * its output is within [tolerance] of the reference (catches GPU drivers that
 * ignore the FP32 request) — or, with [minSnrDb], if its error is at least that many dB
 * below the reference's level (for engines allowed to run in FP16) — and it took less
 * time; otherwise the reference runs
 * the whole job. Either way every window of a run comes from one engine, and
 * the loser is closed. The decision is remembered under [decisionKey] ([decision]) so
 * later runs skip the comparison window — and callers can skip loading a fast engine
 * that already lost. With [persistIn] it survives app restarts, until the OS build
 * (and with it the GPU driver) or the app install changes.
 */
class VerifiedEngine(
    private var fast: InferenceEngine?,
    reference: () -> InferenceEngine,
    private val decisionKey: String? = null,
    private val tolerance: Float = 1e-3f,
    private val minSnrDb: Double? = null,
    /** With [minSnrDb]: windows whose reference output RMS is below this are too quiet to judge by level. */
    private val minLevel: Double = 0.0,
    private val tag: String = "VerifiedEngine"
) : InferenceEngine {

    private var referenceFactory: (() -> InferenceEngine)? = reference
    private var reference: InferenceEngine? = null
    private var active: InferenceEngine? = null

    /** Which engine ended up running the windows: "gpu", "cpu", or "pending". */
    var backend: String = "pending"
        private set

    override fun runBatch(data: FloatArray, shape: LongArray): FloatArray {
        active?.let { return it.runBatch(data, shape) }

        val fastEngine = fast
        if (fastEngine != null && decisionKey != null && decision(decisionKey) == true) {
            return use(fastEngine, "gpu").runBatch(data, shape)
        }
        val ref = reference ?: referenceFactory!!().also { reference = it; referenceFactory = null }
        if (fastEngine == null) return use(ref, "cpu").runBatch(data, shape)

        var t0 = System.nanoTime()
        val expected = ref.runBatch(data, shape)
        val refMs = (System.nanoTime() - t0) / 1_000_000
        if (minSnrDb != null && rms(expected) < minLevel) {
            // Silence (e.g. a song's first chunk) says nothing about the fast engine: judge a later window
            Log.i(tag, "Window too quiet to compare engines — reference output, checking the next one")
            return expected
        }
        t0 = System.nanoTime()
        val got = runCatching { fastEngine.runBatch(data, shape) }
            .onFailure { Log.w(tag, "Fast engine failed: ${it.message}") }
            .getOrNull()
        val fastMs = (System.nanoTime() - t0) / 1_000_000
        val err = maxAbsDiff(got, expected)
        val snr = snrDb(got, expected)

        val accurate = if (minSnrDb != null) snr >= minSnrDb else err <= tolerance
        val keepFast = accurate && fastMs < refMs
        Log.i(tag, "Fast engine: max|Δ| = $err, SNR = ${"%.1f".format(snr)} dB (reference RMS ${"%.3g".format(rms(expected))}), ${fastMs} ms vs reference ${refMs} ms → " +
                if (keepFast) "using fast engine" else "using reference")
        decisionKey?.let { remember(it, keepFast) }
        return if (keepFast) {
            ref.close()
            use(fastEngine, "gpu")
            got!!
        } else {
            fastEngine.close()
            fast = null
            use(ref, "cpu")
            expected
        }
    }

    private fun use(engine: InferenceEngine, name: String): InferenceEngine {
        referenceFactory = null
        reference = null
        active = engine
        backend = name
        return engine
    }

    private fun maxAbsDiff(got: FloatArray?, expected: FloatArray): Float {
        if (got == null || got.size != expected.size) return Float.POSITIVE_INFINITY
        var err = 0f
        for (i in got.indices) {
            val d = abs(got[i] - expected[i])
            if (d.isNaN()) return Float.POSITIVE_INFINITY
            if (d > err) err = d
        }
        return err
    }

    private fun rms(x: FloatArray): Double {
        var sum = 0.0
        for (v in x) sum += v.toDouble() * v
        return kotlin.math.sqrt(sum / x.size)
    }

    /** Reference level over error level in dB; +∞ for identical outputs, −∞ for a missing or NaN one. */
    private fun snrDb(got: FloatArray?, expected: FloatArray): Double {
        if (got == null || got.size != expected.size) return Double.NEGATIVE_INFINITY
        var signal = 0.0
        var noise = 0.0
        for (i in got.indices) {
            val d = (got[i] - expected[i]).toDouble()
            signal += expected[i].toDouble() * expected[i]
            noise += d * d
        }
        if (noise.isNaN()) return Double.NEGATIVE_INFINITY
        return 10 * kotlin.math.log10(signal / noise)
    }

    override val inputNames: Set<String> get() = setOf("input")
    override val outputNames: Set<String> get() = setOf("output")
    override fun expectedInputShape(inputName: String): LongArray? = active?.expectedInputShape(inputName)
    override fun expectedInputRank(inputName: String): Int = active?.expectedInputRank(inputName) ?: -1

    override fun close() {
        active?.close()
        if (active !== fast) fast?.close()
        if (reference !== active) reference?.close()
        active = null
        fast = null
        reference = null
    }

    companion object {
        private val decisions = java.util.concurrent.ConcurrentHashMap<String, Boolean>()
        @Volatile private var prefs: SharedPreferences? = null
        @Volatile private var installScope = ""

        /**
         * Keep decisions across app restarts. They are stored per OS build fingerprint and app
         * install, so an OS update (which can bring a new GPU driver) or a new app version
         * compares the engines again.
         */
        fun persistIn(context: Context) {
            val app = context.applicationContext
            val installed = app.packageManager.getPackageInfo(app.packageName, 0)
            installScope = "${Build.FINGERPRINT}|${PackageInfoCompat.getLongVersionCode(installed)}|${installed.lastUpdateTime}"
            prefs = app.getSharedPreferences("engine_decisions", Context.MODE_PRIVATE)
        }

        /** True/false once a run under [key] decided for/against the fast engine; null if not yet compared. */
        fun decision(key: String): Boolean? {
            decisions[key]?.let { return it }
            val stored = prefs?.takeIf { it.contains(scoped(key)) }?.getBoolean(scoped(key), false) ?: return null
            decisions[key] = stored
            return stored
        }

        private fun remember(key: String, useFast: Boolean) {
            decisions[key] = useFast
            prefs?.edit()?.putBoolean(scoped(key), useFast)?.apply()
        }

        private fun scoped(key: String) = "$installScope|$key"
    }
}
