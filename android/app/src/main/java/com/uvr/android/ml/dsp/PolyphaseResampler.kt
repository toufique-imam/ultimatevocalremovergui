package com.uvr.android.ml.dsp

import kotlin.math.PI
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Port of `scipy.signal.resample_poly(x, up, down)` with its default
 * `('kaiser', 5.0)` window — what desktop UVR uses on Apple Silicon
 * (`librosa.resample(..., res_type='polyphase')`).
 *
 * Unlike scipy, samples are addressed by their *global* index in the whole
 * signal, so any output range can be computed from a slice of the input.
 * That lets the VR pipeline resample one segment at a time and still get
 * exactly the samples a whole-file resample would produce:
 *
 *   y[j] = Σ_k x[k] · h[(j + nPreRemove)·down − k·up]      (x = 0 outside [0, nIn))
 *
 * which is scipy's `upfirdn(h, x, up, down)[nPreRemove : nPreRemove + nOut]`.
 */
class PolyphaseResampler(fromRate: Int, toRate: Int) {

    val up: Int
    val down: Int

    /** True when the rates are equal — [process] is then a plain copy. */
    val isIdentity: Boolean

    private val h: FloatArray        // filter with scipy's n_pre_pad zeros in front
    private val nPreRemove: Long

    init {
        require(fromRate > 0 && toRate > 0) { "Invalid rates $fromRate → $toRate" }
        val g = gcd(fromRate, toRate)
        up = toRate / g
        down = fromRate / g
        isIdentity = up == 1 && down == 1

        if (isIdentity) {
            h = FloatArray(0)
            nPreRemove = 0
        } else {
            val maxRate = max(up, down)
            val halfLen = 10 * maxRate
            val taps = firwinKaiser(2 * halfLen + 1, 1.0 / maxRate, beta = 5.0)
            val nPrePad = down - halfLen % down
            h = FloatArray(nPrePad + taps.size)
            for (i in taps.indices) h[nPrePad + i] = taps[i].toFloat() * up
            nPreRemove = ((halfLen + nPrePad) / down).toLong()
        }
    }

    /** Output length for a whole signal of [nIn] samples: ceil(nIn · up / down). */
    fun outputLength(nIn: Long): Long =
        if (isIdentity) nIn else Math.floorDiv(nIn * up + down - 1, down.toLong())

    /** Input samples [start, end) needed to compute outputs [outStart, outEnd). */
    fun inputSpan(outStart: Long, outEnd: Long): LongRange {
        if (isIdentity) return outStart until outEnd
        val lo = ceilDiv((outStart + nPreRemove) * down - h.size + 1, up.toLong())
        val hi = Math.floorDiv((outEnd - 1 + nPreRemove) * down, up.toLong()) + 1
        return lo until hi
    }

    /** Outputs [start, end) that inputs [inStart, inEnd) contribute to. */
    fun outputSpan(inStart: Long, inEnd: Long): LongRange {
        if (isIdentity) return inStart until inEnd
        val lo = ceilDiv(inStart * up - nPreRemove * down, down.toLong())
        val hi = Math.floorDiv((inEnd - 1) * up - nPreRemove * down + h.size - 1, down.toLong()) + 1
        return lo until hi
    }

    /**
     * Compute outputs [outStart, outStart + out.size) into [out].
     *
     * [x] holds input samples [xStart, xStart + x.size); samples outside that
     * slice and outside [0, nIn) are treated as zero. For exact results the
     * slice must cover [inputSpan] of the requested outputs (clipped to the signal).
     */
    fun process(x: FloatArray, xStart: Long, nIn: Long, out: FloatArray, outStart: Long) {
        val xEnd = minOf(xStart + x.size, nIn)
        val xBegin = maxOf(xStart, 0L)
        if (isIdentity) {
            for (j in out.indices) {
                val k = outStart + j
                out[j] = if (k in xBegin until xEnd) x[(k - xStart).toInt()] else 0f
            }
            return
        }
        val upL = up.toLong()
        val downL = down.toLong()
        for (j in out.indices) {
            val m = (outStart + j + nPreRemove) * downL
            val kLo = maxOf(ceilDiv(m - h.size + 1, upL), xBegin)
            val kHi = minOf(Math.floorDiv(m, upL), xEnd - 1)
            var acc = 0.0
            var k = kLo
            while (k <= kHi) {
                acc += x[(k - xStart).toInt()] * h[(m - k * upL).toInt()]
                k++
            }
            out[j] = acc.toFloat()
        }
    }

    private companion object {
        fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

        fun ceilDiv(a: Long, b: Long): Long = -Math.floorDiv(-a, b)

        /**
         * `scipy.signal.firwin(numTaps, cutoff, window=('kaiser', beta))` —
         * low-pass, cutoff relative to Nyquist, scaled to unit DC gain.
         */
        fun firwinKaiser(numTaps: Int, cutoff: Double, beta: Double): DoubleArray {
            val alpha = 0.5 * (numTaps - 1)
            val i0Beta = besselI0(beta)
            val h = DoubleArray(numTaps)
            var sum = 0.0
            for (n in 0 until numTaps) {
                val m = n - alpha
                val x = cutoff * m
                val sinc = if (x == 0.0) 1.0 else sin(PI * x) / (PI * x)
                val r = (n - alpha) / alpha
                val w = besselI0(beta * sqrt(maxOf(0.0, 1.0 - r * r))) / i0Beta
                h[n] = cutoff * sinc * w
                sum += h[n]
            }
            for (n in h.indices) h[n] /= sum
            return h
        }

        /** Modified Bessel function of the first kind, order 0 (power series). */
        fun besselI0(x: Double): Double {
            var sum = 1.0
            var term = 1.0
            val q = x * x / 4.0
            var k = 1
            while (true) {
                term *= q / (k.toDouble() * k)
                sum += term
                if (term < sum * 1e-17) break
                k++
            }
            return sum
        }
    }
}
