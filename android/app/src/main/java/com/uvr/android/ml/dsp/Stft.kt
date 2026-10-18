package com.uvr.android.ml.dsp

import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.PI
import kotlin.math.cos

/**
 * Frame-level STFT / ISTFT matching `librosa.stft` / `librosa.istft` with their
 * defaults as used by UVR: periodic Hann window, win_length = n_fft,
 * `center=True` with zero padding (`pad_mode='constant'`).
 *
 * Frames are addressed by global index: frame f covers samples
 * [f·hop − nFft/2, f·hop + nFft/2) of the unpadded signal. This lets callers
 * compute any subset of frames from a slice of the signal, and overlap-add
 * any subset of frames, then normalise with the window sum of the *whole*
 * signal ([windowSumSquare]) — giving the same samples as a whole-file ISTFT.
 *
 * Not thread-safe (owns a scratch buffer).
 */
class Stft(val nFft: Int, val hop: Int) {

    val bins = nFft / 2 + 1
    private val half = nFft / 2
    private val fft = FloatFFT_1D(nFft.toLong())
    private val window = FloatArray(nFft) { (0.5 - 0.5 * cos(2.0 * PI * it / nFft)).toFloat() }
    private val windowSq = FloatArray(nFft) { window[it] * window[it] }
    private val buf = FloatArray(nFft)

    /** Number of frames librosa produces for a signal of [n] samples. */
    fun frameCount(n: Long): Long = 1 + n / hop

    /** First sample (inclusive) touched by frame [f]. */
    fun frameStart(f: Long): Long = f * hop - half

    /**
     * Analyse frame [f] of a signal whose samples [xStart, xStart + x.size) are
     * in [x] (samples outside [0, n) are zero). Writes bins 0..nFft/2 into
     * [re]/[im] at `offset + bin * stride`.
     */
    fun analyze(
        x: FloatArray, xStart: Long, n: Long, f: Long,
        re: FloatArray, im: FloatArray, offset: Int, stride: Int
    ) {
        val s0 = frameStart(f)
        for (i in 0 until nFft) {
            val s = s0 + i
            buf[i] = if (s < 0 || s >= n) 0f else window[i] * x[(s - xStart).toInt()]
        }
        fft.realForward(buf)
        // JTransforms packing (even n): [Re0, Re(n/2), Re1, Im1, Re2, Im2, ...]
        re[offset] = buf[0]; im[offset] = 0f
        re[offset + half * stride] = buf[1]; im[offset + half * stride] = 0f
        for (k in 1 until half) {
            re[offset + k * stride] = buf[2 * k]
            im[offset + k * stride] = buf[2 * k + 1]   // JTransforms uses numpy's e^{-i} sign
        }
    }

    /**
     * Inverse-transform one frame (bins at `offset + bin * stride`), apply the
     * synthesis window and add it into [y], which holds samples
     * [yStart, yStart + y.size). Samples falling outside [y] are dropped.
     * Imaginary parts of the DC and Nyquist bins are ignored, as numpy's irfft does.
     */
    fun overlapAdd(
        re: FloatArray, im: FloatArray, offset: Int, stride: Int,
        f: Long, y: FloatArray, yStart: Long
    ) {
        buf[0] = re[offset]
        buf[1] = re[offset + half * stride]
        for (k in 1 until half) {
            buf[2 * k] = re[offset + k * stride]
            buf[2 * k + 1] = im[offset + k * stride]
        }
        fft.realInverse(buf, true)
        val s0 = frameStart(f)
        for (i in 0 until nFft) {
            val idx = s0 + i - yStart
            if (idx >= 0 && idx < y.size) y[idx.toInt()] += window[i] * buf[i]
        }
    }

    /**
     * librosa's `window_sumsquare` for a signal of [nFrames] frames, evaluated at
     * unpadded sample [s] (i.e. padded index s + nFft/2).
     */
    fun windowSumSquare(s: Long, nFrames: Long): Float {
        // frames f with f·hop ≤ s + half < f·hop + nFft
        val p = s + half
        val fHi = minOf(Math.floorDiv(p, hop.toLong()), nFrames - 1)
        var f = maxOf(0L, Math.floorDiv(p - nFft, hop.toLong()) + 1)
        var sum = 0f
        while (f <= fHi) {
            sum += windowSq[(p - f * hop).toInt()]
            f++
        }
        return sum
    }
}
