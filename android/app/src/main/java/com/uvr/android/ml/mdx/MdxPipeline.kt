package com.uvr.android.ml.mdx

import com.uvr.android.ml.vr.MaskModel
import com.uvr.android.ml.vr.StereoSink
import com.uvr.android.ml.vr.StereoSource
import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.PI
import kotlin.math.cos

/**
 * Desktop UVR's MDX-Net separation (separate.py `SeperateMDX.demix` / `run_model`,
 * lib_v5/tfc_tdf_v3.py `STFT`) run as a stream over a 44.1 kHz stereo source.
 *
 * Desktop pads the song (trim zeros in front, up to a whole gen_size behind),
 * cuts it into chunks of hop·(dim_t−1) samples every [step] samples, and per chunk:
 *   torch.stft (center, reflect pad, periodic Hann) → first dim_f bins, bins 0–2
 *   zeroed → model (or with Denoise: ½·model(X) − ½·model(−X)) → torch.istft →
 *   × np.hanning(chunk) → overlap-add, also summing the window into a divider.
 * The result is result/divider, trimmed, × compensate; the secondary stem is
 * mix − primary.
 *
 * Here chunks are processed in order and each output sample is written as soon
 * as no later chunk can reach it, so memory is O(chunk) for any song length.
 */
class MdxPipeline(
    private val nFft: Int,
    private val dimF: Int,
    private val dimT: Int,
    private val compensate: Float,
    /** UVR "Overlap": null = "Default" (step = chunk − n_fft), else 0.25 … 0.99. */
    overlap: Float? = null,
    /** UVR "Denoise Output": also run the model on −X and average (2× model calls). */
    private val denoise: Boolean = false,
    private val hop: Int = 1024,
    private val log: (String) -> Unit = {}
) {
    private val nBins = nFft / 2 + 1
    private val trim = nFft / 2
    private val chunkSize = hop * (dimT - 1)
    private val genSize = chunkSize - 2 * trim
    private val step = if (overlap == null) chunkSize - nFft else ((1 - overlap) * chunkSize).toInt()

    private val fft = FloatFFT_1D(nFft.toLong())
    private val window = FloatArray(nFft) { (0.5 - 0.5 * cos(2.0 * PI * it / nFft)).toFloat() }

    /** torch.istft's window envelope Σ w² over the dim_t frames, for the trimmed output samples. */
    private val envelope = FloatArray(chunkSize).also { env ->
        for (t in 0 until dimT) for (i in 0 until nFft) {
            val s = t * hop + i - trim
            if (s in 0 until chunkSize) env[s] += window[i] * window[i]
        }
    }

    init {
        require(dimF <= nBins) { "dim_f $dimF > n_fft/2+1 = $nBins" }
        require(genSize > 0) { "Chunk ${chunkSize} too short for n_fft $nFft" }
        require(step > 0) { "Overlap leaves no step" }
    }

    fun run(
        source: StereoSource,
        model: MaskModel,
        primary: StereoSink,
        secondary: StereoSink,
        onProgress: (Float) -> Unit = {}
    ) {
        val n = source.length
        val pad = genSize + trim - (n % genSize)
        val mixLen = trim + n + pad              // desktop's padded `mixture`
        val totalChunks = ((mixLen + step - 1) / step).toInt()
        log("MDX n_fft=$nFft dim_f=$dimF dim_t=$dimT chunk=$chunkSize step=$step chunks=$totalChunks denoise=$denoise")

        val shape = longArrayOf(1, 4, dimF.toLong(), dimT.toLong())
        val spec = FloatArray(4 * dimF * dimT)
        val neg = if (denoise) FloatArray(spec.size) else null
        val chunkL = FloatArray(chunkSize)
        val chunkR = FloatArray(chunkSize)
        val waves = arrayOf(FloatArray(chunkSize), FloatArray(chunkSize))
        val acc = Accumulator(chunkSize, n, trim.toLong(), compensate, source, primary, secondary)

        var start = 0L
        var done = 0
        while (start < mixLen) {
            val end = minOf(start + chunkSize, mixLen)
            val actual = (end - start).toInt()
            // mixture[start, start + chunk) = source[start − trim, …); zero outside the song and past mixLen
            source.read(start - trim, chunkL, chunkR)
            if (actual < chunkSize) {
                chunkL.fill(0f, actual, chunkSize); chunkR.fill(0f, actual, chunkSize)
            }
            stft(chunkL, spec, 0)
            stft(chunkR, spec, 2)

            val out = if (neg == null) {
                model.predict(spec, shape).also { checkSize(it) }
            } else {
                for (k in spec.indices) neg[k] = -spec[k]
                val a = model.predict(neg, shape).also { checkSize(it) }
                val b = model.predict(spec, shape).also { checkSize(it) }
                FloatArray(spec.size) { -a[it] * 0.5f + b[it] * 0.5f }
            }
            istft(out, 0, waves[0])
            istft(out, 2, waves[1])

            acc.add(start, actual, waves)
            acc.flush(minOf(start + step, mixLen))
            start += step
            done++
            onProgress(done.toFloat() / totalChunks)
        }
        acc.flush(mixLen)
    }

    private fun checkSize(out: FloatArray) =
        check(out.size == 4 * dimF * dimT) { "Model output ${out.size} ≠ expected ${4 * dimF * dimT}" }

    // -------------------------------------------------------------------------
    // torch.stft / torch.istft as used by tfc_tdf_v3.STFT
    // -------------------------------------------------------------------------

    private val buf = FloatArray(nFft)

    /** STFT of one channel into spec channels [c0] (real) and [c0 + 1] (imag), bins < dim_f, bins 0–2 zeroed. */
    private fun stft(x: FloatArray, spec: FloatArray, c0: Int) {
        val reBase = c0 * dimF * dimT
        val imBase = (c0 + 1) * dimF * dimT
        for (t in 0 until dimT) {
            val s0 = t * hop - trim
            for (i in 0 until nFft) buf[i] = window[i] * x[reflect(s0 + i, x.size)]
            fft.realForward(buf)
            // JTransforms packing (even n): [Re0, Re(n/2), Re1, Im1, ...], same e^{-i} sign as torch.stft
            for (f in 0 until dimF) {
                val re: Float
                val im: Float
                when {
                    f < 3 -> { re = 0f; im = 0f }       // spek[:, :, :3, :] *= 0
                    f == nFft / 2 -> { re = buf[1]; im = 0f }
                    else -> { re = buf[2 * f]; im = buf[2 * f + 1] }
                }
                spec[reBase + f * dimT + t] = re
                spec[imBase + f * dimT + t] = im
            }
        }
    }

    /** torch.istft (center, length hop·(dim_t−1)) of spec channels [c0]/[c0 + 1], bins ≥ dim_f zero. */
    private fun istft(spec: FloatArray, c0: Int, y: FloatArray) {
        y.fill(0f)
        val reBase = c0 * dimF * dimT
        val imBase = (c0 + 1) * dimF * dimT
        for (t in 0 until dimT) {
            buf.fill(0f)
            buf[0] = spec[reBase + t]                                   // DC (imag ignored, as irfft does)
            if (dimF > nFft / 2) buf[1] = spec[reBase + (nFft / 2) * dimT + t]
            for (f in 1 until minOf(dimF, nFft / 2)) {
                buf[2 * f] = spec[reBase + f * dimT + t]
                buf[2 * f + 1] = spec[imBase + f * dimT + t]
            }
            fft.realInverse(buf, true)
            val s0 = t * hop - trim
            for (i in 0 until nFft) {
                val s = s0 + i
                if (s in 0 until chunkSize) y[s] += window[i] * buf[i]
            }
        }
        for (s in 0 until chunkSize) y[s] /= envelope[s]
    }

    /** torch 'reflect' padding index for a signal of [n] samples. */
    private fun reflect(i: Int, n: Int): Int = when {
        i < 0 -> -i
        i >= n -> 2 * n - 2 - i
        else -> i
    }

    // -------------------------------------------------------------------------
    // Windowed overlap-add with desktop's divider, flushed as samples finalise
    // -------------------------------------------------------------------------

    private class Accumulator(
        chunkSize: Int,
        private val songLength: Long,
        private val trim: Long,
        private val compensate: Float,
        private val source: StereoSource,
        private val primary: StereoSink,
        private val secondary: StereoSink
    ) {
        private var base = 0L      // mixture index of result[0]
        private val size = 2 * chunkSize
        private val resL = FloatArray(size)
        private val resR = FloatArray(size)
        private val div = FloatArray(size)
        private val hann = HashMap<Int, FloatArray>()

        fun add(start: Long, actual: Int, waves: Array<FloatArray>) {
            val w = hann.getOrPut(actual) { hanning(actual) }
            val o = (start - base).toInt()
            check(o >= 0 && o + actual <= size) { "Chunk at $start outside buffer (base $base)" }
            for (i in 0 until actual) {
                resL[o + i] += waves[0][i] * w[i]
                resR[o + i] += waves[1][i] * w[i]
                div[o + i] += w[i]
            }
        }

        /** Emit mixture samples [base, until): result/divider, trimmed to the song, × compensate. */
        fun flush(until: Long) {
            val count = (until - base).toInt()
            if (count <= 0) return
            // song sample s = mixture index − trim; keep s in [0, songLength)
            val s0 = maxOf(0L, base - trim)
            val s1 = minOf(songLength, until - trim)
            if (s1 > s0) {
                val m = (s1 - s0).toInt()
                val mixL = FloatArray(m)
                val mixR = FloatArray(m)
                source.read(s0, mixL, mixR)
                val pL = FloatArray(m); val pR = FloatArray(m)
                val sL = FloatArray(m); val sR = FloatArray(m)
                val o = (s0 + trim - base).toInt()
                for (i in 0 until m) {
                    pL[i] = resL[o + i] / div[o + i] * compensate
                    pR[i] = resR[o + i] / div[o + i] * compensate
                    sL[i] = mixL[i] - pL[i]
                    sR[i] = mixR[i] - pR[i]
                }
                primary.write(pL, pR, m)
                secondary.write(sL, sR, m)
            }
            shift(resL, count); shift(resR, count); shift(div, count)
            base = until
        }

        private fun shift(a: FloatArray, n: Int) {
            if (n >= a.size) { a.fill(0f); return }
            a.copyInto(a, 0, n, a.size)
            a.fill(0f, a.size - n)
        }

        /** np.hanning(n) (symmetric), computed in double as numpy does. */
        private fun hanning(n: Int): FloatArray =
            if (n == 1) floatArrayOf(1f)
            else FloatArray(n) { (0.5 - 0.5 * cos(2.0 * PI * it / (n - 1))).toFloat() }
    }
}
