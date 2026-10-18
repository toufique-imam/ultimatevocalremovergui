package com.uvr.android.ml.demucs

import com.uvr.android.ml.vr.StereoSink
import com.uvr.android.ml.vr.StereoSource
import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.sqrt

/** HTDemucs network core ([DemucsModelMeta] ONNX): mix [1,2,TL] + CaC spectrogram [1,4,F,T] → (spec [1,S,4,F,T], wave [1,S,2,TL]). */
fun interface DemucsCore {
    fun predict(mix: FloatArray, mag: FloatArray): Pair<FloatArray, FloatArray>
}

/** Writes one sub-model's (weighted) stems; the pipeline hands it every output sample once, in order. */
fun interface StemsSink {
    /** [stems][channel][i] for i < count, S stems in the model's source order. */
    fun write(stems: Array<Array<FloatArray>>, count: Int)
}

/**
 * Desktop UVR's Demucs v4 separation — separate.py `SeperateDemucs.demix_demucs`
 * with demucs/apply.py `apply_model` and demucs/htdemucs.py `HTDemucs.forward`
 * (the network core runs as ONNX; its STFT/iSTFT are done here) — streamed over
 * a 44.1 kHz stereo source for one sub-model of a bag.
 *
 * The input is normalised by the mono reference's mean/std ([normalization]),
 * split into segments of [DemucsModelMeta.segmentSamples] every
 * (1 − overlap)·segment samples, each run through the model and overlap-added
 * with desktop's triangle weights. With `shifts` > 0 this is repeated on the
 * input shifted by fixed offsets up to 0.5 s ([shiftOffset]; desktop draws them
 * at random) and the results averaged. Every (shift) stream keeps a rolling
 * accumulator, and samples are emitted once no stream can still change them.
 */
class DemucsPipeline(
    private val meta: DemucsModelMeta,
    private val overlap: Float = 0.25f,
    private val shifts: Int = 2,
    private val log: (String) -> Unit = {}
) {
    private val tl = meta.segmentSamples
    private val hop = meta.hop
    private val nFft = meta.nFft
    private val bins = nFft / 2                                 // HTDemucs drops the Nyquist bin
    private val frames = ceil(tl.toDouble() / hop).toInt()      // `le`
    private val specPad = hop / 2 * 3
    private val nStems = meta.sources.size
    private val segment = tl                                    // int(samplerate · segment)
    private val stride = ((1 - overlap) * segment).toInt()
    private val maxShift = (0.5 * meta.sampleRate).toInt()

    private val fft = FloatFFT_1D(nFft.toLong())
    private val window = FloatArray(nFft) { (0.5 - 0.5 * cos(2.0 * PI * it / nFft)).toFloat() }
    private val norm = (1.0 / sqrt(nFft.toDouble())).toFloat()

    /** apply_model's transition weight: a triangle peaking mid-segment, / max. */
    private val weight: FloatArray = run {
        val w = FloatArray(segment)
        for (i in 0 until segment / 2) w[i] = (i + 1).toFloat()
        for (i in 0 until segment - segment / 2) w[segment / 2 + i] = (segment - segment / 2 - i).toFloat()
        val max = w.max()
        FloatArray(segment) { w[it] / max }
    }

    /** torch.istft window envelope Σ w² at each kept output sample (all le+4 frames). */
    private val envelope = FloatArray(tl).also { env ->
        val first = specPad + nFft / 2                          // output n ↔ OLA index n + first
        for (t in 0 until frames + 4) for (i in 0 until nFft) {
            val n = t * hop + i - first
            if (n in 0 until tl) env[n] += window[i] * window[i]
        }
    }

    init {
        require(hop == nFft / 4) { "HTDemucs needs hop = n_fft / 4" }
        require(stride > 0) { "Overlap $overlap leaves no stride" }
    }

    /**
     * Run one sub-model over [source] (already normalised, see [normalization])
     * and hand its stems, averaged over shifts, to [sink].
     */
    fun run(source: StereoSource, core: DemucsCore, sink: StemsSink, onProgress: (Float) -> Unit = {}) {
        val n = source.length
        val streams = if (shifts > 0) (0 until shifts).map { Stream(n, shiftOffset(it, shifts, maxShift)) }
            else listOf(Stream(n, null))
        val total = streams.sumOf { it.chunkCount }
        log("Demucs segment=$segment stride=$stride shifts=$shifts chunks=$total stems=$nStems")

        val chunkMix = FloatArray(2 * tl)
        val left = FloatArray(tl)
        val right = FloatArray(tl)
        var done = 0
        var emitted = 0L
        while (true) {
            // Next chunk in mix order across streams (ties: lower shift index first)
            val s = streams.filter { it.hasNext }.minByOrNull { it.nextMixStart } ?: break
            val (inputStart, cl, viewStart) = s.next()
            source.read(inputStart, left, right)
            left.copyInto(chunkMix, 0); right.copyInto(chunkMix, tl)
            val out = forward(core, chunkMix)
            s.add(out, viewStart, cl)
            done++
            onProgress(done.toFloat() / total)
            val ready = streams.minOf { it.finalUntil }
            if (ready > emitted) { emit(streams, emitted, ready, sink); emitted = ready }
        }
        if (n > emitted) emit(streams, emitted, n, sink)
    }

    /** apply_model: out = Σ_shifts (acc / Σweight) then / shifts, over mix samples [from, to). */
    private fun emit(streams: List<Stream>, from: Long, to: Long, sink: StemsSink) {
        val count = (to - from).toInt()
        val out = Array(nStems) { Array(2) { FloatArray(count) } }
        for (st in streams) st.addNormalizedTo(out, from, count)
        if (shifts > 0) for (k in 0 until nStems) for (c in 0 until 2) {
            val a = out[k][c]
            for (i in 0 until count) a[i] /= shifts.toFloat()
        }
        sink.write(out, count)
        for (st in streams) st.drop(to)
    }

    /**
     * One apply_model split pass: over the whole mix (no shifts) or over the
     * shifted view padded_mix[offset, L + max_shift) of the zero-padded mix.
     * Chunk j of the view covers view samples [j·stride, j·stride + cl) = mix
     * samples [viewOrigin + j·stride, …); apply_model(split=False) centres it in a
     * TL-sample window of real neighbouring samples (TensorChunk.padded) and
     * center-trims the output.
     */
    private inner class Stream(mixLength: Long, shift: Int?) {
        /** Mix index of view sample 0. */
        private val viewOrigin = if (shift == null) 0L else (shift - maxShift).toLong()
        private val viewLength = if (shift == null) mixLength else mixLength + maxShift - shift
        private val mixLength = mixLength
        val chunkCount = ((viewLength + stride - 1) / stride).toInt()
        private var nextChunk = 0

        // Rolling accumulator in mix coordinates, starting at `base`
        private var base = 0L
        private var acc = Array(nStems) { Array(2) { FloatArray(0) } }
        private var sumW = FloatArray(0)

        val hasNext get() = nextChunk < chunkCount
        val nextMixStart get() = viewOrigin + nextChunk.toLong() * stride
        /** Mix samples below this can't change any more. */
        val finalUntil: Long get() = if (hasNext) maxOf(0L, minOf(mixLength, nextMixStart)) else mixLength

        /** (input start in mix coords, chunk length, chunk start in mix coords) of the next chunk. */
        fun next(): Triple<Long, Int, Long> {
            val j = nextChunk.toLong() * stride
            nextChunk++
            val cl = minOf(viewLength - j, segment.toLong()).toInt()
            val delta = tl - cl
            return Triple(viewOrigin + j - delta / 2, cl, viewOrigin + j)
        }

        /** out[stem][ch] is the model's TL-sample output; keep its centre cl samples, weighted. */
        fun add(out: Array<Array<FloatArray>>, mixStart: Long, cl: Int) {
            val trim = (tl - cl) / 2
            // Only mix samples [0, L) are ever emitted; the shifted view's lead-in is dropped
            val from = maxOf(mixStart, base, 0L)
            val to = minOf(mixStart + cl, mixLength)
            if (to <= from) return
            ensure((to - base).toInt())
            for (g in from until to) {
                val i = (g - mixStart).toInt()
                val o = (g - base).toInt()
                val w = weight[i]
                sumW[o] += w
                for (k in 0 until nStems) for (c in 0 until 2) acc[k][c][o] += w * out[k][c][trim + i]
            }
        }

        fun addNormalizedTo(dst: Array<Array<FloatArray>>, from: Long, count: Int) {
            ensure((from + count - base).toInt())
            val o = (from - base).toInt()
            for (k in 0 until nStems) for (c in 0 until 2) {
                val a = acc[k][c]; val d = dst[k][c]
                for (i in 0 until count) d[i] += a[o + i] / sumW[o + i]
            }
        }

        fun drop(until: Long) {
            val n = (until - base).toInt()
            if (n <= 0) return
            val keep = maxOf(0, sumW.size - n)
            sumW = sumW.copyOfRange(minOf(n, sumW.size), sumW.size).copyOf(keep)
            acc = Array(nStems) { k -> Array(2) { c -> acc[k][c].copyOfRange(minOf(n, acc[k][c].size), acc[k][c].size) } }
            base = until
        }

        private fun ensure(size: Int) {
            if (sumW.size >= size) return
            val cap = maxOf(size, segment + stride)
            sumW = sumW.copyOf(cap)
            acc = Array(nStems) { k -> Array(2) { c -> acc[k][c].copyOf(cap) } }
        }
    }

    // -------------------------------------------------------------------------
    // HTDemucs.forward around the ONNX core: _spec → _magnitude → core → _mask → _ispec
    // -------------------------------------------------------------------------

    private val buf = FloatArray(nFft)

    /** Returns [stem][channel][TL] = _ispec(_mask(z, spec)) + wave. */
    private fun forward(core: DemucsCore, mix: FloatArray): Array<Array<FloatArray>> {
        val mag = FloatArray(4 * bins * frames)
        for (c in 0 until 2) spec(mix, c * tl, mag, 2 * c)
        val (specOut, waveOut) = core.predict(mix, mag)
        check(specOut.size == nStems * 4 * bins * frames) { "spec output ${specOut.size}" }
        check(waveOut.size == nStems * 2 * tl) { "wave output ${waveOut.size}" }
        return Array(nStems) { k ->
            Array(2) { c ->
                val y = ispec(specOut, (k * 4 + 2 * c) * bins * frames, (k * 4 + 2 * c + 1) * bins * frames)
                val w = (k * 2 + c) * tl
                for (i in 0 until tl) y[i] += waveOut[w + i]
                y
            }
        }
    }

    /**
     * HTDemucs._spec of one channel (x[off, off+TL)): reflect-pad by 3·hop/2 (and up
     * to a whole number of hops), torch.stft(center, reflect, normalized), drop the
     * Nyquist bin and frames 0–1 / last two. Writes re/im to mag channels [c0]/[c0+1].
     */
    private fun spec(x: FloatArray, off: Int, mag: FloatArray, c0: Int) {
        val padRight = specPad + frames * hop - tl
        val xpLen = tl + specPad + padRight
        val reBase = c0 * bins * frames
        val imBase = (c0 + 1) * bins * frames
        for (t in 0 until frames) {
            val f = t + 2                                        // z[..., 2 : 2 + le]
            val s0 = f * hop - nFft / 2                         // index into xp (stft's own reflect pad)
            for (i in 0 until nFft) {
                val p = reflect(s0 + i, xpLen)                  // stft center padding
                val q = reflect(p - specPad, tl)                // pad1d
                buf[i] = window[i] * x[off + q]
            }
            fft.realForward(buf)
            for (b in 0 until bins) {
                val re = if (b == 0) buf[0] else buf[2 * b]
                val im = if (b == 0) 0f else buf[2 * b + 1]
                mag[reBase + b * frames + t] = re * norm
                mag[imBase + b * frames + t] = im * norm
            }
        }
    }

    /** HTDemucs._ispec of one stem/channel (CaC re/im planes) → TL samples. */
    private fun ispec(z: FloatArray, reOff: Int, imOff: Int): FloatArray {
        val y = FloatArray(tl)
        val first = specPad + nFft / 2
        val scale = 1f / norm                                   // istft(normalized=True)
        for (t in 0 until frames) {
            buf.fill(0f)
            buf[0] = z[reOff + t]                               // DC; Nyquist bin is the zero pad
            for (b in 1 until bins) {
                buf[2 * b] = z[reOff + b * frames + t]
                buf[2 * b + 1] = z[imOff + b * frames + t]
            }
            fft.realInverse(buf, true)
            val s0 = (t + 2) * hop - first                      // 2 zero frames padded in front
            for (i in 0 until nFft) {
                val n = s0 + i
                if (n in 0 until tl) y[n] += window[i] * buf[i] * scale
            }
        }
        for (i in 0 until tl) y[i] /= envelope[i]
        return y
    }

    private fun reflect(i: Int, n: Int): Int = when {
        i < 0 -> -i
        i >= n -> 2 * n - 2 - i
        else -> i
    }

    companion object {
        /** Fixed stand-in for apply_model's random.randint(0, max_shift) for shift [i] of [count]. */
        fun shiftOffset(i: Int, count: Int, maxShift: Int): Int = (i + 1) * maxShift / (count + 1)

        /**
         * demix_demucs's normalisation: mean and (unbiased) std of the mono
         * reference (L + R) / 2 over the whole song.
         */
        fun normalization(source: StereoSource, block: Int = 1 shl 16): Pair<Float, Float> {
            val n = source.length
            val l = FloatArray(block); val r = FloatArray(block)
            var sum = 0.0
            var pos = 0L
            while (pos < n) {
                source.read(pos, l, r)
                val m = minOf(block.toLong(), n - pos).toInt()
                for (i in 0 until m) sum += ((l[i] + r[i]) / 2f).toDouble()
                pos += block
            }
            val mean = sum / n
            var sq = 0.0
            pos = 0L
            while (pos < n) {
                source.read(pos, l, r)
                val m = minOf(block.toLong(), n - pos).toInt()
                for (i in 0 until m) { val d = ((l[i] + r[i]) / 2f) - mean; sq += d * d }
                pos += block
            }
            return mean.toFloat() to sqrt(sq / (n - 1)).toFloat()
        }
    }
}
