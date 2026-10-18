package com.uvr.android.ml.vr

import com.uvr.android.ml.dsp.PolyphaseResampler
import com.uvr.android.ml.dsp.Stft
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.pow
import kotlin.math.sqrt

/** Random-access stereo PCM input. Samples outside [0, length) read as zero. */
interface StereoSource {
    val length: Long
    fun read(start: Long, left: FloatArray, right: FloatArray)
}

/** Sequential stereo PCM output. */
interface StereoSink {
    fun write(left: FloatArray, right: FloatArray, count: Int)
}

/** Runs the VR network on one window: input/output are [1, 2, bins+1, frames], row-major. */
fun interface MaskModel {
    fun predict(input: FloatArray, shape: LongArray): FloatArray
}

/**
 * Desktop UVR's VR-architecture separation (separate.py `SeperateVR` +
 * lib_v5/spec_utils.py) re-implemented to run in bounded memory on any
 * input length.
 *
 * Desktop processes the whole song at once:
 *   multi-band resample → STFT → crop/stack into one (bins+1)-row spectrogram →
 *   |X| / max|X| → model on 512-frame windows (roi = 256 frames each) → mask →
 *   aggressiveness → mask·X and (1−mask)·X → per-band ISTFT → resample up → sum.
 *
 * Here the song is split into segments of [windowsPerSegment] model windows,
 * aligned to the same global window grid desktop uses. Every stage is indexed
 * by global sample/frame position:
 *  - analysis of a segment reads just the input slice its frames (plus
 *    [VrModelMeta.offset] frames of model context each side) depend on;
 *  - synthesis is linear, so each segment's frames are overlap-added into a
 *    small rolling buffer and flushed once no later segment can touch them.
 * The result matches whole-file processing to float rounding, with no seams.
 *
 * Two passes over the input: the first only computes max|X| (desktop
 * normalises the model input by the whole-song maximum), the second
 * separates. Heap use is O(segment), independent of song length.
 *
 * The DSP runs beside the model: the next segment is analysed and the
 * previous one synthesised (primary and secondary on their own threads) while
 * the model works through the current one, and pass 1 is split over
 * [dspThreads] threads. Each thread has its own [Stft]s; every stem's chunks
 * are still added in order, so the output is identical to running serially.
 *
 * With [tta] (desktop "TTA"), a second grid of windows shifted by roi/2 is run
 * and the two masks are averaged, as inference_vr does. The shifted window that
 * straddles two segments is run once and reused.
 */
class VrPipeline(
    private val meta: VrModelMeta,
    private val inputSampleRate: Int,
    private val windowSize: Int = 512,
    /** UVR "Aggression Setting" / 100 (desktop default 5 → 0.05). */
    private val aggression: Float = 0.05f,
    private val windowsPerSegment: Int = 4,
    /** UVR "TTA": average with a half-window-shifted pass (2× model calls). */
    private val tta: Boolean = false,
    private val dspThreads: Int = 3,
    private val log: (String) -> Unit = {}
) {
    private val p = meta.params
    private val bands = p.bands
    private val nb = bands.size
    private val rows = p.bins + 1
    private val offset = meta.offset
    private val roi = (windowSize - 2 * offset).let { if (it == 0) windowSize else it }
    /** Shift of the TTA window grid (desktop pads roi // 2 extra frames on the left). */
    private val ttaShift = roi / 2
    /** Spectrogram frames a segment needs beyond its core, each side. */
    private val context = offset + if (tta) ttaShift else 0

    /** Sample rate of the separated stems (the model's top band rate). */
    val outputSampleRate: Int = bands.last().sr

    /** STFTs for geometry and the calling thread; worker threads make their own (Stft isn't thread-safe). */
    private val stfts = newStfts()
    private fun newStfts() = bands.map { Stft(it.nFft, it.hl) }
    private val inRes = PolyphaseResampler(inputSampleRate, outputSampleRate)
    private val downRes = Array(nb - 1) { d -> PolyphaseResampler(bands[d + 1].sr, bands[d].sr) }
    private val upRes = Array(nb - 1) { d -> PolyphaseResampler(bands[d].sr, bands[d + 1].sr) }

    /** First combined-spectrogram row of each band. */
    private val rowOffset = IntArray(nb)
    /** combine_spectrograms' pre-filter gain per combined row. */
    private val preFilterGain = FloatArray(rows) { 1f }
    /** cmb_spectrogram_to_wave's per-band crop + hp/lp filter gain per band bin. */
    private val bandGain: Array<FloatArray>

    init {
        require(!meta.isVr51) { "VR 5.1 models (nets_new) are not supported yet" }
        require(!p.reverse) { "Model params with 'reverse' are not supported" }
        require(roi > 0) { "Window size $windowSize must be larger than ${2 * offset}" }
        require(bands.last().sr == p.sr) { "Top band rate ${bands.last().sr} ≠ model rate ${p.sr}" }

        var off = 0
        for (d in 0 until nb) {
            rowOffset[d] = off
            off += bands[d].cropStop - bands[d].cropStart
        }
        require(off <= p.bins) { "Too many bins ($off > ${p.bins})" }

        if (p.preFilterStart > 0) {
            if (nb == 1) {
                lowPass(preFilterGain, p.preFilterStart, p.preFilterStop)
            } else {
                var gp = 1.0
                for (b in p.preFilterStart + 1 until p.preFilterStop) {
                    val g = 10.0.pow(-(b - p.preFilterStart) * (3.5 - gp) / 20.0)
                    gp = g
                    preFilterGain[b] = (preFilterGain[b] * g).toFloat()
                }
            }
        }

        bandGain = Array(nb) { d ->
            val b = bands[d]
            val gain = FloatArray(b.nFft / 2 + 1)
            for (r in b.cropStart until b.cropStop) gain[r] = 1f
            when {
                d == nb - 1 -> if (b.hpfStart > 0) highPass(gain, b.hpfStart, b.hpfStop - 1)
                d == 0 -> lowPass(gain, b.lpfStart, b.lpfStop)
                else -> { highPass(gain, b.hpfStart, b.hpfStop - 1); lowPass(gain, b.lpfStart, b.lpfStop) }
            }
            gain
        }
    }

    // Whole-signal lengths, set by run()
    private var nIn = 0L
    private val nBand = LongArray(nb)      // analysis signal length per band
    private val nSynth = LongArray(nb)     // ISTFT output length per band = hop·(N−1)
    private var nFrames = 0L

    /**
     * Separate [source] (at [inputSampleRate]) into [primary] and [secondary]
     * stems at [outputSampleRate]. [onProgress] receives values in [0, 1]; it may
     * throw (e.g. CancellationException) to abort.
     */
    fun run(
        source: StereoSource,
        model: MaskModel,
        primary: StereoSink,
        secondary: StereoSink,
        onProgress: (Float) -> Unit = {}
    ) {
        nIn = source.length
        nBand[nb - 1] = inRes.outputLength(nIn)
        for (d in nb - 2 downTo 0) nBand[d] = downRes[d].outputLength(nBand[d + 1])
        nFrames = (0 until nb).minOf { stfts[it].frameCount(nBand[it]) }
        for (d in 0 until nb) nSynth[d] = bands[d].hl * (nFrames - 1)
        for (d in 0 until nb - 1) {
            check(upRes[d].outputLength(nSynth[d]) == nSynth[d + 1]) { "Band lengths do not line up" }
        }
        val outLength = nSynth[nb - 1]
        val mainWindows = ((nFrames + roi - 1) / roi).toInt()
        val segments = (0 until mainWindows step windowsPerSegment)
            .map { w0 -> w0 until minOf(w0 + windowsPerSegment, mainWindows) }
        val totalWindows = mainWindows + if (tta) ttaWindows(0, nFrames).count() else 0
        log("frames=$nFrames windows=$totalWindows segments=${segments.size} tta=$tta " +
            "bands=$nb inSr=$inputSampleRate outSr=$outputSampleRate outLen=$outLength")

        val pool = Executors.newFixedThreadPool(maxOf(1, dspThreads))
        val analysisThread = Executors.newSingleThreadExecutor()
        val primThread = Executors.newSingleThreadExecutor()
        val secThread = Executors.newSingleThreadExecutor()
        try {
            // ---- Pass 1: max |X| over the whole song ---------------------------
            val workerStfts = ThreadLocal.withInitial { newStfts() }
            val maxima = segments.map { seg ->
                pool.submit(Callable {
                    val (c0, c1) = coreFrames(seg)
                    val spec = analyze(source, c0, c1, workerStfts.get())
                    var m = 0f
                    for (k in spec.re.indices) {
                        val v = sqrt(spec.re[k] * spec.re[k] + spec.im[k] * spec.im[k])
                        if (v > m) m = v
                    }
                    m
                })
            }
            var maxMag = 0f
            for ((i, f) in maxima.withIndex()) {
                maxMag = maxOf(maxMag, await(f))
                onProgress(0.1f * (i + 1) / segments.size)
            }
            log("max|X| = $maxMag")
            val norm = if (maxMag > 0f) 1f / maxMag else 1f

            // ---- Pass 2: separate -------------------------------------------------
            val primAcc = OverlapAccumulator(outLength, primary)
            val secAcc = OverlapAccumulator(outLength, secondary)
            val analysisStfts = newStfts()
            val primStfts = newStfts()
            val secStfts = newStfts()
            val input = FloatArray(2 * rows * windowSize)
            val shape = longArrayOf(1, 2, rows.toLong(), windowSize.toLong())
            var windowsDone = 0
            // Last TTA window of the previous segment, reused when the next segment starts with it
            var ttaCache: Pair<Long, FloatArray>? = null

            fun predict(start: Long, spec: Spec, fa: Long, fb: Long): FloatArray {
                fillModelInput(spec, fa, fb, start, norm, input)
                val out = model.predict(input, shape)
                check(out.size == input.size) { "Model output ${out.size} ≠ expected ${input.size}" }
                windowsDone++
                onProgress(0.1f + 0.9f * windowsDone / totalWindows)
                return out
            }

            fun analyzeSegment(i: Int): Future<Spec> = analysisThread.submit(Callable {
                val (c0, c1) = coreFrames(segments[i])
                analyze(source, maxOf(0L, c0 - context), minOf(nFrames, c1 + context), analysisStfts)
            })

            var nextSpec = analyzeSegment(0)
            var pending: List<Future<*>> = emptyList()
            for ((i, seg) in segments.withIndex()) {
                val (c0, c1) = coreFrames(seg)
                val fa = maxOf(0L, c0 - context)
                val fb = minOf(nFrames, c1 + context)
                val spec = await(nextSpec)
                if (i + 1 < segments.size) nextSpec = analyzeSegment(i + 1)
                val coreLen = (c1 - c0).toInt()
                val mask = FloatArray(2 * rows * coreLen)

                for (w in seg) {
                    val start = w.toLong() * roi - offset
                    val out = predict(start, spec, fa, fb)
                    for (c in 0 until 2) for (r in 0 until rows) {
                        val src = (c * rows + r) * windowSize
                        val dst = (c * rows + r) * coreLen
                        for (t in offset until offset + roi) {
                            val g = start + t
                            if (g >= c1) break
                            mask[dst + (g - c0).toInt()] = out[src + t]
                        }
                    }
                }

                if (tta) {
                    // inference_vr: mask = (mask + mask_tta) * 0.5
                    for (j in ttaWindows(c0, c1)) {
                        val start = j * roi - offset - ttaShift
                        val out = ttaCache?.takeIf { it.first == j }?.second
                            ?: predict(start, spec, fa, fb).also { ttaCache = j to it }
                        for (c in 0 until 2) for (r in 0 until rows) {
                            val src = (c * rows + r) * windowSize
                            val dst = (c * rows + r) * coreLen
                            for (t in offset until offset + roi) {
                                val g = start + t
                                if (g < c0) continue
                                if (g >= c1) break
                                val k = dst + (g - c0).toInt()
                                mask[k] = (mask[k] + out[src + t]) * 0.5f
                            }
                        }
                    }
                }

                adjustAggressiveness(mask, coreLen)
                // At most one segment waits for synthesis while the next one runs on the model
                pending.forEach { await(it) }
                val safe = if (c1 >= nFrames) outLength else contributionStart(c1)
                pending = listOf(
                    primThread.submit {
                        synthesize(spec, fa, mask, c0, c1, isPrimary = true, primStfts)?.let { primAcc.add(it) }
                        primAcc.flush(safe)
                    },
                    secThread.submit {
                        synthesize(spec, fa, mask, c0, c1, isPrimary = false, secStfts)?.let { secAcc.add(it) }
                        secAcc.flush(safe)
                    }
                )
            }
            pending.forEach { await(it) }
            primAcc.flush(outLength)
            secAcc.flush(outLength)
        } finally {
            for (e in listOf(pool, analysisThread, primThread, secThread)) e.shutdownNow()
        }
    }

    private fun <T> await(f: Future<T>): T = try {
        f.get()
    } catch (e: ExecutionException) {
        throw e.cause ?: e
    }

    /** Shifted-grid windows whose roi covers frames of [c0, c1): window j's roi is [j·roi − roi/2, j·roi + roi/2). */
    private fun ttaWindows(c0: Long, c1: Long): LongRange =
        (c0 + ttaShift) / roi..(c1 - 1 + ttaShift) / roi

    private fun coreFrames(windows: IntRange): Pair<Long, Long> =
        windows.first.toLong() * roi to minOf(nFrames, (windows.last + 1).toLong() * roi)

    // -------------------------------------------------------------------------
    // Analysis: frames [fa, fb) of the combined spectrogram
    // -------------------------------------------------------------------------

    /** Combined complex spectrogram, layout [channel][row][frame] (frame fastest). */
    private class Spec(val frames: Int, rows: Int) {
        val re = FloatArray(2 * rows * frames)
        val im = FloatArray(2 * rows * frames)
    }

    private fun analyze(source: StereoSource, fa: Long, fb: Long, stfts: List<Stft>): Spec {
        val frames = (fb - fa).toInt()
        val spec = Spec(frames, rows)
        if (frames <= 0) return spec

        // Sample range each band's frames read, then what that requires from the band above.
        val req = Array(nb) { d ->
            val s = stfts[d]
            maxOf(0L, s.frameStart(fa)) to minOf(nBand[d], s.frameStart(fb - 1) + s.nFft)
        }
        for (d in 1 until nb) {
            val below = downRes[d - 1].inputSpan(req[d - 1].first, req[d - 1].second)
            req[d] = maxOf(0L, minOf(req[d].first, below.first)) to
                minOf(nBand[d], maxOf(req[d].second, below.last + 1))
        }
        val top = req[nb - 1]
        val inSpan = inRes.inputSpan(top.first, top.second)
        val inStart = maxOf(0L, inSpan.first)
        val inEnd = minOf(nIn, inSpan.last + 1)
        val inL = FloatArray((inEnd - inStart).toInt())
        val inR = FloatArray(inL.size)
        synchronized(source) { source.read(inStart, inL, inR) }

        // Band waves, top band first (each lower band is resampled from the one above, as desktop does).
        val waveL = arrayOfNulls<FloatArray>(nb)
        val waveR = arrayOfNulls<FloatArray>(nb)
        waveL[nb - 1] = resample(inRes, inL, inStart, nIn, top)
        waveR[nb - 1] = resample(inRes, inR, inStart, nIn, top)
        for (d in nb - 2 downTo 0) {
            waveL[d] = resample(downRes[d], waveL[d + 1]!!, req[d + 1].first, nBand[d + 1], req[d])
            waveR[d] = resample(downRes[d], waveR[d + 1]!!, req[d + 1].first, nBand[d + 1], req[d])
        }

        for (d in 0 until nb) {
            val b = bands[d]
            val stft = stfts[d]
            val (ch0, ch1) = toModelChannels(waveL[d]!!, waveR[d]!!)
            waveL[d] = null; waveR[d] = null
            val tmpRe = FloatArray(stft.bins)
            val tmpIm = FloatArray(stft.bins)
            for ((c, x) in arrayOf(ch0, ch1).withIndex()) {
                for (t in 0 until frames) {
                    stft.analyze(x, req[d].first, nBand[d], fa + t, tmpRe, tmpIm, 0, 1)
                    for (r in b.cropStart until b.cropStop) {
                        val row = rowOffset[d] + r - b.cropStart
                        val g = preFilterGain[row]
                        val idx = (c * rows + row) * frames + t
                        spec.re[idx] = tmpRe[r] * g
                        spec.im[idx] = tmpIm[r] * g
                    }
                }
            }
        }
        return spec
    }

    private fun resample(
        res: PolyphaseResampler, x: FloatArray, xStart: Long, n: Long, range: Pair<Long, Long>
    ): FloatArray {
        val out = FloatArray((range.second - range.first).toInt().coerceAtLeast(0))
        res.process(x, xStart, n, out, range.first)
        return out
    }

    /** spec_utils.wave_to_spectrogram's channel conversion (non-v5.1 models). */
    private fun toModelChannels(l: FloatArray, r: FloatArray): Pair<FloatArray, FloatArray> = when {
        p.midSide -> FloatArray(l.size) { (l[it] + r[it]) / 2 } to FloatArray(l.size) { l[it] - r[it] }
        p.midSideB2 -> FloatArray(l.size) { r[it] + l[it] * .5f } to FloatArray(l.size) { l[it] - r[it] * .5f }
        else -> l to r
    }

    // -------------------------------------------------------------------------
    // Model
    // -------------------------------------------------------------------------

    /** Normalised |X| for frames [start, start + windowSize); zero outside [fa, fb). */
    private fun fillModelInput(spec: Spec, fa: Long, fb: Long, start: Long, norm: Float, input: FloatArray) {
        input.fill(0f)
        val t0 = maxOf(0L, fa - start).toInt()
        val t1 = minOf(windowSize.toLong(), fb - start).toInt()
        for (c in 0 until 2) for (r in 0 until rows) {
            val dst = (c * rows + r) * windowSize
            val src = (c * rows + r) * spec.frames + (start - fa).toInt()
            for (t in t0 until t1) {
                val re = spec.re[src + t]
                val im = spec.im[src + t]
                input[dst + t] = sqrt(re * re + im * im) * norm
            }
        }
    }

    /** spec_utils.adjust_aggr. */
    private fun adjustAggressiveness(mask: FloatArray, frames: Int) {
        var aggr = aggression * 2
        if (aggr == 0f) return
        if (meta.primaryStem in NON_ACCOM_STEMS) aggr = 1 - aggr
        val split = bands[0].cropStop
        for (c in 0 until 2) {
            val a = aggr + if (c == 0) p.aggrCorrectionLeft else p.aggrCorrectionRight
            for (r in 0 until rows) {
                val e = if (r < split) 1 + a / 3 else 1 + a
                val base = (c * rows + r) * frames
                for (t in 0 until frames) mask[base + t] = mask[base + t].pow(e)
            }
        }
    }

    // -------------------------------------------------------------------------
    // Synthesis: linear contribution of frames [c0, c1) to the output
    // -------------------------------------------------------------------------

    /** Stereo samples [start, start + left.size) of one stem's contribution. */
    private class Chunk(val start: Long, val left: FloatArray, val right: FloatArray) {
        val end get() = start + left.size
    }

    private fun synthesize(
        spec: Spec, fa: Long, mask: FloatArray, c0: Long, c1: Long, isPrimary: Boolean, stfts: List<Stft>
    ): Chunk? {
        val coreLen = (c1 - c0).toInt()
        if (coreLen <= 0) return null
        var acc: Chunk? = null
        for (d in 0 until nb) {
            if (acc != null) acc = resampleChunk(upRes[d - 1], acc, nSynth[d - 1], nSynth[d])
            acc = addChunks(acc, bandContribution(d, spec, fa, mask, c0, coreLen, isPrimary, stfts[d]))
        }
        return acc
    }

    private fun bandContribution(
        d: Int, spec: Spec, fa: Long, mask: FloatArray, c0: Long, coreLen: Int, isPrimary: Boolean, stft: Stft
    ): Chunk? {
        val b = bands[d]
        val gain = bandGain[d]
        val start = maxOf(0L, stft.frameStart(c0))
        val end = minOf(nSynth[d], stft.frameStart(c0 + coreLen - 1) + stft.nFft)
        if (end <= start) return null
        val y = arrayOf(FloatArray((end - start).toInt()), FloatArray((end - start).toInt()))
        val bRe = FloatArray(stft.bins)
        val bIm = FloatArray(stft.bins)
        for (c in 0 until 2) {
            for (t in 0 until coreLen) {
                val sf = (c0 - fa).toInt() + t
                for (r in b.cropStart until b.cropStop) {
                    val row = rowOffset[d] + r - b.cropStart
                    val m = mask[(c * rows + row) * coreLen + t]
                    val k = gain[r] * if (isPrimary) m else 1 - m
                    val idx = (c * rows + row) * spec.frames + sf
                    bRe[r] = spec.re[idx] * k
                    bIm[r] = spec.im[idx] * k
                }
                stft.overlapAdd(bRe, bIm, 0, 1, c0 + t, y[c], start)
            }
            for (i in y[c].indices) {
                val w = stft.windowSumSquare(start + i, nFrames)
                if (w > TINY) y[c][i] /= w
            }
        }
        // spectrogram_to_wave's channel conversion back to L/R
        val (l, r) = when {
            p.midSide -> FloatArray(y[0].size) { y[0][it] + y[1][it] / 2 } to
                FloatArray(y[0].size) { y[0][it] - y[1][it] / 2 }
            p.midSideB2 -> FloatArray(y[0].size) { y[1][it] / 1.25f + .4f * y[0][it] } to
                FloatArray(y[0].size) { y[0][it] / 1.25f - .4f * y[1][it] }
            else -> y[0] to y[1]
        }
        return Chunk(start, l, r)
    }

    private fun resampleChunk(res: PolyphaseResampler, x: Chunk, nIn: Long, nOut: Long): Chunk {
        val span = res.outputSpan(x.start, x.end)
        val start = maxOf(0L, span.first)
        val end = minOf(nOut, span.last + 1)
        val l = FloatArray((end - start).toInt().coerceAtLeast(0))
        val r = FloatArray(l.size)
        res.process(x.left, x.start, nIn, l, start)
        res.process(x.right, x.start, nIn, r, start)
        return Chunk(start, l, r)
    }

    private fun addChunks(a: Chunk?, b: Chunk?): Chunk? {
        if (a == null) return b
        if (b == null) return a
        val start = minOf(a.start, b.start)
        val end = maxOf(a.end, b.end)
        val l = FloatArray((end - start).toInt())
        val r = FloatArray(l.size)
        for (x in arrayOf(a, b)) {
            val o = (x.start - start).toInt()
            for (i in x.left.indices) { l[o + i] += x.left[i]; r[o + i] += x.right[i] }
        }
        return Chunk(start, l, r)
    }

    /** Earliest output sample that frames ≥ [frame] contribute to. */
    private fun contributionStart(frame: Long): Long {
        var earliest = Long.MAX_VALUE
        for (d in 0 until nb) {
            var s = maxOf(0L, stfts[d].frameStart(frame))
            for (u in d until nb - 1) s = maxOf(0L, upRes[u].outputSpan(s, s + 1).first)
            earliest = minOf(earliest, s)
        }
        return earliest
    }

    /** Rolling overlap-add buffer: samples below the flush point are final and written out. */
    private class OverlapAccumulator(private val total: Long, private val sink: StereoSink) {
        private var base = 0L
        private var left = FloatArray(0)
        private var right = FloatArray(0)

        fun add(c: Chunk) {
            val start = maxOf(c.start, 0L)
            val end = minOf(c.end, total)
            if (end <= start) return
            check(start >= base) { "Contribution at $start precedes flushed position $base" }
            ensure((end - base).toInt())
            val o = (start - base).toInt()
            val s = (start - c.start).toInt()
            for (i in 0 until (end - start).toInt()) {
                left[o + i] += c.left[s + i]
                right[o + i] += c.right[s + i]
            }
        }

        fun flush(until: Long) {
            val n = (minOf(until, total) - base).toInt()
            if (n <= 0) return
            ensure(n)
            sink.write(left, right, n)
            left.copyInto(left, 0, n, left.size); left.fill(0f, left.size - n)
            right.copyInto(right, 0, n, right.size); right.fill(0f, right.size - n)
            base += n
        }

        private fun ensure(size: Int) {
            if (size <= left.size) return
            left = left.copyOf(size)
            right = right.copyOf(size)
        }
    }

    companion object {
        /** gui_data/constants.py NON_ACCOM_STEMS — aggression is inverted for these. */
        val NON_ACCOM_STEMS = setOf(
            "Vocals", "Other", "Bass", "Drums", "Guitar", "Piano",
            "Synthesizer", "Strings", "Woodwinds", "Brass", "Wind Inst"
        )

        /** numpy.finfo(float32).tiny — librosa.istft's window-sum threshold. */
        private const val TINY = 1.1754944e-38f

        /** spec_utils.fft_lp_filter as a gain curve. */
        private fun lowPass(gain: FloatArray, start: Int, stop: Int) {
            var g = 1.0
            for (b in start until stop) {
                g -= 1.0 / (stop - start)
                gain[b] = (gain[b] * g).toFloat()
            }
            for (b in stop until gain.size) gain[b] = 0f
        }

        /** spec_utils.fft_hp_filter as a gain curve. */
        private fun highPass(gain: FloatArray, start: Int, stop: Int) {
            var g = 1.0
            for (b in start downTo stop + 1) {
                g -= 1.0 / (start - stop)
                gain[b] = (gain[b] * g).toFloat()
            }
            for (b in 0..stop) gain[b] = 0f
        }
    }
}
