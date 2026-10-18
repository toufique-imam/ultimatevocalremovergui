package com.uvr.android.ml

import kotlin.math.*

/**
 * Android port of UVR's lib_v5/spec_utils.py core operations.
 *
 * Key operations translated from Python/NumPy to Kotlin.
 * Optimized for memory efficiency on Android by providing in-place operations.
 */
object SpectrogramUtils {

    const val DEFAULT_SR = 44100
    const val DEFAULT_N_FFT = 2048
    const val DEFAULT_HOP_LENGTH = 1024
    const val DEFAULT_WINDOW_SIZE = 512

    // -------------------------------------------------------------------------
    // Normalization
    // -------------------------------------------------------------------------

    /** Normalize audio to [-1, 1] range (creates a new array) */
    fun normalize(audio: FloatArray): FloatArray {
        val maxVal = audio.maxOfOrNull { abs(it) } ?: 0f
        if (maxVal == 0f) return audio.copyOf()
        return FloatArray(audio.size) { audio[it] / maxVal }
    }

    /** Normalize audio to [-1, 1] range in-place. Returns the peak magnitude. */
    fun normalizeInPlace(audio: FloatArray): Float {
        var maxVal = 0f
        for (i in audio.indices) {
            val a = abs(audio[i])
            if (a > maxVal) maxVal = a
        }
        if (maxVal > 0f) {
            for (i in audio.indices) {
                audio[i] /= maxVal
            }
        }
        return maxVal
    }

    /** Denormalize by scaling back to original magnitude (creates a new array) */
    fun denormalize(audio: FloatArray, originalMax: Float): FloatArray {
        return FloatArray(audio.size) { audio[it] * originalMax }
    }

    /** Denormalize by scaling back to original magnitude in-place */
    fun denormalizeInPlace(audio: FloatArray, originalMax: Float) {
        if (originalMax == 1f || originalMax == 0f) return
        for (i in audio.indices) {
            audio[i] *= originalMax
        }
    }

    /** Get peak magnitude of audio array */
    fun peakMagnitude(audio: FloatArray): Float = audio.maxOfOrNull { abs(it) } ?: 0f

    // -------------------------------------------------------------------------
    // Spectrogram conversion helpers
    // -------------------------------------------------------------------------

    fun toMagnitudeAndPhase(complexData: FloatArray): Pair<FloatArray, FloatArray> {
        require(complexData.size % 2 == 0) { "Complex data must have even length" }
        val bins = complexData.size / 2
        val magnitude = FloatArray(bins)
        val phase = FloatArray(bins)
        for (i in 0 until bins) {
            val re = complexData[2 * i]
            val im = complexData[2 * i + 1]
            magnitude[i] = sqrt(re * re + im * im)
            phase[i] = atan2(im, re)
        }
        return magnitude to phase
    }

    fun fromMagnitudeAndPhase(magnitude: FloatArray, phase: FloatArray): FloatArray {
        require(magnitude.size == phase.size) { "Magnitude and phase size mismatch" }
        val complexData = FloatArray(magnitude.size * 2)
        for (i in magnitude.indices) {
            complexData[2 * i] = magnitude[i] * cos(phase[i])
            complexData[2 * i + 1] = magnitude[i] * sin(phase[i])
        }
        return complexData
    }

    // -------------------------------------------------------------------------
    // Streaming overlap-add chunking
    // -------------------------------------------------------------------------

    fun chunkCount(audioLen: Int, chunkLen: Int, hopLen: Int): Int {
        if (audioLen == 0) return 0
        var count = 0
        var pos = 0
        while (pos < audioLen) {
            count++
            pos += hopLen
        }
        return count
    }

    fun fillChunkBuffer(audio: FloatArray, pos: Int, buffer: FloatArray): Int {
        buffer.fill(0f)
        val end = minOf(pos + buffer.size, audio.size)
        val copied = end - pos
        if (copied > 0) audio.copyInto(buffer, 0, pos, end)
        return copied
    }

    fun accumulateChunk(output: FloatArray, counts: FloatArray, chunk: FloatArray, pos: Int) {
        val end = minOf(pos + chunk.size, output.size)
        for (i in pos until end) {
            output[i] += chunk[i - pos]
            counts[i] += 1f
        }
    }

    fun finalizeOverlapAdd(output: FloatArray, counts: FloatArray) {
        for (i in output.indices) {
            if (counts[i] > 0f) output[i] /= counts[i]
        }
    }

    // -------------------------------------------------------------------------
    // Stereo helpers
    // -------------------------------------------------------------------------

    /** Deinterleave stereo [L0,R0,L1,R1,...] -> Pair(left[], right[]) */
    fun deinterleave(stereo: FloatArray): Pair<FloatArray, FloatArray> {
        val n = stereo.size / 2
        val left = FloatArray(n)
        val right = FloatArray(n)
        for (i in 0 until n) {
            left[i] = stereo[2 * i]
            right[i] = stereo[2 * i + 1]
        }
        return left to right
    }

    /** Interleave left/right -> [L0,R0,L1,R1,...] */
    fun interleave(left: FloatArray, right: FloatArray): FloatArray {
        val out = FloatArray(left.size * 2)
        for (i in left.indices) {
            out[2 * i] = left[i]
            out[2 * i + 1] = right[i]
        }
        return out
    }

    /** Convert stereo to mono by averaging channels */
    fun toMono(left: FloatArray, right: FloatArray): FloatArray {
        return FloatArray(left.size) { i -> (left[i] + right[i]) * 0.5f }
    }

    /** Apply a soft Wiener filter mask. */
    fun applyWienerMask(
        primarySpec: FloatArray,
        mixSpec: FloatArray,
        epsilon: Float = 1e-10f
    ): FloatArray {
        return FloatArray(primarySpec.size) { i ->
            val mask = primarySpec[i] / (primarySpec[i] + (mixSpec[i] - primarySpec[i]).coerceAtLeast(0f) + epsilon)
            mask * mixSpec[i]
        }
    }

    /** Clamp spectrogram values to [0, mix] range. */
    fun clipToMix(spec: FloatArray, mixSpec: FloatArray): FloatArray {
        return FloatArray(spec.size) { i ->
            spec[i].coerceIn(0f, mixSpec[i])
        }
    }
}
