package com.uvr.android.ml.demucs

import com.google.gson.JsonParser

/**
 * A Demucs v4 model as exported by convert.ipynb: a `<name>.demucs` JSON
 * manifest next to one ONNX network core per sub-model of the bag
 * (htdemucs: 1, htdemucs_ft: 4, each weighted per source).
 */
data class DemucsModelMeta(
    val name: String,
    /** Model source order, e.g. drums, bass, other, vocals. */
    val sources: List<String>,
    val sampleRate: Int,
    /** int(segment · samplerate): the training length every chunk is padded to. */
    val segmentSamples: Int,
    val nFft: Int,
    val hop: Int,
    val models: List<SubModel>
) {
    data class SubModel(val file: String, val weights: List<Float>)

    /**
     * Stem names in desktop UVR's output order: demix_demucs swaps the first two
     * sources, so DEMUCS_4/6_SOURCE_MAPPER reads Bass, Drums, Other, Vocals(, Guitar, Piano).
     */
    val uvrStems: List<String>
        get() = sources.toMutableList().apply { if (size >= 2) add(0, removeAt(1)) }
            .map { it.replaceFirstChar(Char::uppercase) }

    /** Index into [sources] of UVR output stem i. */
    fun uvrToModelIndex(i: Int): Int = when (i) { 0 -> 1; 1 -> 0; else -> i }

    companion object {
        const val EXTENSION = "demucs"

        fun parse(json: String): DemucsModelMeta {
            val o = JsonParser.parseString(json).asJsonObject
            require(o["arch"]?.asString == "demucs_v4") { "Not a Demucs v4 manifest" }
            return DemucsModelMeta(
                name = o["name"].asString,
                sources = o["sources"].asJsonArray.map { it.asString },
                sampleRate = o["samplerate"].asInt,
                segmentSamples = o["segment_samples"].asInt,
                nFft = o["nfft"].asInt,
                hop = o["hop_length"].asInt,
                models = o["models"].asJsonArray.map { m ->
                    val mo = m.asJsonObject
                    SubModel(mo["file"].asString, mo["weights"].asJsonArray.map { it.asFloat })
                }
            )
        }
    }
}
