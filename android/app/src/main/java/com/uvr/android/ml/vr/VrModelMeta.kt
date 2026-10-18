package com.uvr.android.ml.vr

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * One band of a VR model's multi-band spectrogram (a lib_v5/vr_network/modelparams JSON).
 * Filter fields are -1 when absent from the JSON.
 */
data class VrBand(
    val sr: Int,
    val hl: Int,
    val nFft: Int,
    val cropStart: Int,
    val cropStop: Int,
    val lpfStart: Int,
    val lpfStop: Int,
    val hpfStart: Int,
    val hpfStop: Int
)

/** Mirrors `ModelParameters.param` from lib_v5/vr_network/model_param_init.py. */
data class VrModelParams(
    val bins: Int,
    val sr: Int,
    val preFilterStart: Int,
    val preFilterStop: Int,
    val midSide: Boolean,
    val midSideB2: Boolean,
    val reverse: Boolean,
    val bands: List<VrBand>,          // band 1 (lowest) first
    val aggrCorrectionLeft: Float = 0f,
    val aggrCorrectionRight: Float = 0f
)

/**
 * Everything the Android pipeline needs to reproduce desktop UVR's VR
 * processing, embedded in the .ptl by convert.ipynb as `extra/uvr_meta.json`.
 */
data class VrModelMeta(
    val params: VrModelParams,
    val primaryStem: String,
    val secondaryStem: String,
    val offset: Int,
    val isVr51: Boolean,
    val paramName: String
) {
    companion object {
        const val META_ENTRY = "uvr_meta.json"

        /** Reads `<archive>/extra/uvr_meta.json` from a TorchScript zip, or null if absent. */
        fun readFromPtl(file: File): VrModelMeta? =
            RandomAccessFile(file, "r").use { readFromPtl(it.channel, 0, it.length()) }

        /**
         * Same as [readFromPtl] for a zip stored at [start, start + length) of [channel]
         * — e.g. an uncompressed APK asset via `AssetManager.openFd`.
         *
         * Reads the central directory directly: torch stores entries uncompressed with
         * zeroed sizes in the local headers (data-descriptor flag), which
         * java.util.zip.ZipInputStream rejects.
         */
        fun readFromPtl(channel: FileChannel, start: Long, length: Long): VrModelMeta? = runCatching {
            fun read(pos: Long, n: Int): ByteBuffer {
                val bb = ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN)
                while (bb.hasRemaining()) if (channel.read(bb, start + pos + bb.position()) < 0) break
                return bb.flip() as ByteBuffer
            }
            val tailLen = minOf(length, 22L + 0xFFFF).toInt()
            val tail = read(length - tailLen, tailLen)
            val eocd = (tailLen - 22 downTo 0).firstOrNull { tail.getInt(it) == 0x06054b50 }
                ?: return null
            val count = tail.getShort(eocd + 10).toInt() and 0xFFFF
            val cdSize = tail.getInt(eocd + 12)
            val cdOffset = tail.getInt(eocd + 16).toLong() and 0xFFFFFFFFL
            val cd = read(cdOffset, cdSize)
            var p = 0
            repeat(count) {
                if (cd.getInt(p) != 0x02014b50) return null
                val method = cd.getShort(p + 10).toInt()
                val size = cd.getInt(p + 20)
                val nameLen = cd.getShort(p + 28).toInt() and 0xFFFF
                val extraLen = cd.getShort(p + 30).toInt() and 0xFFFF
                val commentLen = cd.getShort(p + 32).toInt() and 0xFFFF
                val localOffset = cd.getInt(p + 42).toLong() and 0xFFFFFFFFL
                val name = String(cd.array(), p + 46, nameLen, Charsets.UTF_8)
                if (name.endsWith("extra/$META_ENTRY") && method == 0) {
                    val lh = read(localOffset, 30)
                    val dataStart = localOffset + 30 +
                        (lh.getShort(26).toInt() and 0xFFFF) + (lh.getShort(28).toInt() and 0xFFFF)
                    return parse(String(read(dataStart, size).array(), Charsets.UTF_8))
                }
                p += 46 + nameLen + extraLen + commentLen
            }
            null
        }.getOrNull()

        fun parse(json: String): VrModelMeta {
            val root = JsonParser.parseString(json).asJsonObject
            val p = root.getAsJsonObject("model_params")
            val bandsObj = p.getAsJsonObject("band")
            val bands = bandsObj.keySet().map { it.toInt() }.sorted().map { key ->
                val b = bandsObj.getAsJsonObject(key.toString())
                VrBand(
                    sr = b.int("sr"), hl = b.int("hl"), nFft = b.int("n_fft"),
                    cropStart = b.int("crop_start"), cropStop = b.int("crop_stop"),
                    lpfStart = b.int("lpf_start"), lpfStop = b.int("lpf_stop"),
                    hpfStart = b.int("hpf_start"), hpfStop = b.int("hpf_stop")
                )
            }
            val aggr = p.getAsJsonObject("aggr_correction")
            val params = VrModelParams(
                bins = if (p.has("n_bins")) p.int("n_bins") else p.int("bins"),
                sr = p.int("sr", 44100),
                preFilterStart = p.int("pre_filter_start"),
                preFilterStop = p.int("pre_filter_stop"),
                midSide = p.bool("mid_side"),
                midSideB2 = p.bool("mid_side_b2"),
                reverse = p.bool("reverse"),
                bands = bands,
                aggrCorrectionLeft = aggr?.get("left")?.asFloat ?: 0f,
                aggrCorrectionRight = aggr?.get("right")?.asFloat ?: 0f
            )
            return VrModelMeta(
                params = params,
                primaryStem = root.get("primary_stem").asString,
                secondaryStem = root.get("secondary_stem").asString,
                offset = root.int("offset", 128),
                isVr51 = root.bool("is_vr51"),
                paramName = root.get("model_param_name")?.asString ?: ""
            )
        }

        private fun JsonObject.int(key: String, default: Int = -1): Int =
            get(key)?.takeIf { !it.isJsonNull }?.asInt ?: default

        private fun JsonObject.bool(key: String): Boolean =
            get(key)?.takeIf { !it.isJsonNull }?.asBoolean ?: false
    }
}
