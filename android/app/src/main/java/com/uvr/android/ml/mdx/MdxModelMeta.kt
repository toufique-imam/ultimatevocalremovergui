package com.uvr.android.ml.mdx

import android.content.Context
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.InputStream
import java.security.MessageDigest

/**
 * MDX-Net model settings, found the way desktop UVR finds them: the MD5 of the
 * last 10 MB of the .onnx is looked up in models/MDX_Net_Models/model_data/
 * model_data.json (shipped as assets/uvr/mdx_model_data.json). So original UVR
 * .onnx downloads work as-is, without conversion.
 */
data class MdxModelMeta(
    val nFft: Int,
    val dimF: Int,
    val dimT: Int,
    val compensate: Float,
    val primaryStem: String,
    val isKaraoke: Boolean
) {
    val secondaryStem: String get() = secondaryStemOf(primaryStem)

    companion object {
        private const val TABLE_ASSET = "uvr/mdx_model_data.json"
        private const val HASH_TAIL = 10_000L * 1024

        @Volatile private var table: JsonObject? = null

        private fun table(context: Context): JsonObject = table ?: synchronized(this) {
            table ?: context.assets.open(TABLE_ASSET).bufferedReader().use {
                JsonParser.parseReader(it).asJsonObject
            }.also { table = it }
        }

        /** Settings for the model with UVR hash [hash] ([modelHash]), or null if UVR doesn't list it. */
        fun forHash(context: Context, hash: String): MdxModelMeta? {
            val entry = table(context).getAsJsonObject(hash) ?: return null
            // MDX23C (.ckpt) entries carry a config_yaml instead of STFT settings
            if (!entry.has("mdx_dim_f_set")) return null
            return MdxModelMeta(
                nFft = entry["mdx_n_fft_scale_set"].asInt,
                dimF = entry["mdx_dim_f_set"].asInt,
                dimT = 1 shl entry["mdx_dim_t_set"].asInt,
                compensate = entry["compensate"].asFloat,
                primaryStem = entry["primary_stem"].asString,
                isKaraoke = entry["is_karaoke"]?.asBoolean ?: false
            )
        }

        /** UVR's model hash: MD5 of the last 10 MB (or the whole file if smaller). */
        fun modelHash(size: Long, open: () -> InputStream): String {
            val md = MessageDigest.getInstance("MD5")
            open().use { input ->
                var skip = maxOf(0L, size - HASH_TAIL)
                while (skip > 0) {
                    val n = input.skip(skip)
                    if (n <= 0) break
                    skip -= n
                }
                val buf = ByteArray(1 shl 16)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        /** gui_data/constants.py secondary_stem(). */
        fun secondaryStemOf(stem: String): String = when (stem) {
            "Vocals" -> "Instrumental"
            "Instrumental" -> "Vocals"
            "lead_only" -> "backing_only"
            "backing_only" -> "lead_only"
            "Primary Stem" -> "Secondary Stem"
            else -> if (stem.contains("No ")) stem.replace("No ", "") else "No $stem"
        }
    }
}
