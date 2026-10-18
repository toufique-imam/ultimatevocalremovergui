package com.uvr.android.utils

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.first

private val Context.settingsDataStore by preferencesDataStore(name = "settings")

/** User choices that survive app restarts. */
data class SavedSettings(
    val modelKey: String? = null,
    val outputFormat: String? = null,
    val vrWindowSize: Int? = null,
    val vrAggression: Float? = null,
    val vrTta: Boolean? = null,
    val vrUseGpu: Boolean? = null,
    /** MDX "Overlap"; −1 = "Default". */
    val mdxOverlap: Float? = null,
    val mdxDenoise: Boolean? = null,
    /** MDX "Fast" (GPU, FP16) instead of "Exact" (CPU). */
    val mdxFastGpu: Boolean? = null,
    val demucsOverlap: Float? = null,
    val demucsShifts: Int? = null,
    /** Desktop UVR's "Trim Silence": drop silent stretches from the stems. */
    val trimSilence: Boolean? = null
)

/** Persists [SavedSettings] with Jetpack DataStore (replaces Tkinter's saved settings). */
class SettingsStore(context: Context) {

    private val store = context.applicationContext.settingsDataStore

    suspend fun load(): SavedSettings {
        val p = store.data.first()
        return SavedSettings(
            modelKey = p[MODEL_KEY],
            outputFormat = p[OUTPUT_FORMAT],
            vrWindowSize = p[VR_WINDOW_SIZE],
            vrAggression = p[VR_AGGRESSION],
            vrTta = p[VR_TTA],
            vrUseGpu = p[VR_USE_GPU],
            mdxOverlap = p[MDX_OVERLAP],
            mdxDenoise = p[MDX_DENOISE],
            mdxFastGpu = p[MDX_FAST_GPU],
            demucsOverlap = p[DEMUCS_OVERLAP],
            demucsShifts = p[DEMUCS_SHIFTS],
            trimSilence = p[TRIM_SILENCE]
        )
    }

    suspend fun saveModel(key: String) = store.edit { it[MODEL_KEY] = key }

    suspend fun saveOutputFormat(name: String) = store.edit { it[OUTPUT_FORMAT] = name }

    suspend fun saveTrimSilence(on: Boolean) = store.edit { it[TRIM_SILENCE] = on }

    suspend fun saveVrParams(windowSize: Int, aggression: Float, tta: Boolean, useGpu: Boolean) = store.edit {
        it[VR_WINDOW_SIZE] = windowSize
        it[VR_AGGRESSION] = aggression
        it[VR_TTA] = tta
        it[VR_USE_GPU] = useGpu
    }

    suspend fun saveMdxParams(overlap: Float?, denoise: Boolean, fastGpu: Boolean) = store.edit {
        it[MDX_OVERLAP] = overlap ?: -1f
        it[MDX_DENOISE] = denoise
        it[MDX_FAST_GPU] = fastGpu
    }

    suspend fun saveDemucsParams(overlap: Float, shifts: Int) = store.edit {
        it[DEMUCS_OVERLAP] = overlap
        it[DEMUCS_SHIFTS] = shifts
    }

    private companion object {
        val MODEL_KEY = stringPreferencesKey("model")
        val OUTPUT_FORMAT = stringPreferencesKey("output_format")
        val VR_WINDOW_SIZE = intPreferencesKey("vr_window_size")
        val VR_AGGRESSION = floatPreferencesKey("vr_aggression")
        val VR_TTA = booleanPreferencesKey("vr_tta")
        val VR_USE_GPU = booleanPreferencesKey("vr_use_gpu")
        val MDX_OVERLAP = floatPreferencesKey("mdx_overlap")
        val MDX_DENOISE = booleanPreferencesKey("mdx_denoise")
        val MDX_FAST_GPU = booleanPreferencesKey("mdx_fast_gpu")
        val DEMUCS_OVERLAP = floatPreferencesKey("demucs_overlap")
        val DEMUCS_SHIFTS = intPreferencesKey("demucs_shifts")
        val TRIM_SILENCE = booleanPreferencesKey("trim_silence")
    }
}
