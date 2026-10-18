package com.uvr.android.utils

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat

/**
 * Utilities for requesting Android runtime permissions needed by UVR.
 * On Android 13+ (API 33), READ_MEDIA_AUDIO replaces READ_EXTERNAL_STORAGE.
 */
object PermissionUtils {

    /** Returns the correct audio read permission for the current API level */
    fun audioReadPermission(): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_AUDIO
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }
    }

    fun hasAudioReadPermission(context: Context): Boolean {
        return ContextCompat.checkSelfPermission(context, audioReadPermission()) ==
                PackageManager.PERMISSION_GRANTED
    }

    fun hasStorageWritePermission(context: Context): Boolean {
        // Android 10+ handles storage via SAF/scoped storage — no runtime permission needed
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            true
        } else {
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.WRITE_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
        }
    }
}
