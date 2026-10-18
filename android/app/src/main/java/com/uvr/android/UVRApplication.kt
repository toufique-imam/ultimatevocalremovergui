package com.uvr.android

import android.app.Application
import android.util.Log
import com.uvr.android.ml.ModelManager
import com.uvr.android.ml.VerifiedEngine
import com.uvr.android.utils.FileUtils

/**
 * Application class — initializes global singletons (ModelManager).
 */
class UVRApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "UVR Application starting")
        // Trigger lazy init of ModelManager so it can scan assets/models on startup
        ModelManager.initialize(this)
        // Don't re-check the GPU against the CPU model on every launch
        VerifiedEngine.persistIn(this)

        // No separation can be running in a fresh process, so any temp file from
        // before this start is an orphan of a killed or failed run.
        val processStart = System.currentTimeMillis()
        Thread {
            val freed = FileUtils.deleteStaleTempFiles(cacheDir, processStart)
            if (freed > 0) Log.i(TAG, "Deleted stale temp files: ${freed / 1_000_000} MB")
        }.start()
    }

    companion object {
        private const val TAG = "UVRApplication"
    }
}
