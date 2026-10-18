package com.uvr.android.audio

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.uvr.android.MainActivity
import com.uvr.android.R
import com.uvr.android.ml.ModelInfo
import com.uvr.android.ml.Separator
import com.uvr.android.ml.SeparationParams
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * Foreground service that runs audio separation in the background.
 *
 * Keeps a persistent notification so Android doesn't kill the process
 * mid-separation (which can take 30s–5min for large files).
 *
 * Mirrors UVR desktop's use of threading (KThread) for background processing.
 */
class SeparationService : Service() {

    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(Dispatchers.Default + Job())

    private val _state = MutableStateFlow<SeparationState>(SeparationState.Idle)
    val state: StateFlow<SeparationState> = _state

    inner class LocalBinder : Binder() {
        fun getService(): SeparationService = this@SeparationService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    /**
     * Keep-alive mode, driven by [keepAlive] / [release]: while a separation runs
     * in the app process, hold a foreground notification and a partial wake lock.
     * Without them the OS stops the work once the screen turns off — Samsung's
     * Freecess freezes the process, others throttle or kill it — and a long song
     * takes several minutes.
     */
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_KEEP_ALIVE -> {
                val notification = buildNotification(this, "Separating…", 0)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
                } else {
                    startForeground(NOTIFICATION_ID, notification)
                }
                if (wakeLock == null) {
                    wakeLock = getSystemService(PowerManager::class.java)
                        .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "uvr:separation")
                        .apply { acquire(WAKE_LOCK_TIMEOUT_MS) }
                }
            }
            ACTION_RELEASE -> {
                releaseWakeLock()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        releaseWakeLock()
        super.onDestroy()
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    /**
     * Start a separation job.
     *
     * @param inputPcm   Decoded PCM from AudioProcessor
     * @param model      Selected ONNX model
     * @param params     Separation parameters
     * @param outputDir  Directory to save stems
     * @param baseName   Base filename for output files
     */
    fun startSeparation(
        inputPcm: DecodedAudio,
        model: ModelInfo,
        params: SeparationParams,
        outputDir: File,
        baseName: String
    ) {
        startForeground(NOTIFICATION_ID, buildNotification(this, "Starting…", 0))
        _state.value = SeparationState.Running(0f)

        serviceScope.launch {
            try {
                val separator = Separator.create(model, params)
                val result = separator.separate(inputPcm, cacheDir) { progress ->
                    _state.value = SeparationState.Running(progress)
                    updateNotification("Separating…", (progress * 100).toInt())
                }

                // Save stems
                AudioProcessor(applicationContext)
                val primaryFile = result.stems[0].file
                val secondaryFile = result.stems[1].file

//                audioProcessor.encode(result., primaryFile)
//                audioProcessor.encode(result.secondaryStem, secondaryFile)

                Log.i(TAG, "Saved: ${primaryFile.name}, ${secondaryFile.name}")
                _state.value = SeparationState.Done(primaryFile, secondaryFile)
                updateNotification("Done!", 100)

            } catch (e: Exception) {
                Log.e(TAG, "Separation failed", e)
                _state.value = SeparationState.Error(e.message ?: "Unknown error")
                updateNotification("Failed: ${e.message}", 0)
            } finally {
                stopForeground(STOP_FOREGROUND_DETACH)
            }
        }
    }

    fun cancelSeparation() {
        serviceScope.launch {
            _state.value = SeparationState.Idle
            stopForeground(STOP_FOREGROUND_REMOVE)
        }
    }

    // -------------------------------------------------------------------------
    // Notifications
    // -------------------------------------------------------------------------

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Audio Separation",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shows progress of vocal separation"
        }
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(channel)
    }

    private fun updateNotification(status: String, progress: Int) = updateProgress(this, status, progress)

    companion object {
        private const val TAG = "SeparationService"
        private const val CHANNEL_ID = "uvr_separation"
        private const val NOTIFICATION_ID = 1001
        private const val ACTION_KEEP_ALIVE = "com.uvr.android.action.KEEP_ALIVE"
        private const val ACTION_RELEASE = "com.uvr.android.action.RELEASE"
        private const val WAKE_LOCK_TIMEOUT_MS = 6 * 60 * 60 * 1000L

        /** Start keep-alive mode. Call while the app is in the foreground (e.g. on the button tap). */
        fun keepAlive(context: Context) = ContextCompat.startForegroundService(
            context, Intent(context, SeparationService::class.java).setAction(ACTION_KEEP_ALIVE))

        fun release(context: Context) {
            context.startService(Intent(context, SeparationService::class.java).setAction(ACTION_RELEASE))
        }

        /** Update the ongoing notification's progress (0–100). */
        fun updateProgress(context: Context, status: String, progress: Int) {
            context.getSystemService(NotificationManager::class.java)
                .notify(NOTIFICATION_ID, buildNotification(context, status, progress))
        }

        private fun buildNotification(context: Context, status: String, progress: Int): Notification {
            val intent = Intent(context, MainActivity::class.java)
            val pi = PendingIntent.getActivity(
                context, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            return NotificationCompat.Builder(context, CHANNEL_ID)
                .setContentTitle("UVR — Vocal Remover")
                .setContentText(status)
                .setSmallIcon(android.R.drawable.ic_media_play)
                .setContentIntent(pi)
                .setProgress(100, progress, progress == 0)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .build()
        }
    }
}

sealed class SeparationState {
    object Idle : SeparationState()
    data class Running(val progress: Float) : SeparationState()
    data class Done(val primaryFile: File, val secondaryFile: File) : SeparationState()
    data class Error(val message: String) : SeparationState()
}
