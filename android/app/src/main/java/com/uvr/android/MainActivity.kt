package com.uvr.android

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import com.uvr.android.ui.UVRNavHost
import com.uvr.android.ui.theme.UVRTheme
import kotlinx.coroutines.flow.MutableStateFlow

/**
 * Single-activity entry point. Navigation is handled by Compose NavHost.
 * Handles external audio file VIEW intents (e.g. open from file manager).
 */
class MainActivity : ComponentActivity() {

    // URI received from an external VIEW intent (e.g. share audio file to UVR),
    // held until the UI has loaded it.
    private val incomingAudioUri = MutableStateFlow<Uri?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Only on a fresh start: after a rotation the activity is recreated with the
        // same intent, and reading it again would reload the file mid-separation.
        if (savedInstanceState == null) handleIntent(intent)

        setContent {
            UVRTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val audioUri by incomingAudioUri.collectAsState()
                    UVRNavHost(
                        incomingAudioUri = audioUri,
                        onIncomingAudioHandled = { incomingAudioUri.value = null }
                    )
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        // "Open with UVR" while the app is already running
        setIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        if (intent?.action == Intent.ACTION_VIEW) intent.data?.let { incomingAudioUri.value = it }
    }
}
