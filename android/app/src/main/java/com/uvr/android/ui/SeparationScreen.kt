package com.uvr.android.ui

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.media.MediaPlayer
import android.net.Uri
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.uvr.android.audio.OutputStem
import com.uvr.android.utils.FileUtils
import com.uvr.android.viewmodel.FileResult
import com.uvr.android.viewmodel.MainViewModel
import com.uvr.android.viewmodel.ProcessingStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Separation progress and result screen.
 * Mirrors UVR's progress bar area and output filename display.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SeparationScreen(
    viewModel: MainViewModel,
    onBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    val isProcessing = uiState.status is ProcessingStatus.Separating

    // Block system back button while processing
    BackHandler(enabled = isProcessing) { /* swallow — user must use Cancel button */ }

    DisposableEffect(Unit) {
        val window = (context as? Activity)?.window
        window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        when (uiState.status) {
                            is ProcessingStatus.Done -> "Result"
                            is ProcessingStatus.Error -> "Failed"
                            else -> "Separating"
                        }
                    )
                },
                navigationIcon = {
                    if (!isProcessing) {
                        IconButton(onClick = onBack) {
                            Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                        }
                    }
                }
            )
        }
    ) { padding ->
        // Scrolls when the content doesn't fit (landscape, large fonts); centred otherwise
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentAlignment = Alignment.Center
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                when (val status = uiState.status) {
                    is ProcessingStatus.Separating -> {
                        SeparatingView(
                            status = status,
                            fileName = status.fileName ?: uiState.selectedFileName,
                            onCancel = {
                                viewModel.cancelSeparation()
                                onBack()
                            }
                        )
                    }
                    is ProcessingStatus.Done -> {
                        DoneView(
                            files = status.files,
                            onSeparateAnother = {
                                viewModel.clearSelection()
                                onBack()
                            }
                        )
                    }
                    is ProcessingStatus.Error -> {
                        ErrorView(
                            message = status.message,
                            onRetry = onBack
                        )
                    }
                    else -> {
                        // Should not land here normally
                        Text("Preparing…")
                    }
                }
            }
        }
    }
}

@Composable
private fun SeparatingView(status: ProcessingStatus.Separating, fileName: String, onCancel: () -> Unit) {
    var showCancelDialog by remember { mutableStateOf(false) }
    val progress = status.progress
    val isFolder = status.fileCount > 1

    if (showCancelDialog) {
        AlertDialog(
            onDismissRequest = { showCancelDialog = false },
            title = { Text("Cancel Separation?") },
            text = {
                Text(
                    if (isFolder) "The separation will be stopped. Files already separated stay saved."
                    else "The separation will be stopped and progress will be lost."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showCancelDialog = false
                    onCancel()
                }) { Text("Cancel Separation") }
            },
            dismissButton = {
                TextButton(onClick = { showCancelDialog = false }) { Text("Keep Going") }
            }
        )
    }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        CircularProgressIndicator(
            progress = { progress },
            modifier = Modifier.size(100.dp),
            strokeWidth = 8.dp,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            "${(progress * 100).toInt()}%",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )
        if (isFolder) {
            Text(
                "File ${status.fileIndex + 1} of ${status.fileCount}: $fileName",
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
        Text(
            status.stage ?: if (isFolder) "Separating…" else "Separating: $fileName",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f)
        )
        Text(
            "This may take a few minutes depending on file length and model size.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f)
        )
        // In a folder, the bar shows the whole run and the circle the current file
        LinearProgressIndicator(
            progress = { if (isFolder) status.overall else progress },
            modifier = Modifier.fillMaxWidth()
        )
        OutlinedButton(
            onClick = { showCancelDialog = true },
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
            border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.error)
        ) {
            Icon(Icons.Default.Close, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Cancel")
        }
    }
}

@Composable
private fun DoneView(
    files: List<FileResult>,
    onSeparateAnother: () -> Unit
) {
    val context = LocalContext.current
    val player = remember { StemPlayer(context.applicationContext) }
    DisposableEffect(player) { onDispose { player.release() } }
    val isFolder = files.size > 1
    val failed = files.count { it.error != null }
    val stems = files.flatMap { it.stems }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Icon(
            if (failed > 0) Icons.Default.Warning else Icons.Default.CheckCircle,
            contentDescription = null,
            tint = if (failed > 0) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(80.dp)
        )
        Text(
            if (failed > 0) "Separated ${files.size - failed} of ${files.size} Files" else "Separation Complete!",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )

        // One card per input file for a folder (failed files show why)
        files.forEach { file ->
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(modifier = Modifier.padding(vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(if (isFolder) file.fileName else "Output Files",
                        style = if (isFolder) MaterialTheme.typography.titleSmall else MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = if (isFolder) 1f else 0.6f),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp))
                    file.error?.let {
                        Text(it, style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                            maxLines = 3,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.padding(horizontal = 16.dp))
                    }
                    // Order comes from the model (the SP-UVR VR models output Instrumental first)
                    file.stems.forEachIndexed { i, (name, stem) ->
                        if (i > 0) HorizontalDivider(color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f))
                        StemRow(label = name, stem = stem, player = player)
                    }
                }
            }
        }

        Text(
            "Saved to ${stems.map { it.second.location.substringBeforeLast('/') }.distinct().joinToString(" and ")}",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f)
        )

        Button(
            onClick = onSeparateAnother,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.Add, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text(if (isFolder) "Separate Another Folder" else "Separate Another File")
        }
    }
}

private fun stemIcon(stemName: String) =
    if (stemName.contains("vocal", ignoreCase = true)) Icons.Default.MicNone else Icons.Default.MusicNote

@Composable
private fun StemRow(label: String, stem: OutputStem, player: StemPlayer) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val saveLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument(stem.mimeType)
    ) { dest ->
        if (dest == null) return@rememberLauncherForActivityResult
        scope.launch {
            val saved = withContext(Dispatchers.IO) {
                runCatching {
                    val resolver = context.contentResolver
                    resolver.openInputStream(stem.uri)!!.use { input ->
                        resolver.openOutputStream(dest)!!.use { input.copyTo(it) }
                    }
                }.isSuccess
            }
            Toast.makeText(context, if (saved) "Saved" else "Couldn't save the file", Toast.LENGTH_SHORT).show()
        }
    }
    val isPlaying = player.playing == stem.uri
    val isVideo = stem.mimeType.startsWith("video/")

    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 16.dp, end = 4.dp)
    ) {
        Icon(if (isVideo) Icons.Default.Movie else stemIcon(label), contentDescription = null,
            tint = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(label, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
            Text(
                "${stem.displayName} • %.1f MB".format(stem.sizeBytes / 1_000_000.0),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        if (isVideo) {
            // Videos play in the device's video player rather than in-app
            IconButton(onClick = {
                player.stop()
                try {
                    context.startActivity(Intent(Intent.ACTION_VIEW)
                        .setDataAndType(stem.uri, stem.mimeType)
                        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
                } catch (e: ActivityNotFoundException) {
                    Toast.makeText(context, "No app can play this video", Toast.LENGTH_SHORT).show()
                }
            }) {
                Icon(Icons.Default.PlayArrow, contentDescription = "Play $label")
            }
        } else {
            IconButton(onClick = { player.toggle(stem.uri) }) {
                Icon(
                    if (isPlaying) Icons.Default.Stop else Icons.Default.PlayArrow,
                    contentDescription = if (isPlaying) "Stop $label" else "Play $label"
                )
            }
        }
        IconButton(onClick = {
            context.startActivity(
                Intent.createChooser(FileUtils.buildShareIntent(stem.uri, stem.mimeType), "Share $label")
            )
        }) {
            Icon(Icons.Default.Share, contentDescription = "Share $label")
        }
        IconButton(onClick = { saveLauncher.launch(stem.displayName) }) {
            Icon(Icons.Default.SaveAlt, contentDescription = "Save $label as…")
        }
    }
}

/** Plays one stem at a time; [playing] drives the play/stop icons. */
private class StemPlayer(private val context: Context) {
    var playing by mutableStateOf<Uri?>(null)
        private set
    private var player: MediaPlayer? = null

    fun toggle(uri: Uri) {
        val wasPlaying = playing == uri
        stop()
        if (wasPlaying) return
        try {
            player = MediaPlayer().apply {
                setDataSource(context, uri)
                setOnPreparedListener { it.start() }
                setOnCompletionListener { stop() }
                prepareAsync()
            }
            playing = uri
        } catch (e: Exception) {
            Log.w("StemPlayer", "Cannot play $uri", e)
            stop()
            Toast.makeText(context, "Can't play this file", Toast.LENGTH_SHORT).show()
        }
    }

    fun stop() {
        player?.release()
        player = null
        playing = null
    }

    fun release() = stop()
}

@Composable
private fun ErrorView(message: String, onRetry: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Icon(
            Icons.Default.Error,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.error,
            modifier = Modifier.size(80.dp)
        )
        Text("Separation Failed", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(8.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.errorContainer
            )
        ) {
            Text(
                message,
                modifier = Modifier.padding(12.dp),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onErrorContainer
            )
        }
        OutlinedButton(onClick = onRetry, modifier = Modifier.fillMaxWidth()) {
            Icon(Icons.Default.ArrowBack, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Go Back")
        }
    }
}
