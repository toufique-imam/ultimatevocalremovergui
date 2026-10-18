package com.uvr.android.ui

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.uvr.android.audio.OutputFormat
import com.uvr.android.ml.ModelInfo
import com.uvr.android.viewmodel.MainViewModel
import com.uvr.android.viewmodel.ProcessingStatus
import com.uvr.android.viewmodel.videoStemOptions
import java.io.File

/**
 * Main home screen — mirrors UVR desktop's root window.
 *
 * Layout:
 *  - Top app bar with settings/models navigation
 *  - File selection area (drag-and-drop equivalent: tap to open picker)
 *  - Waveform preview
 *  - Model selector dropdown
 *  - Output format selector
 *  - Start button
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    viewModel: MainViewModel,
    onNavigateToSettings: () -> Unit,
    onNavigateToModels: () -> Unit,
    onStartSeparation: () -> Unit
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()
    val models by viewModel.models.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val waveform by viewModel.waveform.collectAsState()
    val outputFormat by viewModel.outputFormat.collectAsState()
    val params by viewModel.params.collectAsState()
    val videoStem by viewModel.videoStem.collectAsState()
    val trimSilence by viewModel.trimSilence.collectAsState()

    // Audio/video file picker — the system picker grants access to the chosen file,
    // so no storage/media permission is needed.
    val audioPickerLauncher = rememberLauncherForActivityResult(
        GetAudioOrVideo()
    ) { uri -> uri?.let { viewModel.selectAudioFile(it) } }

    // Folder picker: grants read access to the folder and everything in it
    val folderPickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri -> uri?.let { viewModel.selectFolder(it) } }

    // Output dir for Android 9 and below — newer versions publish to Music/UVR
    val outputDir = remember {
        context.getExternalFilesDir("Separated") ?: context.filesDir
    }

    val start = {
        if (viewModel.startSeparation(outputDir)) onStartSeparation()
    }
    // Android 13+: ask once for notifications so the progress notification is
    // visible; separation starts whatever the answer.
    val notificationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { start() }
    val onStartClick = {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            start()
        }
    }
    val status = uiState.status

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Ultimate Vocal Remover",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                },
                actions = {
                    IconButton(onClick = onNavigateToModels) {
                        Icon(Icons.Default.FolderOpen, contentDescription = "Models")
                    }
                    IconButton(onClick = onNavigateToSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { paddingValues ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item { Spacer(modifier = Modifier.height(8.dp)) }

            // ---- File Selection Card ----
            item {
                FileSelectionCard(
                    fileName = uiState.selectedFileName,
                    durationSeconds = uiState.durationSeconds,
                    isLoading = status == ProcessingStatus.Loading,
                    isVideo = uiState.isVideo,
                    isFolder = uiState.isFolder,
                    folderFileCount = uiState.folderFileCount,
                    onPickFile = { audioPickerLauncher.launch("*/*") },
                    onPickFolder = { folderPickerLauncher.launch(null) },
                    onClear = viewModel::clearSelection
                )
            }

            // ---- Load error (separation errors are shown on the separation screen) ----
            if (status is ProcessingStatus.Error && !uiState.isAudioReady) {
                item {
                    ErrorCard(message = status.message, onDismiss = viewModel::dismissError)
                }
            }

            // ---- Waveform Preview ----
            if (waveform != null) {
                item {
                    WaveformView(
                        waveform = waveform!!,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(80.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(MaterialTheme.colorScheme.surfaceVariant)
                    )
                }
            }

            // ---- Model Selector ----
            item {
                ModelSelectorCard(
                    models = models,
                    selectedModel = selectedModel,
                    onModelSelected = viewModel::selectModel,
                    onManageModels = onNavigateToModels
                )
            }

            // ---- Output Format ----
            item {
                OutputFormatCard(
                    selectedFormat = outputFormat,
                    onFormatSelected = viewModel::setOutputFormat,
                    trimSilence = trimSilence,
                    onTrimSilenceChange = viewModel::setTrimSilence
                )
            }

            // ---- Video output (video inputs only) ----
            if (uiState.isVideo) {
                item {
                    val options = videoStemOptions(selectedModel, params)
                    VideoOutputCard(
                        options = options,
                        selected = videoStem?.takeIf { it in options } ?: options.first(),
                        trimSilence = trimSilence,
                        onSelected = viewModel::setVideoStem
                    )
                }
            }

            // ---- Process Button ----
            item {
                Button(
                    onClick = onStartClick,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    enabled = uiState.isAudioReady
                            && selectedModel?.isSupported == true
                            && status != ProcessingStatus.Loading
                            && status !is ProcessingStatus.Separating,
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary
                    )
                ) {
                    Icon(Icons.Default.PlayArrow, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Start Separation", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                }
            }

            item { Spacer(modifier = Modifier.height(24.dp)) }
        }
    }
}

/** The system file picker, offering audio and video files. */
private class GetAudioOrVideo : ActivityResultContracts.GetContent() {
    override fun createIntent(context: Context, input: String): Intent =
        super.createIntent(context, "*/*").putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/*", "video/*"))
}

@Composable
private fun FileSelectionCard(
    fileName: String,
    durationSeconds: Float,
    isLoading: Boolean,
    isVideo: Boolean,
    isFolder: Boolean,
    folderFileCount: Int,
    onPickFile: () -> Unit,
    onPickFolder: () -> Unit,
    onClear: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { if (isFolder) onPickFolder() else onPickFile() }
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(
                when {
                    isFolder -> Icons.Default.Folder
                    isVideo -> Icons.Default.VideoFile
                    else -> Icons.Default.AudioFile
                },
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(40.dp)
            )
            Column(modifier = Modifier.weight(1f)) {
                if (fileName.isEmpty()) {
                    Text(
                        "Tap to select an audio or video file",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    )
                    Text(
                        "Supports MP3, WAV, FLAC, M4A, MP4, MKV, and more",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
                    )
                } else {
                    Text(
                        fileName,
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    if (isLoading) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp)
                            Text(
                                if (isFolder) "Looking for audio and video files…" else "Decoding…",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                            )
                        }
                    } else if (isFolder && folderFileCount > 0) {
                        Text(
                            if (folderFileCount == 1) "1 file" else "$folderFileCount files, separated one after another",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                        )
                    } else if (durationSeconds > 0f) {
                        val mins = (durationSeconds / 60).toInt()
                        val secs = (durationSeconds % 60).toInt()
                        Text(
                            "Duration: %d:%02d".format(mins, secs),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                        )
                    }
                }
            }
            if (fileName.isNotEmpty()) {
                IconButton(onClick = onClear) {
                    Icon(Icons.Default.Close, contentDescription = "Clear")
                }
            }
        }
        HorizontalDivider(color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f))
        // Desktop UVR takes a dropped folder too: every audio/video file in it, subfolders included
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            TextButton(onClick = onPickFile) {
                Icon(Icons.Default.AudioFile, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Choose File")
            }
            TextButton(onClick = onPickFolder) {
                Icon(Icons.Default.Folder, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Choose Folder")
            }
        }
    }
}

@Composable
private fun ErrorCard(message: String, onDismiss: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)
    ) {
        Row(
            modifier = Modifier.padding(start = 16.dp, top = 8.dp, bottom = 8.dp, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(Icons.Default.Error, contentDescription = null,
                tint = MaterialTheme.colorScheme.onErrorContainer)
            Text(
                message,
                modifier = Modifier.weight(1f),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onErrorContainer
            )
            IconButton(onClick = onDismiss) {
                Icon(Icons.Default.Close, contentDescription = "Dismiss",
                    tint = MaterialTheme.colorScheme.onErrorContainer)
            }
        }
    }
}

@Composable
private fun WaveformView(
    waveform: FloatArray,
    modifier: Modifier = Modifier
) {
    val primaryColor = MaterialTheme.colorScheme.primary
    Canvas(modifier = modifier) {
        val width = size.width
        val height = size.height
        val barWidth = width / waveform.size
        val midY = height / 2f

        waveform.forEachIndexed { i, amplitude ->
            val barHeight = (amplitude * height * 0.9f).coerceIn(1f, height)
            val x = i * barWidth + barWidth / 2f
            drawLine(
                color = primaryColor.copy(alpha = 0.8f),
                start = Offset(x, midY - barHeight / 2f),
                end = Offset(x, midY + barHeight / 2f),
                strokeWidth = (barWidth * 0.6f).coerceAtLeast(1f)
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ModelSelectorCard(
    models: List<ModelInfo>,
    selectedModel: ModelInfo?,
    onModelSelected: (ModelInfo) -> Unit,
    onManageModels: () -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Model", style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.height(8.dp))

            if (models.isEmpty()) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(Icons.Default.Warning, contentDescription = null,
                        tint = MaterialTheme.colorScheme.error)
                    Text("No models found. ", style = MaterialTheme.typography.bodyMedium)
                    TextButton(onClick = onManageModels) { Text("Add Models") }
                }
            } else {
                ExposedDropdownMenuBox(
                    expanded = expanded,
                    onExpandedChange = { expanded = it }
                ) {
                    OutlinedTextField(
                        value = selectedModel?.displayName ?: "Select a model",
                        onValueChange = {},
                        readOnly = true,
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .menuAnchor(),
                        supportingText = selectedModel?.let {
                            { Text("${it.architectureType.label} • ${it.format.label} • ${formatSize(it.fileSizeBytes)}") }
                        }
                    )
                    ExposedDropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false }
                    ) {
                        models.forEach { model ->
                            DropdownMenuItem(
                                text = {
                                    Column {
                                        Text(model.displayName)
                                        Text(
                                            model.unsupportedReason
                                                ?: "${model.architectureType.label} • ${model.format.label}",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurface.copy(0.7f)
                                        )
                                    }
                                },
                                enabled = model.isSupported,
                                onClick = {
                                    onModelSelected(model)
                                    expanded = false
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun OutputFormatCard(
    selectedFormat: OutputFormat,
    onFormatSelected: (OutputFormat) -> Unit,
    trimSilence: Boolean,
    onTrimSilenceChange: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Output Format", style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.height(8.dp))
            // Wraps onto a second line on narrow screens / large font sizes
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutputFormat.values().forEach { format ->
                    FilterChip(
                        selected = format == selectedFormat,
                        onClick = { onFormatSelected(format) },
                        label = { Text(format.extension.uppercase(), maxLines = 1, softWrap = false) }
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onTrimSilenceChange(!trimSilence) },
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Trim Silence", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        "Drop silent stretches (half a second or longer) from the stems",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    )
                }
                Switch(checked = trimSilence, onCheckedChange = onTrimSilenceChange)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun VideoOutputCard(
    options: List<String>,
    selected: String,
    trimSilence: Boolean,
    onSelected: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Video", style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
            Spacer(Modifier.height(8.dp))
            ExposedDropdownMenuBox(
                expanded = expanded,
                onExpandedChange = { expanded = it }
            ) {
                OutlinedTextField(
                    value = selected,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Stem for the video's audio") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor(),
                    supportingText = {
                        Text(
                            if (trimSilence) "Saved to Movies/UVR. Trim Silence is on: the video keeps its length, " +
                                "so its sound runs ahead of the picture after the first cut."
                            else "Saved to Movies/UVR, next to the separated stems in Music/UVR"
                        )
                    }
                )
                ExposedDropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false }
                ) {
                    options.forEach { option ->
                        DropdownMenuItem(
                            text = { Text(option) },
                            onClick = {
                                onSelected(option)
                                expanded = false
                            }
                        )
                    }
                }
            }
        }
    }
}
