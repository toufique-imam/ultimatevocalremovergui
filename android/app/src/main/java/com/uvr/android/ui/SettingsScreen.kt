package com.uvr.android.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.uvr.android.ml.ArchitectureType
import com.uvr.android.viewmodel.MainViewModel
import kotlin.math.roundToInt

/**
 * Settings screen — per-model processing parameters.
 * Mirrors UVR's right-panel settings (overlap, segment size, window, aggression).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    viewModel: MainViewModel,
    onBack: () -> Unit
) {
    val params by viewModel.params.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Separation Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            val arch = selectedModel?.architectureType

            // MDX-Net: desktop UVR's "Overlap" and "Denoise Output"
            if (arch == ArchitectureType.MDX_NET) {
                item {
                    val index = MDX_OVERLAPS.indexOf(params.mdxOverlap).coerceAtLeast(0)
                    SliderSettingCard(
                        title = "Overlap",
                        description = "Overlap between model chunks. Default = n_fft samples (fastest); " +
                            "higher is smoother but slower.",
                        value = index.toFloat(),
                        valueRange = 0f..(MDX_OVERLAPS.size - 1).toFloat(),
                        steps = MDX_OVERLAPS.size - 2,
                        displayValue = MDX_OVERLAPS[index]?.let { "%.2f".format(it) } ?: "Default",
                        onValueChange = {
                            viewModel.updateParams { p -> p.copy(mdxOverlap = MDX_OVERLAPS[it.roundToInt()]) }
                        }
                    )
                }
                item {
                    SwitchSettingCard(
                        title = "Denoise Output",
                        description = "Also runs the model on the inverted spectrogram and averages the two, " +
                            "cancelling model noise. More accurate, twice the model time.",
                        checked = params.mdxDenoise,
                        onCheckedChange = { viewModel.updateParams { p -> p.copy(mdxDenoise = it) } }
                    )
                }
                item {
                    ChoiceSettingCard(
                        title = "Processing",
                        options = listOf("Exact", "Fast (GPU)"),
                        selected = if (params.mdxFastGpu) 1 else 0,
                        description = if (params.mdxFastGpu)
                            "Runs the model on the GPU in half precision: about 2.4× faster on phones whose GPU " +
                                "beats the CPU (checked once), otherwise the CPU. Very close to desktop UVR but not " +
                                "identical; a careful listener may hear it on loud vocals. Needs the model's .tflite file."
                        else
                            "Matches desktop UVR. Runs on the CPU.",
                        onSelect = { i -> viewModel.updateParams { p -> p.copy(mdxFastGpu = i == 1) } }
                    )
                }
            }

            // Demucs: desktop UVR's "Stems", "Overlap" and "Shifts"
            if (arch == ArchitectureType.DEMUCS) {
                val stems = listOf<String?>(null) + (selectedModel?.demucsMeta?.uvrStems ?: emptyList())
                item {
                    val index = stems.indexOf(params.demucsStem).coerceAtLeast(0)
                    SliderSettingCard(
                        title = "Stems",
                        description = "All stems, or one stem plus the rest of the mix (e.g. Vocals + No Vocals).",
                        value = index.toFloat(),
                        valueRange = 0f..(stems.size - 1).coerceAtLeast(1).toFloat(),
                        steps = (stems.size - 2).coerceAtLeast(0),
                        displayValue = stems[index] ?: "All Stems",
                        onValueChange = { viewModel.updateParams { p -> p.copy(demucsStem = stems[it.roundToInt()]) } }
                    )
                }
                item {
                    val index = DEMUCS_OVERLAPS.indexOf(params.demucsOverlap).coerceAtLeast(0)
                    SliderSettingCard(
                        title = "Overlap",
                        description = "Overlap between 7.8 s segments. Higher is smoother but slower.",
                        value = index.toFloat(),
                        valueRange = 0f..(DEMUCS_OVERLAPS.size - 1).toFloat(),
                        steps = DEMUCS_OVERLAPS.size - 2,
                        displayValue = "%.2f".format(DEMUCS_OVERLAPS[index]),
                        onValueChange = {
                            viewModel.updateParams { p -> p.copy(demucsOverlap = DEMUCS_OVERLAPS[it.roundToInt()]) }
                        }
                    )
                }
                item {
                    SliderSettingCard(
                        title = "Shifts",
                        description = "Extra passes on the input shifted by up to 0.5 s, averaged. More accurate " +
                            "(desktop default 2); each shift adds a full pass. 0 = one unshifted pass.",
                        value = params.demucsShifts.toFloat(),
                        valueRange = 0f..5f,
                        steps = 4,
                        displayValue = "${params.demucsShifts}",
                        onValueChange = { viewModel.updateParams { p -> p.copy(demucsShifts = it.roundToInt()) } }
                    )
                }
            }

            // VR: desktop UVR's "Window Size" and "Aggression Setting"
            if (arch == ArchitectureType.VR_ARCH) {
                item {
                    val index = VR_WINDOW_SIZES.indexOf(params.windowSize).coerceAtLeast(0)
                    SliderSettingCard(
                        title = "Window Size",
                        description = "Spectrogram frames per model pass. 320 = better quality, 1024 = faster. " +
                            "GPU acceleration is built for 512.",
                        value = index.toFloat(),
                        valueRange = 0f..(VR_WINDOW_SIZES.size - 1).toFloat(),
                        steps = VR_WINDOW_SIZES.size - 2,
                        displayValue = "${VR_WINDOW_SIZES[index]}",
                        onValueChange = {
                            viewModel.updateParams { p -> p.copy(windowSize = VR_WINDOW_SIZES[it.roundToInt()]) }
                        }
                    )
                }
                item {
                    SliderSettingCard(
                        title = "Aggression Setting",
                        description = "Strength of primary stem extraction. Desktop UVR default is 5.",
                        value = params.aggressiveness,
                        valueRange = 0f..0.5f,
                        steps = 49,
                        displayValue = "%d".format((params.aggressiveness * 100).roundToInt()),
                        onValueChange = { viewModel.updateParams { p -> p.copy(aggressiveness = it) } }
                    )
                }
                item {
                    SwitchSettingCard(
                        title = "TTA",
                        description = "Test-Time Augmentation: runs a second, half-window-shifted pass and " +
                            "averages the two. More accurate, twice the model time. Off by default, as on desktop UVR.",
                        checked = params.tta,
                        onCheckedChange = { viewModel.updateParams { p -> p.copy(tta = it) } }
                    )
                }
                item {
                    SwitchSettingCard(
                        title = "GPU Acceleration",
                        description = "Runs the model on the GPU in full precision (FP32), checked against " +
                            "the CPU result before use. About 3× faster. Needs the model's .tflite file.",
                        checked = params.useGpu,
                        onCheckedChange = { viewModel.updateParams { p -> p.copy(useGpu = it) } }
                    )
                }
            }

            // Stem name customization (Demucs stems are named by the model)
            if (arch != ArchitectureType.DEMUCS) item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                ) {
                    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Stem Names", style = MaterialTheme.typography.bodyLarge)
                        OutlinedTextField(
                            value = params.primaryStemName,
                            onValueChange = { viewModel.updateParams { p -> p.copy(primaryStemName = it) } },
                            label = { Text("Primary Stem (model output)") },
                            modifier = Modifier.fillMaxWidth()
                        )
                        OutlinedTextField(
                            value = params.secondaryStemName,
                            onValueChange = { viewModel.updateParams { p -> p.copy(secondaryStemName = it) } },
                            label = { Text("Secondary Stem (residual)") },
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }
        }
    }
}

private val VR_WINDOW_SIZES = listOf(320, 512, 1024)
private val MDX_OVERLAPS = listOf(null, 0.25f, 0.5f, 0.75f, 0.99f)
private val DEMUCS_OVERLAPS = listOf(0.25f, 0.5f, 0.75f, 0.99f)

@Composable
private fun SwitchSettingCard(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.bodyLarge)
                Text(
                    description,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
            }
            Spacer(Modifier.width(12.dp))
            Switch(checked = checked, onCheckedChange = onCheckedChange)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChoiceSettingCard(
    title: String,
    options: List<String>,
    selected: Int,
    description: String,
    onSelect: (Int) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Spacer(Modifier.height(8.dp))
            SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                options.forEachIndexed { i, label ->
                    SegmentedButton(
                        selected = i == selected,
                        onClick = { onSelect(i) },
                        shape = SegmentedButtonDefaults.itemShape(index = i, count = options.size)
                    ) { Text(label) }
                }
            }
            Spacer(Modifier.height(8.dp))
            Text(
                description,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
        }
    }
}

@Composable
private fun SliderSettingCard(
    title: String,
    description: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int,
    displayValue: String,
    onValueChange: (Float) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(title, style = MaterialTheme.typography.bodyLarge)
                Text(
                    displayValue,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            Text(
                description,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
            Spacer(Modifier.height(8.dp))
            Slider(
                value = value,
                onValueChange = onValueChange,
                valueRange = valueRange,
                steps = steps,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}
