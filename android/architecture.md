# Ultimate Vocal Remover (UVR) Android Architecture

This document outlines the high-level architecture and component interactions of the UVR Android application.

## Overview
UVR Android is a mobile port of the Ultimate Vocal Remover GUI. It utilizes deep learning models (ONNX and PyTorch Lite) to perform high-quality audio source separation (e.g., separating vocals from instrumentals) directly on the device.

---

## 1. High-Level Layers

### **UI Layer (Jetpack Compose)**
- **MainActivity**: The single entry point of the application. Handles deep links and external audio file intents.
- **NavHost**: Manages navigation between different screens.
- **Screens**:
    - `HomeScreen`: Main dashboard for selecting files and initiating separation.
    - `ModelsScreen`: Management of downloaded/available AI models.
    - `SettingsScreen`: Configuration for inference (segment size, overlap, etc.).
    - `SeparationScreen`: Real-time progress and results of the separation process.
- **ViewModel (`MainViewModel`)**: Holds UI state, coordinates between the UI and background services, and manages user preferences via DataStore.

### **Service Layer**
- **SeparationService**: A Foreground Service that performs the heavy lifting of audio separation. Using a foreground service ensures the process isn't killed by the OS during long-running inference tasks.

### **Inference Engine Layer (ML)**
- **Separator (Abstract)**: Base class for architecture-specific separation logic.
    - `VRSeparator`: Handles VR Architecture models via `ml/vr/VrPipeline` (see below).
    - `MDXSeparator`: MDX-Net — `ml/mdx/MdxPipeline` ports desktop `SeperateMDX.demix` (torch.stft/istft, chunk windowing, Denoise). Settings come from UVR's `model_data.json` by model hash (`MdxModelMeta`, table in `assets/uvr/`), so original UVR `.onnx` files work unconverted.
    - `DemucsSeparator`: Demucs v4 — `ml/demucs/DemucsPipeline` ports `apply_model` (segments, triangle weights, shifts with fixed offsets) and HTDemucs `_spec`/`_ispec`; `DemucsRunner` adds demix_demucs' normalisation, bag combining and UVR's stem order. Models are a `<name>.demucs` manifest plus `<name>_<i>.onnx` cores exported by `demucs_onnx.py` (attention computed in query chunks to keep ORT under ~2 GB).
- **InferenceEngine (Interface)**: Abstraction over the underlying ML runtime.
    - `OnnxInferenceEngine`: ONNX Runtime on the CPU in FP32 (no NNAPI), threads = performance cores; `lowMemory` disables the arena for Demucs.
    - `PtlInferenceEngine`: Wraps PyTorch Lite for `.ptl` models.
    - `TfliteGpuEngine`: LiteRT GPU delegate in FP32 for the `<name>_vr.tflite` exported next to each VR `.ptl` (~3× faster than the `.ptl` on CPU, closer to the `.pth`).
    - `VerifiedEngine`: runs the first window on both the GPU engine and the `.ptl`; keeps the GPU only if it is within 1e-3 and faster (on the Exynos 850 / Mali-G52 the CPU can win), remembering the choice per model.
- **ModelManager**: Handles loading model files from assets or internal storage and resolving configurations.
- **VrModelMeta**: Model params, primary/secondary stem and window offset, embedded in each VR `.ptl` as `extra/uvr_meta.json` by `convert.ipynb`.

### **VR Pipeline (`ml/vr`, `ml/dsp`)**
A port of desktop UVR's `SeperateVR` (separate.py) and `lib_v5/spec_utils.py` that matches desktop output to float rounding (checked by `VrPipelineFixtureTest` against the desktop code).
- Multi-band analysis: polyphase resampling (`PolyphaseResampler`, a port of `scipy.signal.resample_poly`), STFT per band (`Stft`, librosa-compatible), crop/stack into one `bins+1`-row spectrogram, pre-filter.
- Model: `|X| / max|X|` on 512-frame windows (roi 256 frames, the model offset of 128 frames of context each side), mask, aggressiveness (`adjust_aggr`).
- Synthesis: `mask·X` and `(1−mask)·X` → per-band ISTFT, band filters, resample up, sum.
- Bounded memory: the song is processed in segments of 4 model windows (~12 s). Every stage is indexed by global sample/frame, and synthesis is linear, so segment outputs overlap-add into a small rolling buffer and are flushed to disk — no seams, heap use independent of song length. Two passes: the first only computes the whole-song `max|X|` desktop normalises by.

### **Audio Processing Layer**
- **AudioProcessor**: Handles audio decoding (extracting PCM from MP3/WAV/FLAC) and encoding (saving stems back to disk) using **FFmpegKit**.
- **SpectrogramUtils**: Simple array helpers (normalisation, interleaving) used by the MDX/Demucs separators.
- Decoded audio is a headerless interleaved f32le file (44.1 kHz stereo); stems are written the same way and encoded (resampled to 44.1 kHz) at the end.
- **SilenceTrimmer**: desktop UVR's "Trim Silence" (`spec_utils.silence_keep_ranges` / `trim_silence`) on the raw stem files, streamed in two passes; `SilenceTrimmerFixtureTest` checks it against the desktop code (`UVR_TRIM_FIXTURES`). Videos keep their full length when it's on.
- **Video inputs**: when the file has a real video stream (FFprobe; cover art doesn't count), `decode` keeps a copy of it. After separation `muxVideo` saves the video once per chosen stem: the video stream is copied untouched (rotation kept) and the stem is encoded as AAC (MKV if MP4 can't hold the codec), then published to `Movies/UVR`. AAC encoding is the slow part (~2× realtime on an Exynos 850), so it shows as its own "Saving … video" progress stage.

### **Performance & Benchmarking Layer**
- **BenchmarkUtils**: Provides high-resolution timing and metrics collection for inference tasks.
- **PerformanceMetrics**: Data structure that captures:
    - `totalTimeMs`: E2E time including pre/post processing.
    - `inferenceTimeMs`: Pure ML model execution time.
    - `speedFactor`: Processing speed relative to real-time (e.g., 2.5x).
- **Logging**: All separation runs automatically log benchmark results to Logcat under the `UVR-Benchmark` tag.

---

## 2. Data Flow: The Separation Pipeline

1.  **Selection**: User selects an audio file and a model in the UI.
2.  **Initialization**: `MainViewModel` binds to `SeparationService`.
3.  **Decoding**: `AudioProcessor` uses FFmpeg to decode the source file into raw 32-bit float PCM data.
4.  **Inference**:
    - The `Separator` subclass (e.g., `MDXSeparator`) splits the PCM into segments.
    - Each segment is passed to the `InferenceEngine` (ONNX or PTL).
    - The engine returns separated frequency-domain data or time-domain stems.
5.  **Post-Processing**: Stems are reconstructed using ISTFT and inverted/mixed as needed.
6.  **Encoding**: `AudioProcessor` encodes the resulting PCM buffers into WAV files.
7.  **Benchmarking**: High-resolution timers record the duration of each stage. A `PerformanceMetrics` report is generated and logged, providing data on "Speed Factor" (e.g., how many times faster than real-time the model ran).
8.  **Completion**: The UI updates to show the output files and provides options to play or share them.

---

## 3. Technology Stack
- **Language**: Kotlin
- **UI**: Jetpack Compose
- **Concurrency**: Kotlin Coroutines & Flow
- **ML Runtimes**: 
    - ONNX Runtime (Android)
    - PyTorch Android Lite
- **Audio Core**: FFmpegKit
- **Persistence**: Jetpack DataStore (Preferences)
- **JSON Handling**: Gson
- **Background Tasks**: Foreground Service & WorkManager
- **Image Loading**: Coil (for waveform/UI assets)

---

## 4. Project Structure
```text
com.uvr.android
├── audio          # Audio decoding, encoding, and Foreground Service
├── ml             # Inference engines, Separator logic, and Model management
├── ui             # Compose screens, Theme, and Navigation
│   └── theme      # Material3 Design System implementation
├── utils          # File, Permission, and Math utilities
├── viewmodel      # MainViewModel and UI state management
└── MainActivity   # App entry point
```
