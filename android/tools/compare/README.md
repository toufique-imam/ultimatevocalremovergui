# Desktop vs phone comparison

Checks that the Android app's converted models give the same stems as desktop UVR's
originals, on real music. Both sides separate the same decoded clips, so the only
difference is the model runtime (VR `.pth` → `.ptl`/`.tflite`, Demucs `.th` → `.onnx`,
MDX-Net the same `.onnx` on mobile ONNX Runtime and the Kotlin pipeline port).

Run from the conda base env, with the phone connected over adb:

```bash
W=~/Desktop/test_uvr/output/compare
python prepare_clips.py ~/Desktop/test_uvr/test_case $W      # 30 s of each song in SONGS
python desktop_refs.py $W                                     # desktop UVR, CPU, float WAV
python phone_run.py $W                                        # app + test APK installed first
python compare.py $W                                          # report.csv, report.md
```

- Settings are UVR's defaults on both sides (VR window 512, aggression 5, TTA off; MDX
  overlap Default, no denoise), except Demucs shifts = 0: desktop picks its shift offsets
  at random, so shifted runs can't match.
- `phone_run.py` resumes where it stopped; push the models it names to the app's
  external `files/models` folder first (the VR models are bundled).
- SNR is phone vs desktop in dB: above ~60 dB is numerically near-identical; 16-bit
  audio's own rounding is ~96 dB. Stems under −60 dBFS are left out of the summary.
