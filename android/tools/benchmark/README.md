# Desktop vs phone benchmark

Quality, speed, memory and value of each model on desktop UVR and on phones, and whether the
phones rank the models the way the desktop does. Quality is SDR against the true stems of the
MUSDB18 7-second test previews (`pip install musdb` downloads them).

```bash
W=~/Desktop/test_uvr/output/benchmark
python prepare_musdb.py ~/Desktop/test_uvr/musdb18_7s $W           # 5 files of 10 songs, 3 s gaps
python desktop_bench.py $W                                          # UVR.py: cpu, gpu, zeroed variants
ANDROID_SERIAL=<serial> python phone_bench.py $W [--out phone_s20fe] # group1 per model, per phone
python progress.py $W                                               # live progress of all devices
python score.py $W                                                  # benchmark.md, models.csv, songs.csv
```

- Models are listed in `models.py`; settings are UVR's defaults on both sides except float WAV
  output and Demucs shifts = 0 (desktop's shift offsets are random).
- Each desktop model/variant runs in its own process so peak memory is its own; `zeroed` reruns
  VR with `cmb_spectrogram_to_wave`'s array zero-initialised (now fixed in `lib_v5/spec_utils.py`).
- Both runners resume where they stopped. Install the app and test APKs on each phone first and
  set `ANDROID_SERIAL` when several devices are connected.
