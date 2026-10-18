"""Compares the phone's stems (phone_run.py) with desktop UVR's (desktop_refs.py) for every phone
run (each <work>/phone*/ folder), model, clip and stem. Writes <work>/report.csv and
<work>/report.md and prints the summary; with --listen, also writes each run's most audible difference
(desktop, phone and their difference, ±2 s of context) to <work>/listen/.

SNR = 10·log10(Σ desktop² / Σ (desktop − phone)²): how far below the desktop stem the difference
lies. Above ~60 dB the two are numerically near-identical; 16-bit audio's own rounding is ~96 dB.
Worst-second SNR is the lowest SNR over the 1-second blocks where the desktop stem is above
−50 dBFS (so a quiet passage with a relatively large error shows up). Worst vs mix is the same
over 100 ms blocks but against the input mix's level — the closest of these to "audible"; the
--listen files are cut around that moment. Difference level is the difference's RMS in dBFS;
high band is the share of its energy above 8 kHz.
Near-silent stems (desktop level under −60 dBFS) are listed but left out of the summary.

    python compare.py <work dir> [--listen]
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics

import librosa
import numpy as np
import soundfile as sf

QUIET_DBFS = -60
BLOCK_QUIET_DBFS = -50
HIGH_HZ = 8000

def db(x):
    return 10 * np.log10(x + 1e-30)

def block_snr(ref, diff, rate):
    """(lowest SNR, its start sample) over the 1-s blocks loud enough to judge."""
    n = len(ref) // rate
    if n == 0:
        return float('nan'), 0
    sig = np.sum(ref[:n * rate].reshape(n, rate, -1) ** 2, axis=(1, 2))
    err = np.sum(diff[:n * rate].reshape(n, rate, -1) ** 2, axis=(1, 2))
    loud = db(sig / (rate * ref.shape[1])) > BLOCK_QUIET_DBFS
    if not loud.any():
        return float('nan'), 0
    snr = np.where(loud, db(sig) - db(err), np.inf)
    worst = int(np.argmin(snr))
    return float(snr[worst]), worst * rate

def worst_vs_mix(mix, diff, rate):
    """(lowest mix-level-over-difference in dB, its start sample) over 100 ms blocks of a non-silent mix."""
    h = rate // 10
    n = min(len(mix), len(diff)) // h
    if n == 0:
        return float('nan'), 0
    m = np.mean(mix[:n * h].reshape(n, h, -1) ** 2, axis=(1, 2))
    e = np.mean(diff[:n * h].reshape(n, h, -1) ** 2, axis=(1, 2))
    ratio = np.where(db(m) > BLOCK_QUIET_DBFS, db(m) - db(e), np.inf)
    worst = int(np.argmin(ratio))
    return float(ratio[worst]), worst * h

def high_band_share(diff, rate):
    spec = np.abs(np.fft.rfft(diff.mean(axis=1))) ** 2
    freqs = np.fft.rfftfreq(len(diff), 1 / rate)
    return float(spec[freqs >= HIGH_HZ].sum() / max(spec.sum(), 1e-30))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work')
    parser.add_argument('--listen', action='store_true')
    args = parser.parse_args()
    work = args.work
    clips = {c['id']: c for c in json.load(open(os.path.join(work, 'clips.json')))}
    rows, worst_moments = [], {}

    for model_dir in sorted(glob.glob(os.path.join(work, 'phone*', '*'))):
        run, key = os.path.basename(os.path.dirname(model_dir)), os.path.basename(model_dir)
        timings_path = os.path.join(model_dir, 'timings.json')
        if not os.path.exists(timings_path):
            continue
        timings = json.load(open(timings_path))
        desktop = {}
        for path in glob.glob(os.path.join(work, 'desktop', key, '*.wav')):
            m = re.match(r'^\d+_(clip\d+)_\((.+)\)\.wav$', os.path.basename(path))
            if m:
                desktop[(m.group(1), m.group(2))] = path

        for cid, info in sorted(timings.items()):
            seconds = clips[cid]['seconds']
            for stem, out in info['stems'].items():
                row = {'run': run, 'model': key, 'clip': cid, 'song': clips[cid]['song'], 'stem': stem, 'seconds': round(seconds, 1),
                       'phone_s': info['separation_ms'] / 1000, 'phone_x_realtime': round(seconds / (info['separation_ms'] / 1000), 3)}
                ref_path = desktop.get((cid, stem))
                if ref_path is None:
                    rows.append({**row, 'note': 'no desktop stem'})
                    continue
                ref, ref_rate = sf.read(ref_path, dtype='float32', always_2d=True)
                phone = np.fromfile(os.path.join(model_dir, out['file']), np.float32).reshape(-1, 2)
                notes = []
                if ref_rate != out['rate']:
                    # Desktop VR resamples a 32 kHz model's output to 44.1 kHz this way before saving
                    # (separate.py SeperateVR); the phone keeps the model's rate until it encodes
                    phone = librosa.resample(phone.T.astype(np.float64), orig_sr=out['rate'], target_sr=ref_rate).T
                n = min(len(ref), len(phone))
                ref64, phone64 = ref[:n].astype(np.float64), phone[:n].astype(np.float64)
                diff = ref64 - phone64
                signal = float(np.sum(ref64 ** 2))
                noise = float(np.sum(diff ** 2))
                if len(ref) != len(phone):
                    notes.append(f'length {len(phone)} vs desktop {len(ref)}')
                worst_snr, _ = block_snr(ref64, diff, ref_rate)
                mix = np.fromfile(os.path.join(work, 'clips', f'{cid}.f32'), np.float32).reshape(-1, 2).astype(np.float64)
                vs_mix, worst_at = worst_vs_mix(mix, diff, ref_rate) if ref_rate == 44100 else (float('nan'), 0)
                rows.append({**row, 'rate': ref_rate,
                             'snr_db': round(db(signal) - db(noise), 1) if noise > 0 else float('inf'),
                             'worst_second_snr_db': round(worst_snr, 1), 'worst_vs_mix_db': round(vs_mix, 1),
                             'worst_vs_mix_at_s': round(worst_at / ref_rate, 1),
                             'diff_dbfs': round(db(noise / max(diff.size, 1)), 1),
                             'diff_peak': float(np.abs(diff).max()) if n else float('nan'),
                             'diff_high_band': round(high_band_share(diff, ref_rate), 3) if noise > 0 else 0.0,
                             'desktop_dbfs': round(db(signal / max(ref64.size, 1)), 1), 'note': '; '.join(notes)})
                moment = worst_moments.get((run, key, stem))
                # Same pick as the summary table (first clip with the lowest rounded value), quiet stems excluded
                if (args.listen and np.isfinite(vs_mix) and rows[-1]['desktop_dbfs'] > QUIET_DBFS
                        and (moment is None or round(vs_mix, 1) < moment[0])):
                    a, b = max(worst_at - 2 * ref_rate, 0), min(worst_at + 2 * ref_rate, n)
                    worst_moments[(run, key, stem)] = (round(vs_mix, 1), cid, ref64[a:b], phone64[a:b], ref_rate, a / ref_rate)

    fields = ['run', 'model', 'clip', 'song', 'stem', 'seconds', 'rate', 'snr_db', 'worst_second_snr_db', 'worst_vs_mix_db', 'worst_vs_mix_at_s',
              'diff_dbfs', 'diff_peak', 'diff_high_band', 'desktop_dbfs', 'phone_s', 'phone_x_realtime', 'note']
    with open(os.path.join(work, 'report.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ['| Run | Model | Stem | Clips | Median SNR (dB) | Worst SNR (dB) | Worst second (dB) | Worst vs mix (dB) | '
             'Loudest difference (dBFS) | Difference above 8 kHz | Phone speed (× realtime) |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for run, key in sorted({(r['run'], r['model']) for r in rows}):
        run_rows = [r for r in rows if r['run'] == run and r['model'] == key]
        speed = sum({r['clip']: r['seconds'] for r in run_rows}.values()) / sum({r['clip']: r['phone_s'] for r in run_rows}.values())
        for stem in sorted({r['stem'] for r in run_rows}):
            scored = [r for r in run_rows if r['stem'] == stem and 'snr_db' in r and r['desktop_dbfs'] > QUIET_DBFS]
            if not scored:
                lines.append(f'| {run} | {key} | {stem} | 0 | – | – | – | – | – | – | {speed:.2f} |')
                continue
            worst = min(scored, key=lambda r: r['snr_db'])
            worst_second = min(scored, key=lambda r: r['worst_second_snr_db'])
            worst_mix = min(scored, key=lambda r: r['worst_vs_mix_db'])
            lines.append(f"| {run} | {key} | {stem} | {len(scored)} | {statistics.median(r['snr_db'] for r in scored):.1f} | "
                         f"{worst['snr_db']:.1f} ({worst['clip']}) | {worst_second['worst_second_snr_db']:.1f} ({worst_second['clip']}) | "
                         f"{worst_mix['worst_vs_mix_db']:.1f} ({worst_mix['clip']} @ {worst_mix['worst_vs_mix_at_s']} s) | "
                         f"{max(r['diff_dbfs'] for r in scored):.1f} | "
                         f"{statistics.median(r['diff_high_band'] for r in scored):.0%} | {speed:.2f} |")
    problems = [r for r in rows if r.get('note')]
    if problems:
        lines += ['', 'Notes:'] + [f"- {r['run']} {r['model']} {r['clip']} {r['stem']}: {r['note']}" for r in problems]

    if worst_moments:
        listen = os.path.join(work, 'listen')
        os.makedirs(listen, exist_ok=True)
        lines += ['', f'Worst moment vs mix of each run, ±2 s, in {listen}:']
        for (run, key, stem), (snr, cid, ref, phone, rate, start) in sorted(worst_moments.items()):
            base = os.path.join(listen, f'{run}_{key}_{stem}_{cid}')
            # 24-bit PCM plays everywhere (QuickTime, browsers); the difference is also given 20 dB louder
            sf.write(base + '_desktop.wav', ref, rate, subtype='PCM_24')
            sf.write(base + '_phone.wav', phone, rate, subtype='PCM_24')
            sf.write(base + '_difference.wav', ref - phone, rate, subtype='PCM_24')
            sf.write(base + '_difference_+20dB.wav', np.clip(10 * (ref - phone), -1, 1), rate, subtype='PCM_24')
            lines.append(f'- {os.path.basename(base)} (from {start:.1f} s): difference {snr:.1f} dB below the mix')

    report = '\n'.join(lines)
    with open(os.path.join(work, 'report.md'), 'w') as f:
        f.write(report + '\n')
    print(report)

if __name__ == '__main__':
    main()
