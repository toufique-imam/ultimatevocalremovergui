"""Scores the benchmark and checks whether the phone mirrors the desktop. Writes
<work>/benchmark.md, <work>/models.csv and <work>/songs.csv, and prints the report.

Quality: SDR = 10·log10(Σ true² / Σ (true − estimate)²) per song and stem against MUSDB's true
stems, median over songs (stems quieter than −60 dBFS in a 7 s preview are skipped). For
Demucs, the instrumental is the sum of its non-vocal stems. Speed: seconds of audio per second
of processing (× realtime), model loading included, as a user experiences one run. Value:
quality × speed. Pareto: no other model is both faster and better on that platform.

    python score.py <work dir>
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

import librosa
import numpy as np
import soundfile as sf
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODELS

QUIET_DBFS = -60
TARGETS = ('vocals', 'instrumental', 'drums', 'bass', 'other')

def sdr(ref, est):
    n = min(len(ref), len(est))
    ref, est = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    return 10 * np.log10((np.sum(ref ** 2) + 1e-7) / (np.sum((ref - est) ** 2) + 1e-7))

def to_targets(stems):
    """Model stems (by UVR name) -> MUSDB targets."""
    targets = {}
    if 'Vocals' in stems:
        targets['vocals'] = stems['Vocals']
    if 'Instrumental' in stems:
        targets['instrumental'] = stems['Instrumental']
    elif 'Vocals' in stems:
        targets['instrumental'] = sum(a for name, a in stems.items() if name != 'Vocals')
    for target, names in (('drums', ['Drums']), ('bass', ['Bass']), ('other', ['Other', 'Guitar', 'Piano'])):
        present = [stems[n] for n in names if n in stems]
        if present and 'Instrumental' not in stems:
            targets[target] = sum(present)
    return targets

def load_desktop(path_glob):
    stems = {}
    for path in glob.glob(path_glob):
        m = re.search(r'_\((.+)\)\.wav$', path)
        if m:
            stems[m.group(1)] = sf.read(path, dtype='float32', always_2d=True)[0]
    return stems

def load_phone(model_dir, result):
    stems = {}
    for name, info in result['stems'].items():
        audio = np.fromfile(os.path.join(model_dir, info['file']), np.float32).reshape(-1, 2)
        if info['rate'] != 44100:
            # Desktop VR resamples a 32 kHz model's output like this before saving (separate.py SeperateVR)
            audio = librosa.resample(audio.T.astype(np.float64), orig_sr=info['rate'], target_sr=44100).T.astype(np.float32)
        stems[name] = audio
    return stems

def score_group(work, group, stems):
    """{song id: {target: SDR}} for one separated group file."""
    scores = {}
    targets = to_targets(stems)
    for song in group['songs']:
        a, b = song['offset'], song['offset'] + song['length']
        scores[song['id']] = {}
        for target, est in targets.items():
            ref = np.fromfile(os.path.join(work, 'refs', f"{song['id']}_{target}.f32"), np.float32).reshape(-1, 2)
            if 10 * np.log10(np.mean(ref.astype(np.float64) ** 2) + 1e-30) < QUIET_DBFS:
                continue
            scores[song['id']][target] = sdr(ref, est[a:b])
    return scores

def median_scores(scores, songs=None):
    out = {}
    for target in TARGETS:
        values = [s[target] for sid, s in scores.items() if target in s and (songs is None or sid in songs)]
        if values:
            out[target] = statistics.median(values)
    return out

def quality(m):
    """One number per model: mean of median vocals and instrumental SDR."""
    return (m['vocals'] + m['instrumental']) / 2

def pareto(rows, speed_key):
    best = set()
    for r in rows:
        if r.get(speed_key) is None:
            continue
        if not any(o is not r and o.get(speed_key) is not None and o[speed_key] >= r[speed_key] and o['q'] >= r['q']
                   and (o[speed_key] > r[speed_key] or o['q'] > r['q']) for o in rows):
            best.add(r['model'])
    return best

DEVICE_NAMES = {'SM-M127G': 'Galaxy M12', 'SM-G780G': 'Galaxy S20 FE'}

def phones(work):
    """(folder, label) for every phone run: <work>/phone*/device.json."""
    found = []
    for d in sorted(glob.glob(os.path.join(work, 'phone*'))):
        info_path = os.path.join(d, 'device.json')
        if os.path.exists(info_path):
            info = json.load(open(info_path))
            found.append((os.path.basename(d), f"{DEVICE_NAMES.get(info['model'], info['model'])} ({info['soc']})"))
    return found

def main():
    work = sys.argv[1]
    groups = json.load(open(os.path.join(work, 'groups.json')))
    if not glob.glob(os.path.join(work, 'refs', '*.f32')):
        raise SystemExit(f'No true stems in {work}/refs: rebuild them with prepare_musdb.py first')
    devices = phones(work)
    rows, song_rows, mirror = [], [], []

    for key in MODELS:
        cpu_result = os.path.join(work, 'desktop', key, 'cpu', 'result.json')
        # Desktop audio: the CPU run, or for VR whose buggy CPU run was deleted, the fixed (zeroed) run.
        # Models whose separated audio was deleted after scoring (their scores are in models.csv) are left out.
        audio = next((v for v in ('cpu', 'zeroed') if glob.glob(os.path.join(work, 'desktop', key, v, '*.wav'))), None)
        if not os.path.exists(cpu_result) or audio is None:
            continue
        # Size from the desktop's model file; None once that file is deleted (results stay scorable)
        size = os.path.getsize(MODELS[key][3]) / 1e6 if os.path.exists(MODELS[key][3]) else None
        row = {'model': key, 'size_mb': size}
        desktop_scores = {}
        for group in groups:
            desktop_scores.update(score_group(work, group, load_desktop(os.path.join(work, 'desktop', key, audio, f"1_{group['id']}_(*).wav"))))
        row.update({f'desktop_{t}': v for t, v in median_scores(desktop_scores).items()})
        cpu = json.load(open(cpu_result))
        row['desktop_cpu_x'] = statistics.median(r['audio_s'] / r['wall_s'] for r in cpu['runs'])
        row['desktop_peak_mb'] = cpu['runs'][0]['peak_rss_kb_so_far'] / 1024
        gpu_result = os.path.join(work, 'desktop', key, 'gpu', 'result.json')
        if os.path.exists(gpu_result):
            run = json.load(open(gpu_result))['runs'][0]
            row['desktop_gpu_x'] = run['audio_s'] / run['wall_s']
        for sid, sc in desktop_scores.items():
            song_rows.append({'model': key, 'platform': 'desktop', 'song': sid, **{t: round(v, 2) for t, v in sc.items()}})

        zeroed_scores = None
        if os.path.exists(os.path.join(work, 'desktop', key, 'zeroed', 'result.json')):
            zeroed_scores = {}
            for group in groups:
                zeroed_scores.update(score_group(work, group, load_desktop(os.path.join(work, 'desktop', key, 'zeroed', f"1_{group['id']}_(*).wav"))))
            row.update({f'zeroed_{t}': v for t, v in median_scores(zeroed_scores).items()})

        for folder, label in devices:
            phone_dir = os.path.join(work, folder, key)
            if not os.path.exists(os.path.join(phone_dir, 'result.json')):
                continue
            result = json.load(open(os.path.join(phone_dir, 'result.json')))
            phone_scores = score_group(work, groups[0], load_phone(phone_dir, result))
            row[f'{folder}_x'] = groups[0]['seconds'] / (result['separation_ms'] / 1000)
            row[f'{folder}_peak_mb'] = (result['peak_rss_kb'] or 0) / 1024 or None
            row.update({f'{folder}_{t}': v for t, v in median_scores(phone_scores).items()})
            for sid, sc in phone_scores.items():
                song_rows.append({'model': key, 'platform': folder, 'song': sid, **{t: round(v, 2) for t, v in sc.items()}})
            # Mirror: same 10 songs, same targets
            for against, ref_scores in (('desktop', desktop_scores), ('desktop, bug fixed', zeroed_scores)):
                if ref_scores is None:
                    continue
                diffs = [phone_scores[sid][t] - ref_scores[sid][t] for sid in phone_scores for t in phone_scores[sid] if t in ref_scores.get(sid, {})]
                mirror.append({'model': key, 'phone': label, 'against': against,
                               'max_abs_sdr_diff': max(abs(d) for d in diffs), 'mean_sdr_diff': statistics.mean(diffs)})
        row['q'] = quality({t: row[f'desktop_{t}'] for t in ('vocals', 'instrumental')})
        row['desktop_value'] = row['q'] * row['desktop_cpu_x']
        for folder, _ in devices:
            if f'{folder}_x' in row:
                row[f'{folder}_value'] = row['q'] * row[f'{folder}_x']
        rows.append(row)

    f1 = lambda v: '–' if v is None else f'{v:.1f}'
    f2 = lambda v: '–' if v is None else f'{v:.2f}'
    lines = ['## Quality (desktop, 50 MUSDB18 songs, median SDR dB)', '',
             '| Model | Vocals | Instrumental | Drums | Bass | Other | VR with desktop bug fixed (voc / inst) |', '|---|---|---|---|---|---|---|']
    for r in rows:
        fixed = f"{f1(r.get('zeroed_vocals'))} / {f1(r.get('zeroed_instrumental'))}" if 'zeroed_vocals' in r else '–'
        lines.append(f"| {r['model']} | {f1(r.get('desktop_vocals'))} | {f1(r.get('desktop_instrumental'))} | {f1(r.get('desktop_drums'))} | "
                     f"{f1(r.get('desktop_bass'))} | {f1(r.get('desktop_other'))} | {fixed} |")

    labels = [label for _, label in devices]
    lines += ['', '## Speed and efficiency (95 s of audio, model loading included; × realtime, higher is faster)', '',
              '| Model | Desktop CPU | Desktop GPU | ' + ' | '.join(labels) + ' | Desktop peak MB | ' + ' | '.join(f'{l} peak MB' for l in labels) + ' | Model MB |',
              '|---|---|---|' + '---|' * len(labels) + '---|' + '---|' * len(labels) + '---|']
    for r in rows:
        lines.append(f"| {r['model']} | {f2(r['desktop_cpu_x'])} | {f2(r.get('desktop_gpu_x'))} | "
                     + ' | '.join(f2(r.get(f'{f}_x')) for f, _ in devices)
                     + f" | {r['desktop_peak_mb']:.0f} | " + ' | '.join(f"{r[f'{f}_peak_mb']:.0f}" if r.get(f'{f}_peak_mb') else '–' for f, _ in devices)
                     + f" | {'–' if r['size_mb'] is None else round(r['size_mb'])} |")

    platforms = [('desktop', 'Desktop CPU', 'desktop_cpu_x')] + [(f, l, f'{f}_x') for f, l in devices]
    ranks, fronts = {}, {}
    for key, _, speed in platforms:
        have = [r for r in rows if f'{key}_value' in r]
        ranks[key] = {r['model']: i + 1 for i, r in enumerate(sorted(have, key=lambda r: -r[f'{key}_value']))}
        fronts[key] = pareto(have, speed)
    lines += ['', '## Value (quality × speed; rank 1 = best value; ✓ = Pareto-optimal, no model both faster and better)', '',
              '| Model | Quality (mean voc/inst SDR) | ' + ' | '.join(f'{name} rank' for _, name, _ in platforms) + ' |',
              '|---|---|' + '---|' * len(platforms)]
    for r in sorted(rows, key=lambda r: ranks['desktop'][r['model']]):
        lines.append(f"| {r['model']} | {r['q']:.2f} | " + ' | '.join(
            f"{ranks[k].get(r['model'], '–')}{' ✓' if r['model'] in fronts[k] else ''}" for k, _, _ in platforms) + ' |')

    lines += ['', '## Does the phone mirror the desktop?', '']
    if mirror:
        lines += ['Quality on the same 10 songs (phone SDR − desktop SDR, dB):', '',
                  '| Model | Phone | Against | Mean difference | Largest difference |', '|---|---|---|---|---|']
        lines += [f"| {m['model']} | {m['phone']} | {m['against']} | {m['mean_sdr_diff']:+.3f} | {m['max_abs_sdr_diff']:.3f} |" for m in mirror]

    def rho(a, b):
        both = [r for r in rows if r.get(a) is not None and r.get(b) is not None]
        return (spearmanr([r[a] for r in both], [r[b] for r in both])[0], len(both)) if len(both) >= 3 else (None, len(both))
    lines += ['', 'Rank agreement across models (Spearman ρ: 1 = same order, 0 = unrelated):', '',
              '| Comparison | Speed | Value | Peak memory | Models |', '|---|---|---|---|---|']
    pairs = [(f'Desktop CPU vs {l}', 'desktop_cpu_x', f'{f}_x', 'desktop_value', f'{f}_value', 'desktop_peak_mb', f'{f}_peak_mb') for f, l in devices]
    pairs += [(f'Desktop GPU vs {l}', 'desktop_gpu_x', f'{f}_x', None, None, None, None) for f, l in devices]
    if len(devices) == 2:
        (a, la), (b, lb) = devices
        pairs.append((f'{la} vs {lb}', f'{a}_x', f'{b}_x', f'{a}_value', f'{b}_value', f'{a}_peak_mb', f'{b}_peak_mb'))
    for name, s1, s2, v1, v2, m1, m2 in pairs:
        speed, n = rho(s1, s2)
        value = rho(v1, v2)[0] if v1 else None
        memory = rho(m1, m2)[0] if m1 else None
        lines.append(f'| {name} | {f2(speed)} | {f2(value)} | {f2(memory)} | {n} |')
    for key, name, _ in platforms[1:]:
        both = fronts['desktop'] & fronts[key]
        lines.append(f"\nPareto-optimal on desktop and {name}: {', '.join(sorted(both)) or 'none'}; "
                     f"desktop only: {', '.join(sorted(fronts['desktop'] - fronts[key])) or 'none'}; "
                     f"{name} only: {', '.join(sorted(fronts[key] - fronts['desktop'])) or 'none'}")

    report = '\n'.join(lines)
    open(os.path.join(work, 'benchmark.md'), 'w').write(report + '\n')
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != 'model', k))
    with open(os.path.join(work, 'models.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(os.path.join(work, 'songs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'platform', 'song', *TARGETS]); w.writeheader(); w.writerows(song_rows)
    print(report)

if __name__ == '__main__':
    main()
