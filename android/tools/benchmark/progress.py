"""Live progress of the benchmark on the desktop and every phone, refreshed every few seconds.
Phones' in-run progress comes from the app's own log lines (VR windows, MDX chunks); Demucs and
the desktop show elapsed time against the average of the runs already finished. Ctrl+C to stop.

    python progress.py <work dir> [--once]
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desktop_bench import variants
from models import MODELS

ADB = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
DEVICE_NAMES = {'SM-M127G': 'Galaxy M12', 'SM-G780G': 'Galaxy S20 FE'}
TAGS = ['Separator[VR_ARCH/PTL]:D', 'Separator[MDX_NET/ONNX]:D', 'Separator[DEMUCS/DEMUCS]:D']
BAR = 24

def bar(fraction):
    filled = int(round(BAR * max(0.0, min(1.0, fraction))))
    return '█' * filled + '░' * (BAR - filled)

def clock(seconds):
    if seconds is None:
        return '?'
    seconds = int(seconds)
    return f'{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}' if seconds >= 3600 else f'{seconds // 60}:{seconds % 60:02d}'

def arch(key):
    return MODELS[key][0]

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None

def desktop_status(work, now):
    runs = [(k, v) for k in MODELS for v in variants(k)]
    done, current = {}, None
    for key, variant in runs:
        d = os.path.join(work, 'desktop', key, variant)
        if os.path.exists(os.path.join(d, 'result.json')):
            done[(key, variant)] = sum(r['wall_s'] for r in json.load(open(os.path.join(d, 'result.json')))['runs'])
        elif os.path.isdir(d) and current is None:
            current = (key, variant, d)
    expected = lambda k, v: mean(t for (dk, dv), t in done.items() if arch(dk) == arch(k) and dv == v) or mean(done.values())
    line = f"{len(done)}/{len(runs)} runs"
    remaining = sum(expected(k, v) or 0 for k, v in runs if (k, v) not in done)
    if current:
        key, variant, d = current
        groups = 1 if variant == 'gpu' else 5
        finished = len({re.match(r'1_(group\d)_', os.path.basename(p)).group(1) for p in glob.glob(os.path.join(d, '1_group*_(*).wav'))})
        elapsed = now - os.stat(d).st_birthtime
        line += f"   now: {key} / {variant}  group {min(finished + 1, groups)}/{groups}  elapsed {clock(elapsed)}"
        remaining -= min(elapsed, expected(key, variant) or 0)
    return len(done) / len(runs), line + (f"   ~{clock(remaining)} left" if done and current else '')

def phone_log(serial, since):
    try:
        # Only lines since the run started; adb can be slow while the runner moves stems over it
        out = subprocess.run([ADB, '-s', serial, 'logcat', '-d', '-v', 'epoch', '-t', f'{since:.3f}', '-s', *TAGS],
                             capture_output=True, text=True, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return []
    lines = []
    for line in out.splitlines():
        m = re.match(r'\s*(\d+\.\d+)\s', line)
        if m and float(m.group(1)) >= since:
            lines.append(line)
    return lines

def phone_status(work, folder, serial, now):
    order = [k for k in MODELS]
    results = {k: json.load(open(p)) for k in order for p in [os.path.join(work, folder, k, 'result.json')] if os.path.exists(p)}
    current = next((k for k in order if k not in results and os.path.isdir(os.path.join(work, folder, k))), None)
    expected = lambda k: mean(r['separation_ms'] / 1000 for dk, r in results.items() if arch(dk) == arch(k))
    line = f"{len(results)}/{len(order)} models"
    if current is None:
        return len(results) / len(order), line
    start = os.stat(os.path.join(work, folder, current)).st_birthtime
    elapsed = now - start
    detail, fraction = '', None
    log = phone_log(serial, start)
    total = next((int(m.group(1)) for l in reversed(log) for m in [re.search(r'(?:windows|chunks)=(\d+)', l)] if m), None)
    step = next((int(m.group(1)) for l in reversed(log) for m in [re.search(r'model (?:window|call) (\d+)', l)] if m), None)
    if total and step is None:
        step = 0
    if total and step is not None:
        fraction = step / total
        detail = f"{'window' if arch(current) == 'VR_ARCH_PM' else 'chunk'} {step}/{total}"
    elif step is not None:
        detail = f"model call {step}"
    guess = expected(current)
    if fraction is None and guess:
        fraction = elapsed / guess
    line += f"   now: {current}  {detail}  elapsed {clock(elapsed)}" + (f" ({int(fraction * 100)}%)" if fraction is not None else '')
    remaining = [expected(k) for k in order if k not in results and k != current]
    if guess and all(r is not None for r in remaining):
        line += f"   ~{clock(max(guess - elapsed, 0) + sum(remaining))} left"
    return (len(results) + min(fraction or 0, 0.99)) / len(order), line

def render(work):
    now = time.time()
    rows = [('Desktop (this Mac)',) + desktop_status(work, now)]
    for d in sorted(glob.glob(os.path.join(work, 'phone*', 'device.json'))):
        info = json.load(open(d))
        folder = os.path.basename(os.path.dirname(d))
        rows.append((f"{DEVICE_NAMES.get(info['model'], info['model'])} ({info['soc']})",) + phone_status(work, folder, info['serial'], now))
    width = max(len(r[0]) for r in rows)
    out = [f"UVR benchmark — {time.strftime('%H:%M:%S')}  (Ctrl+C to stop)", '']
    out += [f"{name.ljust(width)}  {bar(fraction)} {int(fraction * 100):3d}%   {line}" for name, fraction, line in rows]
    return '\n'.join(out)

def main():
    work = sys.argv[1]
    if '--once' in sys.argv:
        print(render(work))
        return
    try:
        while True:
            text = render(work)
            print('\033[2J\033[H' + text, flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
