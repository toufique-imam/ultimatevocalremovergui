"""Live progress of the comparison runs: desktop UVR and every phone run (<work>/phone*/), per
song, with the current song's MDX chunk / VR window from the app's log and a finish estimate from
the speed measured so far. Refreshes every few seconds; Ctrl+C to stop.

    python progress.py <work dir> [--once]
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

ADB = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
DEVICE_NAMES = {'SM-M127G': 'Galaxy M12', 'SM-G780G': 'Galaxy S20 FE', 'CPH2891': 'OPPO CPH2891'}
TAGS = ['Separator[VR_ARCH/PTL]:*', 'Separator[MDX_NET/ONNX]:*', 'Separator[DEMUCS/DEMUCS]:*']
BAR = 24

def bar(fraction):
    filled = int(round(BAR * max(0.0, min(1.0, fraction))))
    return '█' * filled + '░' * (BAR - filled)

def clock(seconds):
    if seconds is None:
        return '?'
    seconds = int(seconds)
    return f'{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}' if seconds >= 3600 else f'{seconds // 60}:{seconds % 60:02d}'

def phone_log(serial, since):
    try:
        # Only lines since the current song started; adb can be slow while stems are pulled
        out = subprocess.run([ADB, '-s', serial, 'logcat', '-d', '-v', 'epoch', '-t', f'{since:.3f}', '-s', *TAGS],
                             capture_output=True, text=True, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return []
    return [l for l in out.splitlines() for m in [re.match(r'\s*(\d+\.\d+)\s', l)] if m and float(m.group(1)) >= since]

def summary(clips, done_seconds, done_time, elapsed, fraction_of_current, current):
    """(overall fraction, '…  ~left') from audio seconds done vs total."""
    total = sum(c['seconds'] for c in clips)
    done = done_seconds + (current['seconds'] * fraction_of_current if current else 0)
    rate = done_seconds / done_time if done_time else None     # audio seconds per second
    if rate is None and current and fraction_of_current > 0 and elapsed > 0:
        rate = current['seconds'] * fraction_of_current / elapsed   # first song: from its chunks so far
    left = (total - done) / rate if rate else None
    if rate and current and fraction_of_current == 0:
        left = max((total - done_seconds) / rate - elapsed, 0)
    return done / total, (f'   {rate:.2f}× realtime   ~{clock(left)} left' if rate else '')

def desktop_status(work, clips):
    rows = []
    for d in sorted(glob.glob(os.path.join(work, 'desktop', '*'))):
        wavs = glob.glob(os.path.join(d, '*.wav'))
        done_ids = {re.match(r'^\d+_(clip\d+)_', os.path.basename(p)).group(1) for p in wavs if re.match(r'^\d+_(clip\d+)_', os.path.basename(p))}
        done = [c for c in clips if c['id'] in done_ids]
        if not wavs:
            rows.append((f'Desktop {os.path.basename(d)}', 0.0, 'starting'))
            continue
        last = max(os.path.getmtime(p) for p in wavs)
        busy = len(done) < len(clips) and time.time() - last < 900
        fraction, eta = summary(clips, sum(c['seconds'] for c in done), last - os.stat(d).st_birthtime, time.time() - last, 0,
                                next((c for c in clips if c['id'] not in done_ids), None) if busy else None)
        state = 'done' if len(done) == len(clips) else ('running' if busy else 'stopped')
        rows.append((f'Desktop {os.path.basename(d)}', fraction, f'{len(done)}/{len(clips)} songs   {state}' + (eta if busy else '')))
    return rows

def phone_status(work, folder, info, clips):
    rows = []
    run_dirs = sorted(d for d in glob.glob(os.path.join(work, folder, '*')) if os.path.isdir(d))
    if not run_dirs:
        return [(f"{DEVICE_NAMES.get(info['model'], info['model'])} {'GPU' if info.get('gpu', True) else 'CPU'}", 0.0, 'waiting')]
    for d in run_dirs:
        name = f"{DEVICE_NAMES.get(info['model'], info['model'])} {'GPU' if info.get('gpu', True) else 'CPU'} {os.path.basename(d)}"
        path = os.path.join(d, 'timings.json')
        timings = json.load(open(path)) if os.path.exists(path) else {}
        done = [c for c in clips if c['id'] in timings]
        current = next((c for c in clips if c['id'] not in timings), None)
        since = os.path.getmtime(path) if timings else os.stat(d).st_birthtime
        done_time = sum(t['wall_s'] for t in timings.values())
        detail, fraction = '', 0.0
        if current:
            log = phone_log(info['serial'], since)
            total = next((int(m.group(1)) for l in reversed(log) for m in [re.search(r'(?:windows|chunks)=(\d+)', l)] if m), None)
            step = next((int(m.group(1)) for l in reversed(log) for m in [re.search(r'model (?:window|call) (\d+)', l)] if m), None)
            check = next((l.split(': ', 1)[-1] for l in reversed(log) if 'Fast engine' in l or 'too quiet' in l), None)
            if total:
                fraction = (step or 0) / total
                detail = f"   now {current['id']} chunk {step or 0}/{total}"
            else:
                detail = f"   now {current['id']} loading"
            if check:
                detail += f"   [{check[:90]}]"
        overall, eta = summary(clips, sum(c['seconds'] for c in done), done_time, time.time() - since, fraction, current)
        rows.append((name, overall, f"{len(done)}/{len(clips)} songs{detail}{eta}"))
    return rows

def render(work):
    clips = json.load(open(os.path.join(work, 'clips.json')))
    rows = desktop_status(work, clips)
    for d in sorted(glob.glob(os.path.join(work, 'phone*', 'device.json'))):
        rows += phone_status(work, os.path.basename(os.path.dirname(d)), json.load(open(d)), clips)
    width = max(len(r[0]) for r in rows)
    out = [f"UVR comparison — {time.strftime('%H:%M:%S')}  {len(clips)} songs, {sum(c['seconds'] for c in clips) / 60:.0f} min  (Ctrl+C to stop)", '']
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
