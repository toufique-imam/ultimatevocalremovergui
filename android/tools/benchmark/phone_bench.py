"""Phone side of the benchmark: separates group1 (10 MUSDB songs) with each model through the
SeparationCheck instrumentation test, recording time and peak memory, and pulls the stems back
to <work>/<out>/<model>/ (--out, default "phone"). Resumable: models with a result.json are skipped.

With several devices connected, set ANDROID_SERIAL. Install the app and test APKs first, then:
    python phone_bench.py <work dir> [--out phone_s20fe] [model ...]
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODELS, PHONE_EXTRA, PHONE_FILES

ADB = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
FILES = '/sdcard/Android/data/com.uvr.android/files'
RUNNER = 'com.uvr.android.test/androidx.test.runner.AndroidJUnitRunner'
CLIP = 'group1'

def adb(*args, check=True):
    return subprocess.run([ADB, *args], capture_output=True, text=True, check=check).stdout

def remote_exists(path):
    return adb('shell', f'ls "{path}" 2>/dev/null', check=False).strip() == path

def check_device():
    """With several devices connected, adb needs ANDROID_SERIAL to know which one to use."""
    devices = [l.split()[0] for l in adb('devices').splitlines()[1:] if l.strip().endswith('device')]
    if len(devices) > 1 and not os.environ.get('ANDROID_SERIAL'):
        raise SystemExit(f'Several devices connected ({", ".join(devices)}): set ANDROID_SERIAL to the one to use')

def main():
    check_device()
    work = sys.argv[1]
    args = sys.argv[2:]
    out_name = 'phone'
    if args[:1] == ['--out']:
        out_name, args = args[1], args[2:]
    keys = args or list(MODELS)
    os.makedirs(os.path.join(work, out_name), exist_ok=True)
    json.dump({'serial': adb('get-serialno').strip(), 'model': adb('shell', 'getprop', 'ro.product.model').strip(),
               'soc': adb('shell', 'getprop', 'ro.soc.model').strip() or adb('shell', 'getprop', 'ro.board.platform').strip()},
              open(os.path.join(work, out_name, 'device.json'), 'w'), indent=2)
    stay_on = adb('shell', 'settings', 'get', 'global', 'stay_on_while_plugged_in').strip()
    adb('shell', 'svc', 'power', 'stayon', 'usb')   # screen-off runs the CPU ~4x slower
    adb('shell', 'input', 'keyevent', 'KEYCODE_WAKEUP')
    # The app creates its folders (it can't read files in folders adb made on some Android 15 phones)
    adb('shell', 'am', 'instrument', '-w', '-e', 'class', 'com.uvr.android.bench.SeparationCheck', '-e', 'setup', '1', RUNNER)
    if not remote_exists(f'{FILES}/bench/{CLIP}.f32'):
        adb('push', os.path.join(work, 'groups', f'{CLIP}.f32'), f'{FILES}/bench/{CLIP}.f32')
    try:
        for key in keys:
            if MODELS[key][2] is None:
                continue   # desktop-only model
            out_dir = os.path.join(work, out_name, key)
            if os.path.exists(os.path.join(out_dir, 'result.json')):
                continue
            os.makedirs(out_dir, exist_ok=True)
            for path in PHONE_FILES.get(key, []):
                if not remote_exists(f'{FILES}/models/{os.path.basename(path)}'):
                    adb('push', path, f'{FILES}/models/')

            for attempt in (1, 2):
                t0 = time.time()
                log = adb('shell', 'am', 'instrument', '-w', '-e', 'class', 'com.uvr.android.bench.SeparationCheck',
                          '-e', 'model', MODELS[key][2], '-e', 'clip', f'{CLIP}.f32', '-e', 'out', f'{key}_',
                          *PHONE_EXTRA.get(key, []), RUNNER, check=False)
                outs = re.findall(r'OUT (\S+\.f32) (\d+)', log)
                sep = re.search(r'SEP .*?: (\d+) ms', log)
                peak = re.search(r'peak: (\d+) kB', log)
                if 'OK (1 test)' in log and outs and sep:
                    break
                print(f'{key}: attempt {attempt} failed\n{log[-1500:]}', flush=True)
            else:
                continue

            stems = {}
            for name, rate in outs:
                adb('pull', f'{FILES}/bench/out/{name}', os.path.join(out_dir, name))
                adb('shell', 'rm', f'{FILES}/bench/out/{name}')
                stems[name[len(key) + 1:-4]] = {'file': name, 'rate': int(rate)}
            result = {'separation_ms': int(sep.group(1)), 'peak_rss_kb': int(peak.group(1)) if peak else None,
                      'wall_s': round(time.time() - t0, 1), 'stems': stems}
            json.dump(result, open(os.path.join(out_dir, 'result.json'), 'w'), indent=2)
            print(f'{key}: {result["separation_ms"] / 1000:.1f} s, peak {result["peak_rss_kb"]} kB, stems {list(stems)}', flush=True)
    finally:
        adb('shell', 'settings', 'put', 'global', 'stay_on_while_plugged_in', stay_on or '0', check=False)

if __name__ == '__main__':
    main()
