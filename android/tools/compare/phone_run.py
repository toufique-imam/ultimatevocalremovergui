"""Phone side: separates every clip with the app's converted models through the SeparationCheck
instrumentation test and pulls each clip's stems back to <work>/phone/<model>/. Resumable: clips
whose stems were already pulled are skipped, so it can be re-run after an interruption.
--out names the results folder (default "phone"); --cpu turns the GPU off (it is used whenever
the model has a .tflite in the app's models folder).

With several devices connected, set ANDROID_SERIAL. Install the app and test APKs first (gradle assembleDebug assembleDebugAndroidTest), then:
    python phone_run.py <work dir> [--out phone_s20fe_gpu] [--cpu] [model ...]
"""
import argparse
import json
import os
import re
import subprocess
import time

ADB = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
PACKAGE_FILES = '/sdcard/Android/data/com.uvr.android/files'
RUNNER = 'com.uvr.android.test/androidx.test.runner.AndroidJUnitRunner'

# key -> (model file name in the app, extra SeparationCheck arguments). Settings match desktop UVR's
# defaults (VR window 512, aggression 5, TTA off; MDX overlap Default, no denoise), except Demucs
# shifts = 0: desktop picks shift offsets at random, which the phone can't reproduce.
MODELS = {
    'vr13': ('13_SP-UVR-4B-44100-1_vr.ptl', []),
    'mdx_kim2': ('Kim_Vocal_2.onnx', []),
    'htdemucs': ('htdemucs.demucs', ['-e', 'shifts', '0']),
}

def adb(*args, check=True):
    return subprocess.run([ADB, *args], capture_output=True, text=True, check=check).stdout

def check_device():
    """With several devices connected, adb needs ANDROID_SERIAL to know which one to use."""
    devices = [l.split()[0] for l in adb('devices').splitlines()[1:] if l.strip().endswith('device')]
    if len(devices) > 1 and not os.environ.get('ANDROID_SERIAL'):
        raise SystemExit(f'Several devices connected ({", ".join(devices)}): set ANDROID_SERIAL to the one to use')

def main():
    check_device()
    parser = argparse.ArgumentParser()
    parser.add_argument('work')
    parser.add_argument('models', nargs='*')
    parser.add_argument('--out', default='phone')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_intermixed_args()
    work, keys = args.work, args.models or list(MODELS)
    clips = json.load(open(os.path.join(work, 'clips.json')))
    os.makedirs(os.path.join(work, args.out), exist_ok=True)
    json.dump({'serial': adb('get-serialno').strip(), 'model': adb('shell', 'getprop', 'ro.product.model').strip(),
               'gpu': not args.cpu}, open(os.path.join(work, args.out, 'device.json'), 'w'), indent=2)   # for progress.py

    stay_on = adb('shell', 'settings', 'get', 'global', 'stay_on_while_plugged_in').strip()
    adb('shell', 'svc', 'power', 'stayon', 'usb')   # screen-off runs the CPU ~4x slower
    adb('shell', 'input', 'keyevent', 'KEYCODE_WAKEUP')
    # The app creates its folders (it can't read files in folders adb made on some Android 15 phones)
    adb('shell', 'am', 'instrument', '-w', '-e', 'class', 'com.uvr.android.bench.SeparationCheck', '-e', 'setup', '1', RUNNER)
    try:
        for key in keys:
            model, extra = MODELS[key]
            extra = extra + (['-e', 'gpu', '0'] if args.cpu else [])
            out_dir = os.path.join(work, args.out, key)
            os.makedirs(out_dir, exist_ok=True)
            timings_path = os.path.join(out_dir, 'timings.json')
            timings = json.load(open(timings_path)) if os.path.exists(timings_path) else {}

            for clip in clips:
                cid = clip['id']
                if cid in timings:
                    continue
                remote_clip = f'{PACKAGE_FILES}/bench/{cid}.f32'
                local_clip = os.path.join(work, 'clips', f'{cid}.f32')
                # By size: another work dir's clips (e.g. 30-s cuts) use the same names
                if adb('shell', f'stat -c %s {remote_clip} 2>/dev/null', check=False).strip() != str(os.path.getsize(local_clip)):
                    adb('push', local_clip, remote_clip)

                for attempt in (1, 2):
                    t0 = time.time()
                    log = adb('shell', 'am', 'instrument', '-w', '-e', 'class', 'com.uvr.android.bench.SeparationCheck',
                              '-e', 'model', model, '-e', 'clip', f'{cid}.f32', '-e', 'out', f'{cid}_', *extra, RUNNER,
                              check=False)
                    outs = re.findall(r'OUT (\S+\.f32) (\d+)', log)
                    sep = re.search(r'SEP .*?: (\d+) ms', log)
                    if 'OK (1 test)' in log and outs and sep:
                        break
                    print(f'{key} {cid}: attempt {attempt} failed\n{log[-1500:]}', flush=True)
                else:
                    continue

                stems = {}
                for name, rate in outs:
                    adb('pull', f'{PACKAGE_FILES}/bench/out/{name}', os.path.join(out_dir, name))
                    adb('shell', 'rm', f'{PACKAGE_FILES}/bench/out/{name}')
                    stems[name[len(cid) + 1:-4]] = {'file': name, 'rate': int(rate)}
                timings[cid] = {'separation_ms': int(sep.group(1)), 'wall_s': round(time.time() - t0, 1), 'stems': stems}
                json.dump(timings, open(timings_path, 'w'), indent=2)
                print(f'{key} {cid}: {int(sep.group(1)) / 1000:.1f} s, stems {list(stems)}', flush=True)
    finally:
        adb('shell', 'settings', 'put', 'global', 'stay_on_while_plugged_in', stay_on or '0', check=False)

if __name__ == '__main__':
    main()
