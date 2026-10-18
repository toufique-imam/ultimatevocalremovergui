"""Writes a playable 24-bit WAV next to every raw phone stem (.f32: headerless interleaved stereo
float, as the app's test runner saves it) under the given folders, except the clips/ and groups/
inputs. The sample rate comes from the folder's timings.json / result.json when present (VR's
32 kHz models), else 44.1 kHz. Existing WAVs are kept.

    python to_wav.py <folder> [folder ...]      # e.g. ~/Desktop/test_uvr/output/compare_full
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

def rates(folder):
    """{f32 file name: sample rate} from the runners' timings.json / result.json."""
    found = {}
    for name in ('timings.json', 'result.json'):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        runs = data.values() if name == 'timings.json' else [data]
        for run in runs:
            for stem in run.get('stems', {}).values():
                found[stem['file']] = stem['rate']
    return found

def main():
    for root in sys.argv[1:]:
        for path in sorted(glob.glob(os.path.join(os.path.expanduser(root), '**', '*.f32'), recursive=True)):
            wav = path[:-4] + '.wav'
            if os.path.exists(wav) or any(f'{os.sep}{d}{os.sep}' in path for d in ('clips', 'groups')):   # inputs
                continue
            rate = rates(os.path.dirname(path)).get(os.path.basename(path), 44100)
            audio = np.fromfile(path, np.float32).reshape(-1, 2)
            sf.write(wav, np.clip(audio, -1, 1), rate, subtype='PCM_24')
            print(f'{wav}  ({len(audio) / rate:.1f} s)')

if __name__ == '__main__':
    main()
