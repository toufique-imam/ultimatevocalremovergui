"""Benchmark input from the MUSDB18 7-second test previews: 5 files of 10 songs each, joined
with 3 s of silence so no song's separation hears its neighbour. group1 is the phone's set.
Writes <work>/groups/groupN.{wav,f32}, the true stems per song in <work>/refs/, and groups.json.

    python prepare_musdb.py <musdb root> <work dir>
"""
import json
import os
import sys

import musdb
import numpy as np
import soundfile as sf

GAP_SECONDS = 3
PER_GROUP = 10
RATE = 44100

def main():
    root, work = sys.argv[1], sys.argv[2]
    db = musdb.DB(root=root, subsets='test')
    for sub in ('groups', 'refs'):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    gap = np.zeros((GAP_SECONDS * RATE, 2), np.float32)
    groups = []

    for g in range(len(db) // PER_GROUP):
        parts, songs, pos = [], [], 0
        for i, track in enumerate(db[g * PER_GROUP:(g + 1) * PER_GROUP]):
            assert track.rate == RATE
            song_id = f'g{g + 1}s{i + 1:02d}'
            targets = {name: track.targets[name].audio.astype(np.float32) for name in ('vocals', 'drums', 'bass', 'other')}
            targets['instrumental'] = targets['drums'] + targets['bass'] + targets['other']
            mix = track.audio.astype(np.float32)
            for name, audio in targets.items():
                audio.tofile(os.path.join(work, 'refs', f'{song_id}_{name}.f32'))
            if parts:
                parts.append(gap)
                pos += len(gap)
            parts.append(mix)
            songs.append({'id': song_id, 'name': track.name, 'offset': pos, 'length': len(mix)})
            pos += len(mix)
        audio = np.concatenate(parts)
        name = f'group{g + 1}'
        audio.tofile(os.path.join(work, 'groups', f'{name}.f32'))
        sf.write(os.path.join(work, 'groups', f'{name}.wav'), audio, RATE, subtype='FLOAT')
        groups.append({'id': name, 'seconds': len(audio) / RATE, 'songs': songs})
        print(f'{name}: {len(songs)} songs, {len(audio) / RATE:.1f} s')

    json.dump(groups, open(os.path.join(work, 'groups.json'), 'w'), indent=2)

if __name__ == '__main__':
    main()
