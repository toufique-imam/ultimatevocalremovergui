"""Cut the comparison clips: LENGTH seconds of each chosen song from START, decoded once to
44.1 kHz stereo float. Desktop UVR gets clipNN.wav and the phone clipNN.f32 — the same samples,
so any difference in the results comes from the models, not from decoding.

    python prepare_clips.py <songs dir> <work dir> [--start 60] [--length 30]
"""
import argparse
import json
import os
import subprocess

import numpy as np
import soundfile as sf

# A spread of styles; each entry must match exactly one file name in the songs dir
SONGS = [
    'Queen – Bohemian Rhapsody',
    'Nirvana - Smells Like Teen Spirit',
    'Whitney Houston - I Will Always Love You',
    'Michael Bublé - Feeling Good',
    'Eminem - Stan',
    'The Black Eyed Peas - Where Is The Love',
    'Gotye - Somebody That I Used To Know',
    'Vanessa Carlton - A Thousand Miles',
    'Ed Sheeran - The A Team',
    'The Beatles - Yesterday',
    'Muse - Supermassive Black Hole',
    'The Dandy Warhols - Bohemian Like You',
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('songs')
    parser.add_argument('work')
    parser.add_argument('--start', type=float, default=60)
    parser.add_argument('--length', type=float, default=30)
    args = parser.parse_args()

    clips_dir = os.path.join(args.work, 'clips')
    os.makedirs(clips_dir, exist_ok=True)
    files = sorted(os.listdir(args.songs))
    manifest = []

    for i, song in enumerate(SONGS, start=1):
        matches = [f for f in files if song in f]
        if len(matches) != 1:
            raise SystemExit(f'"{song}" matches {len(matches)} files: {matches}')
        clip_id = f'clip{i:02d}'
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(args.start), '-t', str(args.length),
                              '-i', os.path.join(args.songs, matches[0]), '-map', '0:a:0', '-vn',
                              '-ac', '2', '-ar', '44100', '-f', 'f32le', '-'],
                             capture_output=True, check=True).stdout
        audio = np.frombuffer(raw, np.float32).reshape(-1, 2)
        audio.tofile(os.path.join(clips_dir, f'{clip_id}.f32'))
        sf.write(os.path.join(clips_dir, f'{clip_id}.wav'), audio, 44100, subtype='FLOAT')
        manifest.append({'id': clip_id, 'song': matches[0], 'start': args.start, 'seconds': len(audio) / 44100})
        print(f'{clip_id}: {len(audio) / 44100:.1f} s from {matches[0]}')

    with open(os.path.join(args.work, 'clips.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
