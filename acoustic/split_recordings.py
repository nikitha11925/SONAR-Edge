"""
Cut a long recording (WAV / MP3 / M4A, also the .m4a.mpeg WhatsApp names) into 10-second WAV clips (16 kHz, mono, 16-bit).

    python split_recordings.py --src "<file>" --name healthy_session1
writes  own_data/clips/healthy_session1/healthy_session1_001.wav, _002.wav, ...   (the last piece shorter than 10 s is dropped)
Clips are numbered in time order, which matters: the dashboard calibrates on the FIRST clips of the healthy folder.
"""
import os
import argparse
import numpy as np
import soundfile as sf
import own_recording_test as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(HERE), 'own_data', 'clips')


def split(src, name, out_root=OUT_ROOT):
    y, sr = T.read_audio(src)
    clips = T.cut_clips(y, sr)
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    for i, c in enumerate(clips, 1):
        sf.write(os.path.join(d, f'{name}_{i:03d}.wav'), np.clip(c, -1.0, 1.0), 16000, subtype='PCM_16')
    return d, len(clips), len(y) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    d, n, secs = split(a.src, a.name)
    print(f'{os.path.basename(a.src)}: {secs:.1f} s -> {n} clips of 10 s in {d}')


if __name__ == '__main__':
    main()
