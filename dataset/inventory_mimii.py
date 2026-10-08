"""
MIMII dataset inventory.
Counts WAV files per (SNR, machine type, model ID, normal/abnormal) and checks
every file header (sample rate, channels, bit depth, duration).
Layout expected: <root>/<snr>_dB_<type>/<type>/id_XX/{normal,abnormal}/*.wav
"""

import os
import csv
import wave
import argparse
from collections import Counter, defaultdict


def scan(root):
    rows = []
    for folder in sorted(os.listdir(root)):
        parts = folder.split('_dB_')
        full = os.path.join(root, folder)
        if len(parts) != 2 or not os.path.isdir(full):
            continue
        snr, mtype = parts[0], parts[1]
        base = os.path.join(full, mtype)
        if not os.path.isdir(base):
            continue
        for mid in sorted(os.listdir(base)):
            for label in ('normal', 'abnormal'):
                d = os.path.join(base, mid, label)
                if not os.path.isdir(d):
                    continue
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith('.wav'):
                        rows.append((snr, mtype, mid, label, os.path.join(d, f)))
    return rows


def header(path):
    try:
        with wave.open(path, 'rb') as w:
            return (w.getframerate(), w.getnchannels(), w.getsampwidth() * 8,
                    round(w.getnframes() / w.getframerate(), 2))
    except Exception as e:
        return ('ERROR', str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'N:\dev\Sonar-acoustic-dataset')
    ap.add_argument('--out_csv', default='dataset/mimii_inventory.csv')
    args = ap.parse_args()

    rows = scan(args.root)
    print(f"[INFO] {len(rows)} WAV files found under {args.root}")

    counts = Counter((s, t, m, l) for s, t, m, l, _ in rows)
    headers = Counter()
    bad = []
    for s, t, m, l, p in rows:
        h = header(p)
        headers[h] += 1
        if h[0] == 'ERROR':
            bad.append((p, h[1]))

    print("\n[HEADER FORMATS] (rate, channels, bits, seconds) -> files")
    for h, n in headers.most_common():
        print(f"  {h}: {n}")
    print(f"[CORRUPT/UNREADABLE] {len(bad)}")
    for p, e in bad[:10]:
        print("  ", p, e)

    print("\n[COUNTS] normal / abnormal per type and ID (per noise level)")
    table = defaultdict(dict)
    for (s, t, m, l), n in counts.items():
        table[(t, m)].setdefault(s, {})[l] = n
    for (t, m) in sorted(table):
        cells = []
        for s in sorted(table[(t, m)], key=int):
            c = table[(t, m)][s]
            cells.append(f"{s}dB {c.get('normal', 0)}/{c.get('abnormal', 0)}")
        print(f"  {t:7s} {m}: " + " | ".join(cells))

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr_db', 'machine', 'model_id', 'label', 'path'])
        w.writerows(rows)
    print(f"\n[SUCCESS] File list written to {args.out_csv}")


if __name__ == '__main__':
    main()
