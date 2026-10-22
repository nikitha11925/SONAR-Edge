"""Extract log-mel features with a different number of mel bands (for the band-count test).
   python extract_bands.py --n_mels 32 --snrs 6 --workers 6
   Output: N:\\dev\\mimii_features_b32\\<snr>_<type>_<id>_<label>.npy   (clips, n_mels, 313)"""
import os
import time
import argparse
import numpy as np
from multiprocessing import Pool
import features as F


def job(args):
    snr, mtype, mid, label, n_mels, out_dir = args
    out = os.path.join(out_dir, f'{snr}_{mtype}_{mid}_{label}.npy')
    if os.path.exists(out):
        return args, 'skip', 0
    t0 = time.time()
    files = F.list_wavs(snr, mtype, mid, label)
    arr = np.stack([F.logmel(F.load_channel0(p), n_mels=n_mels) for p in files])
    tmp = out + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, out)
    return args, str(arr.shape), time.time() - t0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_mels', type=int, required=True)
    ap.add_argument('--snrs', nargs='*', default=['6'])
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    out_dir = rf'N:\dev\mimii_features_b{a.n_mels}'
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, t, i, l, a.n_mels, out_dir) for s in a.snrs for t in F.TYPES for i in F.IDS for l in ('normal', 'abnormal')]
    with Pool(a.workers) as p:
        for k, (arg, msg, dt) in enumerate(p.imap_unordered(job, jobs), 1):
            print(f'[{k}/{len(jobs)}] {arg[:4]} {msg} {dt:.0f}s', flush=True)
    print('DONE', flush=True)
