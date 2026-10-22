"""Extract channel-1 log-mel for every MIMII file (parallel, resumable)."""
import os
import time
import argparse
import numpy as np
from multiprocessing import Pool
import features as F


def job(args):
    snr, mtype, mid, label = args
    out = F.feature_path(snr, mtype, mid, label)
    if os.path.exists(out):
        return args, 'skip', 0
    t0 = time.time()
    files = F.list_wavs(snr, mtype, mid, label)
    arr = np.stack([F.logmel(F.load_channel0(p)) for p in files])
    tmp = out + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, out)
    return args, f'{arr.shape}', time.time() - t0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(F.FEATURE_ROOT, exist_ok=True)
    jobs = [(s, t, i, l) for s in F.SNRS for t in F.TYPES for i in F.IDS for l in ('normal', 'abnormal')]
    print(f'{len(jobs)} jobs', flush=True)
    with Pool(args.workers) as p:
        for k, (a, msg, dt) in enumerate(p.imap_unordered(job, jobs), 1):
            print(f'[{k}/{len(jobs)}] {a} {msg} {dt:.0f}s', flush=True)
    print('DONE', flush=True)
