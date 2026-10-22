"""
Noise-mismatch experiment: what happens when the noise at test time differs from the noise at calibration time?

For each machine type and each held-out machine h (leave-one-ID-out; the model never sees h):
  * model A  : trained only on 6 dB healthy sound of the other three machines
  * model B  : trained on healthy sound of the other three machines at ALL three noise levels (6, 0, -6 dB)
  (each training machine/noise condition is standardised with its own statistics, as calibration does on a device)
  * calibration condition c in {6, 0, -6, mix}:
        stats  : K = 30 healthy clips of h at noise level c   ('mix' = 10 clips from each of the three levels)
        alert  : mean + 3*std of the scores of the NEXT K healthy clips (disjoint clips)
  * test condition t in {6, 0, -6}: Hitachi's evaluation split of h at noise level t
    (the first len(abnormal) healthy clips + all abnormal clips; disjoint from all calibration clips)
Design per type = the stage-2 winner's structure with 64 mel bands (features for 0 and -6 dB exist only at 64 bands).
Metrics: AUC, false-alarm rate, fault-detection rate at the device's own alert level.
"""
import os
import csv
import time
import argparse
import numpy as np
import torch
import features as F
import model_selection as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'noise_mismatch_results.csv')
SNRS = ['6', '0', '-6']
K = 30
FIELDS = ['machine', 'held_out', 'model', 'calib', 'test', 'auc', 'fpr', 'tpr']


def load(snr, mtype, mid, label):
    return np.load(F.feature_path(snr, mtype, mid, label), mmap_mode='r')


def design_for(mtype):
    rows = {r['machine']: r for r in csv.DictReader(open(os.path.join(HERE, 'results', 'model_selection_choice_final.csv')))}
    r = rows[mtype]
    return (r['design'], int(r['frames']), r['depth'], int(r['hidden']), int(r['bottleneck']))


def run_fold(mtype, held, epochs, seed):
    name, frames, depth, hidden, bott = design_for(mtype)
    train_ids = [i for i in F.IDS if i != held]
    # ---- training sets (per machine and noise level standardised with its own statistics) ----
    parts = {s: [] for s in SNRS}
    for s in SNRS:
        for i in train_ids:
            n, a = load(s, mtype, i, 'normal'), load(s, mtype, i, 'abnormal')
            v = M.stack(n[len(a):], frames, M.TRAIN_STRIDE)
            mu, sd = M.stats(v)
            parts[s].append(((v.reshape(-1, v.shape[-1]) - mu) / sd).astype(np.float32))
    X_a = np.concatenate(parts['6'])
    X_b = np.concatenate([p for s in SNRS for p in parts[s]])
    dim = X_a.shape[1]
    model_a = M.train(M.build(dim, depth, hidden, bott), X_a, epochs, seed)
    model_b = M.train(M.build(dim, depth, hidden, bott), X_b, epochs, seed)
    del parts, X_a, X_b
    # ---- held-out machine at the three noise levels ----
    nh = {s: load(s, mtype, held, 'normal') for s in SNRS}
    ah = {s: load(s, mtype, held, 'abnormal') for s in SNRS}
    k = len(ah['6'])
    ev = {s: (M.stack(nh[s][:k], frames), M.stack(ah[s], frames)) for s in SNRS}
    pool = {s: nh[s][k:] for s in SNRS}
    calib = {}
    for c in SNRS:
        calib[c] = (M.stack(pool[c][:K], frames), M.stack(pool[c][K:2 * K], frames))
    calib['mix'] = (np.concatenate([M.stack(pool[s][j * 10:(j + 1) * 10], frames) for j, s in enumerate(SNRS)]),
                    np.concatenate([M.stack(pool[s][K + j * 10:K + (j + 1) * 10], frames) for j, s in enumerate(SNRS)]))
    rows = []
    for mname, model in (('trained@6dB', model_a), ('trained@all', model_b)):
        for c, (stat_v, thr_v) in calib.items():
            mu, sd = M.stats(stat_v)
            alert = M.tau(M.file_scores(model, thr_v, mu, sd))
            for t in SNRS:
                sn = M.file_scores(model, ev[t][0], mu, sd)
                sa = M.file_scores(model, ev[t][1], mu, sd)
                auc, fpr, tpr = M.metrics(sn, sa, alert)
                rows.append(dict(machine=mtype, held_out=held, model=mname, calib=c, test=t, auc=auc, fpr=fpr, tpr=tpr))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', nargs='*', default=F.TYPES)
    ap.add_argument('--held', nargs='*', default=F.IDS)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    done = set()
    if os.path.exists(a.out):
        done = {(r['machine'], r['held_out']) for r in csv.DictReader(open(a.out))}
    new = not os.path.exists(a.out)
    with open(a.out, 'a', newline='') as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for t in a.types:
            for h in a.held:
                if (t, h) in done:
                    continue
                t0 = time.time()
                w.writerows(run_fold(t, h, a.epochs, a.seed))
                fh.flush()
                print(f'{t} held-out {h} done in {time.time() - t0:.0f}s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
