"""
Model selection under the calibration setup.

Protocol (fixed in advance):
  * Per machine type, IDs 00, 02 and 04 are the TUNING machines; ID 06 is SEALED (never used to choose anything).
  * 'tune' stage: each tuning machine is held out in turn; the model trains on the OTHER TWO tuning machines
    (each standardised with its own statistics) and is evaluated as a new machine with full calibration:
    feature statistics from K clips, alert level (mean + 3 std) from the next K clips (K = 30).
  * Selection rule (per machine type): the SMALLEST model whose mean tuning AUC is within 0.02 of the best.
  * 'final' stage: train on all three tuning machines, calibrate on the sealed machine's healthy clips, evaluate once.
Designs vary: frames stacked (1/3/5), network depth/width, bottleneck (8/16); mel bands (32/64/128) in the 'bands' stage.
"""

import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
OUT_TUNE = os.path.join(RES, 'model_selection_tune.csv')
OUT_FINAL = os.path.join(RES, 'model_selection_final.csv')
CHOICE = os.path.join(RES, 'model_selection_choice.csv')
FEATURE_ROOT_64 = F.FEATURE_ROOT
K = 30
TRAIN_STRIDE = 4
TUNE_IDS = ['id_00', 'id_02', 'id_04']
SEALED = 'id_06'

# name, frames, depth, hidden, bottleneck
DESIGNS = [
    ('D_F5_H64_b8', 5, 'deep', 64, 8),      # = Hitachi baseline architecture
    ('D_F5_H64_b16', 5, 'deep', 64, 16),
    ('D_F5_H32_b8', 5, 'deep', 32, 8),
    ('D_F3_H64_b8', 3, 'deep', 64, 8),
    ('D_F3_H32_b8', 3, 'deep', 32, 8),
    ('D_F3_H32_b16', 3, 'deep', 32, 16),
    ('D_F1_H64_b8', 1, 'deep', 64, 8),
    ('D_F1_H32_b8', 1, 'deep', 32, 8),
    ('S_F5_b16', 5, 'shallow', 32, 16),     # deck-style 32-16-32
    ('S_F3_b16', 3, 'shallow', 32, 16),
    ('S_F1_b16', 1, 'shallow', 32, 16),
    ('T_F1_b8', 1, 'tiny', 16, 8),
]
FIELDS = ['stage', 'snr', 'machine', 'design', 'mels', 'frames', 'depth', 'hidden', 'bottleneck', 'params',
          'held_out', 'seed', 'auc', 'fpr', 'tpr']


def build(in_dim, depth, hidden, bott):
    if depth == 'deep':
        sizes = [in_dim, hidden, hidden, bott, hidden, hidden, in_dim]
    elif depth == 'shallow':
        sizes = [in_dim, 32, bott, 32, in_dim]
    else:
        sizes = [in_dim, 16, bott, 16, in_dim]
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, X, epochs, seed, batch=512):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_val = int(0.1 * len(X))
    Xtr, Xva = torch.from_numpy(X[:len(X) - n_val]), torch.from_numpy(X[len(X) - n_val:])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(Xtr)))
        for i in range(0, len(Xtr), batch):
            xb = Xtr[perm[i:i + batch]]
            opt.zero_grad()
            loss_fn(model(xb), xb).backward()
            opt.step()
    return model


def stack(lms, frames, stride=1):
    v = np.stack([F.stack_frames(lm, frames=frames) for lm in lms])
    return v[:, ::stride, :] if stride > 1 else v


def stats(vecs):
    X = vecs.reshape(-1, vecs.shape[-1])
    return X.mean(0), X.std(0) + 1e-6


def file_scores(model, vecs, mu, sd, chunk=32):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(vecs), chunk):
            v = vecs[i:i + chunk]
            nf, nv, d = v.shape
            x = torch.from_numpy(((v - mu) / sd).astype(np.float32)).reshape(-1, d)
            out.extend(((model(x) - x) ** 2).mean(1).reshape(nf, nv).mean(1).tolist())
    return np.array(out)


def tau(s):
    return float(s.mean() + 3 * s.std())


def metrics(sn, sa, t):
    y = np.r_[np.zeros(len(sn)), np.ones(len(sa))]
    return float(roc_auc_score(y, np.r_[sn, sa])), float((sn > t).mean()), float((sa > t).mean())


def design_by_name(name):
    for d in DESIGNS:
        if d[0] == name:
            return d
    raise KeyError(name)


def run_one(model, held_data, norm_train, train_ids, frames, epochs, seed, mu_h, sd_h, thr_clips):
    X = np.concatenate([norm_train[i] for i in train_ids])
    train(model, X, epochs, seed)
    t = tau(file_scores(model, thr_clips, mu_h, sd_h))
    return metrics(file_scores(model, held_data[0], mu_h, sd_h), file_scores(model, held_data[1], mu_h, sd_h), t)


def run_stage(stage, snr, types, designs, epochs, seeds, feature_root, mels, writer, fh, train_ids_pool, held_list, label=''):
    F.FEATURE_ROOT = feature_root
    for mtype in types:
        data = {i: (F.load_logmels(snr, mtype, i, 'normal'), F.load_logmels(snr, mtype, i, 'abnormal')) for i in F.IDS}
        for frames in sorted({d[1] for d in designs}):
            norm_train = {}
            for i in train_ids_pool:
                n, a = data[i]
                v = stack(n[len(a):], frames, TRAIN_STRIDE)
                mu, sd = stats(v)
                norm_train[i] = ((v.reshape(-1, v.shape[-1]) - mu) / sd).astype(np.float32)
            for held in held_list:
                nh, ah = data[held]
                k = len(ah)
                ev = (stack(nh[:k], frames), stack(ah, frames))
                pool = nh[k:]
                mu_h, sd_h = stats(stack(pool[:K], frames))
                thr = stack(pool[K:2 * K], frames)
                train_ids = [i for i in train_ids_pool if i != held]
                for name, fr, depth, hidden, bott in [d for d in designs if d[1] == frames]:
                    for seed in seeds:
                        t0 = time.time()
                        model = build(ev[0].shape[-1], depth, hidden, bott)
                        auc, fpr, tpr = run_one(model, ev, norm_train, train_ids, frames, epochs, seed, mu_h, sd_h, thr)
                        writer.writerow(dict(stage=stage, snr=snr, machine=mtype, design=name + label, mels=mels, frames=frames,
                                             depth=depth, hidden=hidden, bottleneck=bott, params=n_params(model),
                                             held_out=held, seed=seed, auc=auc, fpr=fpr, tpr=tpr))
                        fh.flush()
                print(f'[{stage}] {snr}dB {mtype} frames={frames} held-out {held} done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['tune', 'bands', 'final'], required=True)
    ap.add_argument('--snr', default='6')
    ap.add_argument('--types', nargs='*', default=F.TYPES)
    ap.add_argument('--designs', nargs='*', default=None)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seeds', nargs='*', type=int, default=[0])
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--out', default=None)
    ap.add_argument('--mels', type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    designs = DESIGNS if a.designs is None else [design_by_name(n) for n in a.designs]
    def feature_root(mels):
        return FEATURE_ROOT_64 if mels == 64 else rf'N:\dev\mimii_features_b{mels}'

    def read_choice(path):
        return {r['machine']: r for r in csv.DictReader(open(path))}

    def tuple_of(r):
        return (r['design'], int(r['frames']), r['depth'], int(r['hidden']), int(r['bottleneck']))

    if a.stage == 'tune':
        out = a.out or OUT_TUNE
        new = not os.path.exists(out)
        with open(out, 'a', newline='') as fh:
            w = csv.DictWriter(fh, FIELDS)
            if new:
                w.writeheader()
            run_stage('tune', a.snr, a.types, designs, a.epochs, a.seeds, FEATURE_ROOT_64, 64, w, fh, TUNE_IDS, TUNE_IDS)
    elif a.stage == 'bands':
        # the stage-1 winner of each type, rebuilt with 32 and 128 mel bands (features must be extracted first)
        choice = read_choice(CHOICE)
        out = a.out or OUT_TUNE
        with open(out, 'a', newline='') as fh:
            w = csv.DictWriter(fh, FIELDS)
            for mt in a.types:
                for mels in (32, 128):
                    run_stage('tune', a.snr, [mt], [tuple_of(choice[mt])], a.epochs, a.seeds, feature_root(mels), mels, w, fh,
                              TUNE_IDS, TUNE_IDS, label=f'_m{mels}')
    elif a.stage == 'final':
        # sealed machine: train on all three tuning machines, calibrate on the sealed machine, evaluate once
        choice = read_choice(os.path.join(RES, 'model_selection_choice_final.csv'))
        base = DESIGNS[0]
        out = a.out or OUT_FINAL
        new = not os.path.exists(out)
        with open(out, 'a', newline='') as fh:
            w = csv.DictWriter(fh, FIELDS)
            if new:
                w.writeheader()
            for mt in a.types:
                c = choice[mt]
                mels_c = int(c['mels'])
                run_stage('final', a.snr, [mt], [base], a.epochs, a.seeds, feature_root(64), 64, w, fh, TUNE_IDS, [SEALED], label='_baseline')
                if tuple_of(c) != base or mels_c != 64:
                    run_stage('final', a.snr, [mt], [tuple_of(c)], a.epochs, a.seeds, feature_root(mels_c), mels_c, w, fh,
                              TUNE_IDS, [SEALED], label=f'_chosen_m{mels_c}')
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
