"""
FINAL EVALUATION - the headline results, with every metric.

For each machine type, each noise level (6, 0, -6 dB), each of the 4 machines as the "new machine" (leave-one-out):
  * design = the type's chosen design from model selection (results/model_selection_choice_final.csv)
  * the model trains on the other 3 machines' healthy sound at the SAME noise level
    (each machine standardised with its own statistics)
  * calibration on the new machine at that noise level: feature numbers from 30 healthy clips,
    alert level = mean + 3*std of the next 30 healthy clips (disjoint)
  * test: Hitachi split (N healthy + N faulty clips), disjoint from all calibration clips
  * 3 random seeds
Metrics per case: AUC, precision, recall, F1, accuracy, false-alarm rate, balanced accuracy, confusion counts.
NOTE: model choices were made on machines 00, 02, 04 at 6 dB; machine 06 was sealed. Report both subsets.
"""
import os
import csv
import time
import argparse
import numpy as np
import torch
import features as F
import model_selection as M
import evaluation_lib as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'final_evaluation_results.csv')
SNRS = ['6', '0', '-6']
K = 30
FIELDS = ['machine', 'snr', 'held_out', 'seed', 'design', 'mels', 'params', 'auc', 'precision', 'recall', 'f1', 'accuracy',
          'fpr', 'specificity', 'balanced_accuracy', 'tp', 'fp', 'tn', 'fn']


def choice():
    rows = {r['machine']: r for r in csv.DictReader(open(os.path.join(HERE, 'results', 'model_selection_choice_final.csv')))}
    return rows


def root_for(mels):
    return M.FEATURE_ROOT_64 if mels == 64 else rf'N:\dev\mimii_features_b{mels}'


def wait_for_features(root, snr, mtype, timeout_min=90):
    need = [os.path.join(root, f'{snr}_{mtype}_{i}_{l}.npy') for i in F.IDS for l in ('normal', 'abnormal')]
    t0 = time.time()
    while not all(os.path.exists(p) for p in need):
        if time.time() - t0 > timeout_min * 60:
            raise SystemExit(f'features missing: {[p for p in need if not os.path.exists(p)][:3]}')
        time.sleep(30)


def run_case(mtype, snr, held, seeds, epochs, ch):
    name, frames, depth, hidden, bott = ch['design'], int(ch['frames']), ch['depth'], int(ch['hidden']), int(ch['bottleneck'])
    mels = int(ch['mels'])
    root = root_for(mels)
    wait_for_features(root, snr, mtype)
    F.FEATURE_ROOT = root
    load = lambda i, lab: np.load(F.feature_path(snr, mtype, i, lab), mmap_mode='r')
    train_ids = [i for i in F.IDS if i != held]
    parts = []
    for i in train_ids:
        n, a = load(i, 'normal'), load(i, 'abnormal')
        v = M.stack(n[len(a):], frames, M.TRAIN_STRIDE)
        mu, sd = M.stats(v)
        parts.append(((v.reshape(-1, v.shape[-1]) - mu) / sd).astype(np.float32))
    X = np.concatenate(parts)
    del parts
    nh, ah = load(held, 'normal'), load(held, 'abnormal')
    k = len(ah)
    ev_n, ev_a = M.stack(nh[:k], frames), M.stack(ah, frames)
    pool = nh[k:]
    mu_h, sd_h = M.stats(M.stack(pool[:K], frames))
    thr = M.stack(pool[K:2 * K], frames)
    rows = []
    for seed in seeds:
        model = M.build(X.shape[1], depth, hidden, bott)
        M.train(model, X, epochs, seed)
        alert = M.tau(M.file_scores(model, thr, mu_h, sd_h))
        met = E.full_metrics(M.file_scores(model, ev_n, mu_h, sd_h), M.file_scores(model, ev_a, mu_h, sd_h), alert)
        rows.append(dict(machine=mtype, snr=snr, held_out=held, seed=seed, design=name, mels=mels, params=M.n_params(model), **met))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', nargs='*', default=F.TYPES)
    ap.add_argument('--snrs', nargs='*', default=SNRS)
    ap.add_argument('--held', nargs='*', default=F.IDS)
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ch_all = choice()
    done = set()
    if os.path.exists(a.out):
        done = {(r['machine'], r['snr'], r['held_out']) for r in csv.DictReader(open(a.out))}
    new = not os.path.exists(a.out)
    with open(a.out, 'a', newline='') as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for snr in a.snrs:
            for t in a.types:
                for h in a.held:
                    if (t, snr, h) in done:
                        continue
                    t0 = time.time()
                    w.writerows(run_case(t, snr, h, a.seeds, a.epochs, ch_all[t]))
                    fh.flush()
                    print(f'{snr}dB {t} held-out {h} done in {time.time() - t0:.0f}s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
