"""
Calibration experiment v2 (leave-one-machine-out), MIMII, channel 1.

Fixes over v1:
  * Calibration clips are split: the first K clips give the machine's feature statistics, the NEXT K clips (disjoint)
    give the alert level tau = mean + 3*std. (v1 used the same clips for both, which made tau too low.)
  * Two controls are added:
      S  simple detector with NO neural network: distance of a clip's features from the machine's own healthy
         statistics (mean squared z-score), same calibration clips, same alert-level rule.
      O  per-machine model ("oracle"): trained on the held-out machine's OWN healthy clips (standardised with its own
         statistics); this is the best case that needs per-machine training, which a real device cannot do.
  * Several random seeds for the neural models.

Methods on the held-out (new) machine h:
  M0 no calibration : pooled-statistics standardisation, alert level from the training machines
  M1 threshold only : same model as M0, alert level from h's own threshold clips
  M2 full calibration: per-machine standardisation from h's stats clips, alert level from h's threshold clips
  S  simple z-score control (no network)
  O  oracle per-machine model
All use the Hitachi evaluation split of h (first len(abnormal) healthy clips + all abnormal clips).
"""

import os
import csv
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import features as F
import baseline_ae as B

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'calibration_v2_results.csv')
KS = [10, 30, 100]
FIELDS = ['snr', 'machine', 'held_out_id', 'seed', 'method', 'K', 'auc', 'fpr', 'tpr', 'tau']
TRAIN_STRIDE = 4


def stack(lm, stride=1):
    v = B.stack_all(lm)
    return v[:, ::stride, :] if stride > 1 else v


def stats(vecs):
    X = vecs.reshape(-1, vecs.shape[-1])
    return X.mean(0), X.std(0) + 1e-6


def file_scores(model, vecs, mu, sd):
    model.eval()
    out = []
    with torch.no_grad():
        for v in vecs:
            x = torch.from_numpy(((v - mu) / sd).astype(np.float32))
            out.append(torch.mean((model(x) - x) ** 2).item())
    return np.array(out)


def simple_scores(vecs, mu, sd):
    return np.array([float((((v - mu) / sd) ** 2).mean()) for v in vecs])


def tau(s):
    return float(s.mean() + 3 * s.std())


def metrics(s_norm, s_abn, t):
    y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_abn))]
    s = np.r_[s_norm, s_abn]
    return float(roc_auc_score(y, s)), float((s_norm > t).mean()), float((s_abn > t).mean())


def run_fold(snr, mtype, held, epochs, seeds):
    ids = F.IDS
    train_ids = [i for i in ids if i != held]
    data = {i: (F.load_logmels(snr, mtype, i, 'normal'), F.load_logmels(snr, mtype, i, 'abnormal')) for i in ids}
    nh, ah = data[held]
    k = len(ah)
    ev_n, ev_a = stack(nh[:k]), stack(ah)
    pool = nh[k:]
    cal_end = 2 * max(KS)
    assert len(pool) > cal_end + 100, 'not enough healthy clips for this split'
    own_train = stack(pool[cal_end:], TRAIN_STRIDE)       # the oracle's own training clips
    tr_vec, thr_vec = {}, {}
    for i in train_ids:
        n, a = data[i]
        tr_vec[i] = stack(n[len(a):], TRAIN_STRIDE)
        thr_vec[i] = stack(n[:len(a)])
    base = dict(snr=snr, machine=mtype, held_out_id=held)
    rows = []

    # ---- S: simple control (no neural network), independent of the seed ----
    for K in KS:
        mu_h, sd_h = stats(stack(pool[:K]))
        t = tau(simple_scores(stack(pool[K:2 * K]), mu_h, sd_h))
        auc, fpr, tpr = metrics(simple_scores(ev_n, mu_h, sd_h), simple_scores(ev_a, mu_h, sd_h), t)
        rows.append(dict(base, seed=-1, method='S_simple_zscore', K=K, auc=auc, fpr=fpr, tpr=tpr, tau=t))

    allv = np.concatenate([tr_vec[i].reshape(-1, 320) for i in train_ids])
    mu_p, sd_p = stats(allv)
    allv_n = ((allv - mu_p) / sd_p).astype(np.float32)
    del allv
    parts = []
    for i in train_ids:
        mu_i, sd_i = stats(tr_vec[i])
        parts.append(((tr_vec[i].reshape(-1, 320) - mu_i) / sd_i).astype(np.float32))
    parts = np.concatenate(parts)
    mu_o, sd_o = stats(own_train)
    own_n = ((own_train.reshape(-1, 320) - mu_o) / sd_o).astype(np.float32)

    for seed in seeds:
        # ---- model P (pooled normalisation) ----
        model_p, _ = B.train_model(allv_n, epochs=epochs, seed=seed)
        tau_p = tau(np.concatenate([file_scores(model_p, thr_vec[i], mu_p, sd_p) for i in train_ids]))
        sn, sa = file_scores(model_p, ev_n, mu_p, sd_p), file_scores(model_p, ev_a, mu_p, sd_p)
        auc, fpr, tpr = metrics(sn, sa, tau_p)
        rows.append(dict(base, seed=seed, method='M0_no_calibration', K=0, auc=auc, fpr=fpr, tpr=tpr, tau=tau_p))
        for K in KS:
            t1 = tau(file_scores(model_p, stack(pool[K:2 * K]), mu_p, sd_p))
            auc, fpr, tpr = metrics(sn, sa, t1)
            rows.append(dict(base, seed=seed, method='M1_threshold_only', K=K, auc=auc, fpr=fpr, tpr=tpr, tau=t1))
        # ---- model C (per-machine normalisation) ----
        model_c, _ = B.train_model(parts, epochs=epochs, seed=seed)
        for K in KS:
            mu_h, sd_h = stats(stack(pool[:K]))
            t2 = tau(file_scores(model_c, stack(pool[K:2 * K]), mu_h, sd_h))
            auc, fpr, tpr = metrics(file_scores(model_c, ev_n, mu_h, sd_h), file_scores(model_c, ev_a, mu_h, sd_h), t2)
            rows.append(dict(base, seed=seed, method='M2_full_calibration', K=K, auc=auc, fpr=fpr, tpr=tpr, tau=t2))
        # ---- oracle: trained on the machine's own healthy clips ----
        model_o, _ = B.train_model(own_n, epochs=epochs, seed=seed)
        so_n, so_a = file_scores(model_o, ev_n, mu_o, sd_o), file_scores(model_o, ev_a, mu_o, sd_o)
        for K in KS:
            t3 = tau(file_scores(model_o, stack(pool[K:2 * K]), mu_o, sd_o))
            auc, fpr, tpr = metrics(so_n, so_a, t3)
            rows.append(dict(base, seed=seed, method='O_oracle_per_machine', K=K, auc=auc, fpr=fpr, tpr=tpr, tau=t3))
    return rows


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r['snr'], r['machine'], r['held_out_id'], r['seed']) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snr', default='6')
    ap.add_argument('--types', nargs='*', default=F.TYPES)
    ap.add_argument('--held', nargs='*', default=F.IDS)
    ap.add_argument('--seeds', nargs='*', type=int, default=[0])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    done = done_keys(a.out)
    with open(a.out, 'a', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for t in a.types:
            for h in a.held:
                todo = [s for s in a.seeds if (a.snr, t, h, str(s)) not in done]
                if not todo:
                    continue
                t0 = time.time()
                for r in run_fold(a.snr, t, h, a.epochs, todo):
                    w.writerow(r)
                f.flush()
                print(f'{a.snr}dB {t} held-out {h} seeds {todo} done in {time.time() - t0:.0f}s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
