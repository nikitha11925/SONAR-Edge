"""
Calibration experiment (leave-one-ID-out), MIMII, channel 1.

For each machine type and each held-out ID h:
  * the model is trained on the OTHER three IDs only (healthy sound only);
  * h plays the role of a NEW machine that arrives at a factory;
  * calibration uses K healthy clips of h that are NOT in h's evaluation set;
  * evaluation set of h = Hitachi's split (first len(abnormal) normal clips + all abnormal clips).

Methods compared on the held-out machine:
  M0  no calibration       : features normalised with pooled training statistics,
                              threshold from the training machines' healthy errors
  M1  threshold only        : same model as M0, threshold tau = mean + 3*std of errors on K calibration clips
  M2  full calibration      : model trained with per-machine feature normalisation; the new machine's own
                              K calibration clips give BOTH its feature statistics and tau = mean + 3*std

AUC is unchanged by a pure threshold change (M0 vs M1), so we also report the false-alarm rate (FPR) and the
detection rate (TPR) at the device's own threshold.
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'calibration_results.csv')
KS = [10, 30, 100]
FIELDS = ['snr', 'machine', 'held_out_id', 'method', 'K', 'auc', 'fpr', 'tpr', 'tau', 'n_eval_normal', 'n_eval_abnormal']
TRAIN_STRIDE = 4          # use every 4th input vector for training (neighbouring vectors overlap heavily)


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


def tau(scores):
    return float(scores.mean() + 3 * scores.std())


def metrics(s_norm, s_abn, t):
    y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_abn))]
    s = np.r_[s_norm, s_abn]
    return float(roc_auc_score(y, s)), float((s_norm > t).mean()), float((s_abn > t).mean())


def run_fold(snr, mtype, held, epochs, seed=0):
    ids = F.IDS
    train_ids = [i for i in ids if i != held]
    data = {i: (F.load_logmels(snr, mtype, i, 'normal'), F.load_logmels(snr, mtype, i, 'abnormal')) for i in ids}
    nh, ah = data[held]
    k = len(ah)
    eval_norm_lm, eval_abn_lm, calib_pool = nh[:k], ah, nh[k:]
    tr_vec, thr_vec = {}, {}
    for i in train_ids:
        n, a = data[i]
        tr_vec[i] = stack(n[len(a):], TRAIN_STRIDE)       # training clips of a training machine
        thr_vec[i] = stack(n[:len(a)])                     # its held-back healthy clips -> threshold of the uncalibrated model
    ev_n, ev_a = stack(eval_norm_lm), stack(eval_abn_lm)
    rows = []
    base = dict(snr=snr, machine=mtype, held_out_id=held, n_eval_normal=len(ev_n), n_eval_abnormal=len(ev_a))

    # ---- model P: pooled normalisation (no calibration) ----
    allv = np.concatenate([tr_vec[i].reshape(-1, 320) for i in train_ids])
    mu_p, sd_p = stats(allv)
    model_p, _ = B.train_model(((allv - mu_p) / sd_p).astype(np.float32), epochs=epochs, seed=seed)
    del allv
    tau_p = tau(np.concatenate([file_scores(model_p, thr_vec[i], mu_p, sd_p) for i in train_ids]))
    sn, sa = file_scores(model_p, ev_n, mu_p, sd_p), file_scores(model_p, ev_a, mu_p, sd_p)
    auc, fpr, tpr = metrics(sn, sa, tau_p)
    rows.append(dict(base, method='M0_no_calibration', K=0, auc=auc, fpr=fpr, tpr=tpr, tau=tau_p))

    # ---- model C: per-machine normalisation ----
    parts = []
    for i in train_ids:
        mu_i, sd_i = stats(tr_vec[i])
        parts.append(((tr_vec[i].reshape(-1, 320) - mu_i) / sd_i).astype(np.float32))
    model_c, _ = B.train_model(np.concatenate(parts), epochs=epochs, seed=seed)
    del parts

    for K in KS:
        cal = stack(calib_pool[:K])
        # M1: same model, threshold from the new machine's calibration clips only
        t1 = tau(file_scores(model_p, cal, mu_p, sd_p))
        auc, fpr, tpr = metrics(sn, sa, t1)
        rows.append(dict(base, method='M1_threshold_only', K=K, auc=auc, fpr=fpr, tpr=tpr, tau=t1))
        # M2: new machine's own feature statistics + threshold
        mu_h, sd_h = stats(cal)
        t2 = tau(file_scores(model_c, cal, mu_h, sd_h))
        sn2, sa2 = file_scores(model_c, ev_n, mu_h, sd_h), file_scores(model_c, ev_a, mu_h, sd_h)
        auc, fpr, tpr = metrics(sn2, sa2, t2)
        rows.append(dict(base, method='M2_full_calibration', K=K, auc=auc, fpr=fpr, tpr=tpr, tau=t2))
    return rows


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r['snr'], r['machine'], r['held_out_id']) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snr', default='6')
    ap.add_argument('--types', nargs='*', default=F.TYPES)
    ap.add_argument('--held', nargs='*', default=F.IDS)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    done = done_keys(out)
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for t in args.types:
            for h in args.held:
                if (args.snr, t, h) in done:
                    continue
                t0 = time.time()
                for r in run_fold(args.snr, t, h, args.epochs):
                    w.writerow(r)
                f.flush()
                print(f'{args.snr}dB {t} held-out {h} done in {time.time() - t0:.0f}s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
