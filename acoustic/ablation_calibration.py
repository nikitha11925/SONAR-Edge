"""
CALIBRATION ABLATION - how much does calibration on the new machine actually add?

Same model, same training, same test clips as final_evaluation.py. Only what is learned from the NEW machine changes:
  none        : new machine is standardised with the AVERAGE statistics of the training machines;
                alert level comes from the training machines' healthy scores (nothing from the new machine)
  threshold   : same average statistics, but the alert level comes from 30 healthy clips of the new machine
  full        : new machine's own statistics (30 clips) AND its own alert level (next 30 clips)  = the real method
AUC does not depend on the alert level, so none vs threshold show the same AUC; full shows the effect of standardising.
"""
import os
import csv
import argparse
import numpy as np
import torch
import features as F
import model_selection as M
import evaluation_lib as E
import final_evaluation as FE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'ablation_calibration.csv')
K = 30
FIELDS = ['machine', 'snr', 'held_out', 'seed', 'variant', 'auc', 'precision', 'recall', 'f1', 'fpr', 'tp', 'fp', 'tn', 'fn']


def run_case(mtype, snr, held, seeds, epochs, ch):
    frames, depth, hidden, bott, mels = int(ch['frames']), ch['depth'], int(ch['hidden']), int(ch['bottleneck']), int(ch['mels'])
    root = FE.root_for(mels)
    FE.wait_for_features(root, snr, mtype)
    F.FEATURE_ROOT = root
    load = lambda i, lab: np.load(F.feature_path(snr, mtype, i, lab), mmap_mode='r')
    parts, mus, sds, train_healthy = [], [], [], []
    for i in [x for x in F.IDS if x != held]:
        n, a = load(i, 'normal'), load(i, 'abnormal')
        v = M.stack(n[len(a):], frames, M.TRAIN_STRIDE)
        mu, sd = M.stats(v)
        mus.append(mu)
        sds.append(sd)
        parts.append(((v.reshape(-1, v.shape[-1]) - mu) / sd).astype(np.float32))
        train_healthy.append(M.stack(n[:K], frames))          # a few of the training machine's own clips, for the alert level
    X = np.concatenate(parts)
    mu_avg, sd_avg = np.mean(mus, axis=0), np.mean(sds, axis=0)
    nh, ah = load(held, 'normal'), load(held, 'abnormal')
    k = len(ah)
    ev_n, ev_a = M.stack(nh[:k], frames), M.stack(ah, frames)
    pool = nh[k:]
    mu_h, sd_h = M.stats(M.stack(pool[:K], frames))
    thr_new = M.stack(pool[K:2 * K], frames)
    thr_first30 = M.stack(pool[:K], frames)
    rows = []
    for seed in seeds:
        model = M.build(X.shape[1], depth, hidden, bott)
        M.train(model, X, epochs, seed)
        # none: alert level from the training machines' healthy clips (each scored with its own statistics)
        s_train = np.concatenate([M.file_scores(model, th, m, s) for th, m, s in zip(train_healthy, mus, sds)])
        variants = {
            'none': (mu_avg, sd_avg, M.tau(s_train)),
            'threshold': (mu_avg, sd_avg, M.tau(M.file_scores(model, thr_first30, mu_avg, sd_avg))),
            'full': (mu_h, sd_h, M.tau(M.file_scores(model, thr_new, mu_h, sd_h))),
        }
        for name, (mu, sd, alert) in variants.items():
            met = E.full_metrics(M.file_scores(model, ev_n, mu, sd), M.file_scores(model, ev_a, mu, sd), alert)
            rows.append(dict(machine=mtype, snr=snr, held_out=held, seed=seed, variant=name, **{c: met[c] for c in FIELDS[5:]}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', nargs='*', default=['fan', 'pump', 'slider'])
    ap.add_argument('--snrs', nargs='*', default=['6'])
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ch_all = FE.choice()
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for snr in a.snrs:
            for t in a.types:
                for h in F.IDS:
                    w.writerows(run_case(t, snr, h, a.seeds, a.epochs, ch_all[t]))
                    fh.flush()
                    print(f'{snr}dB {t} held-out {h} done', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
