"""
CWRU vibration, version 2 (clean).

Fixes over dataset/clean_cwru_dataset.py:
  * NO fabricated channel: only drive-end (DE) and fan-end (FE), which exist in all four recordings.
    (The old script invented a 'BA' channel for the healthy file with random noise.)
  * Time-ordered split of the healthy recording with gaps (no shuffling of overlapping frames):
        block 1 (first 40%)  -> feature statistics ("calibration")
        block 2 (next 20%)   -> alert level  tau = mean + 3*std
        block 3 (last 40%)   -> healthy test frames
  * Two sensor paths:
        full : 12 kHz, band-pass 10-5000 Hz (reference; what CWRU's sensors provide)
        mpu  : DC removed, low-pass 260 Hz, resampled to 1 kHz (imitates a low-bandwidth sensor such as the MPU6050)
  * Optional white sensor noise added to the 'mpu' path, relative to the healthy signal strength, to see how much
    margin the detection has.
Detector: no neural network. Each 32 ms frame -> RMS per channel -> standardised with the healthy block-1 numbers ->
score = mean squared z-score. Alert level from block 2. Metrics: AUC, false-alarm rate (block 3), detection rate.
Null check: AUC of healthy block 2 vs healthy block 3 should be near 0.5 (no drift inside the healthy recording).
"""

import os
import sys
import csv
import numpy as np
from scipy import signal
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean_cwru_dataset as C          # re-used only for loading and the standard cleaning chain

FS = 12000
FILES = {'healthy': '97', 'inner_race': '105', 'ball': '118', 'outer_race': '130'}
CHANNELS = ['DE', 'FE']
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 4.0]   # noise RMS as a multiple of the healthy signal RMS
NOISE_DRAWS = 5
OUT_DIR = os.path.join(HERE, 'cwru', 'processed_v2')
os.makedirs(OUT_DIR, exist_ok=True)


def load(name):
    p = C.parse_mat_file(os.path.join(HERE, 'cwru', f'{name}.mat'))
    return {ch: p[ch] for ch in CHANNELS}, p['RPM']


def path_full(x):
    return C.clean_signal_pipeline(x, fs=FS), FS


def path_mpu(x):
    x = x - x.mean()
    b, a = signal.butter(4, 260 / (FS / 2))
    y = signal.filtfilt(b, a, x)
    y = signal.resample_poly(y, 1, 12)
    return y - y.mean(), 1000


def frames_rms(x, fs):
    n = int(round(0.032 * fs))
    hop = n // 2
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    return np.sqrt((x[idx] ** 2).mean(axis=1))


def features(path_fn, noise_sigma=None, rng=None):
    """returns {condition: (frames, 2) RMS matrix}; noise_sigma = per-channel noise std (same for all recordings)"""
    out = {}
    for cond, f in FILES.items():
        chans, _ = load(f)
        cols = []
        for ch in CHANNELS:
            y, fs = path_fn(chans[ch])
            if noise_sigma is not None:
                y = y + rng.normal(0, noise_sigma[ch], len(y))
            cols.append(frames_rms(y, fs))
        out[cond] = np.stack(cols, axis=1)
    return out


def split_healthy(n):
    g = 2
    b1 = (0, int(0.4 * n))
    b2 = (b1[1] + g, int(0.6 * n))
    b3 = (b2[1] + g, n)
    return b1, b2, b3


def score(F, mu, sd, cols):
    z = (F[:, cols] - mu[cols]) / sd[cols]
    return (z ** 2).mean(axis=1)


def evaluate(feat):
    h = feat['healthy']
    b1, b2, b3 = split_healthy(len(h))
    mu, sd = h[b1[0]:b1[1]].mean(0), h[b1[0]:b1[1]].std(0) + 1e-9
    rows = []
    for det, cols in (('DE+FE', [0, 1]), ('DE only', [0]), ('FE only', [1])):
        s2 = score(h[b2[0]:b2[1]], mu, sd, cols)
        s3 = score(h[b3[0]:b3[1]], mu, sd, cols)
        tau = s2.mean() + 3 * s2.std()
        null_auc = roc_auc_score(np.r_[np.zeros(len(s2)), np.ones(len(s3))], np.r_[s2, s3])
        fpr = float((s3 > tau).mean())
        for fault in ('inner_race', 'ball', 'outer_race'):
            sf = score(feat[fault], mu, sd, cols)
            auc = roc_auc_score(np.r_[np.zeros(len(s3)), np.ones(len(sf))], np.r_[s3, sf])
            rows.append(dict(detector=det, fault=fault, auc=auc, fpr=fpr, tpr=float((sf > tau).mean()), null_auc=null_auc))
    return rows


def main():
    results = []
    # reference: full band, no noise
    for r in evaluate(features(path_full)):
        results.append(dict(r, path='full 12 kHz', noise=0.0))
    # MPU-like path with increasing sensor noise
    base = features(path_mpu)
    h_rms = {ch: np.sqrt(np.mean(path_mpu(load('97')[0][ch])[0] ** 2)) for ch in CHANNELS}
    for lvl in NOISE_LEVELS:
        if lvl == 0:
            rows = evaluate(base)
            for r in rows:
                results.append(dict(r, path='MPU-like 1 kHz', noise=lvl))
            continue
        acc = []
        for d in range(NOISE_DRAWS):
            rng = np.random.default_rng(1000 + d)
            sig = {ch: lvl * h_rms[ch] for ch in CHANNELS}
            acc.append(evaluate(features(path_mpu, sig, rng)))
        for i in range(len(acc[0])):
            r0 = acc[0][i]
            r = dict(r0)
            for k in ('auc', 'fpr', 'tpr', 'null_auc'):
                r[k] = float(np.mean([a[i][k] for a in acc]))
            results.append(dict(r, path='MPU-like 1 kHz', noise=lvl))

    fields = ['path', 'noise', 'detector', 'fault', 'auc', 'fpr', 'tpr', 'null_auc']
    csv_path = os.path.join(HERE, 'cwru_vibration_v2_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(results)

    # ---- summary ----
    lines = ['CWRU vibration v2 - clean pipeline (no fabricated channel, time-ordered split, no neural network)', '']
    lines.append('Health check on the healthy recording: AUC between its block 2 and block 3 (0.5 = no drift)')
    nulls = {(r['path'], r['noise'], r['detector']): r['null_auc'] for r in results}
    lines.append(f"  full band, DE+FE: {nulls[('full 12 kHz', 0.0, 'DE+FE')]:.2f};  MPU-like, DE+FE: {nulls[('MPU-like 1 kHz', 0.0, 'DE+FE')]:.2f}")
    lines.append('')
    lines.append('Detector DE+FE: AUC per fault, with false-alarm rate (healthy test block) and detection rate')
    lines.append('| sensor path | added noise (x healthy RMS) | inner race | ball | outer race | false alarms | detected (avg of 3 faults) |')
    lines.append('|---|---|---|---|---|---|---|')
    for path, noise in [('full 12 kHz', 0.0)] + [('MPU-like 1 kHz', n) for n in NOISE_LEVELS]:
        sel = {r['fault']: r for r in results if r['path'] == path and r['noise'] == noise and r['detector'] == 'DE+FE'}
        fpr = list(sel.values())[0]['fpr']
        tpr = np.mean([sel[f]['tpr'] for f in sel])
        lines.append(f"| {path} | {noise:g} | {sel['inner_race']['auc']:.3f} | {sel['ball']['auc']:.3f} | {sel['outer_race']['auc']:.3f} | {fpr*100:.0f}% | {tpr*100:.0f}% |")
    lines.append('')
    lines.append('Single-channel detectors, MPU-like path, no added noise (AUC inner/ball/outer):')
    for det in ('DE only', 'FE only'):
        sel = {r['fault']: r['auc'] for r in results if r['path'] == 'MPU-like 1 kHz' and r['noise'] == 0.0 and r['detector'] == det}
        lines.append(f"  {det}: {sel['inner_race']:.3f} / {sel['ball']:.3f} / {sel['outer_race']:.3f}")
    summary = '\n'.join(lines)
    print(summary)
    with open(os.path.join(HERE, 'cwru_vibration_v2_summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')

    # ---- picture: healthy vs faults, RMS of the two real channels ----
    feat = features(path_full)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    b1, b2, b3 = split_healthy(len(feat['healthy']))
    ax[0].scatter(feat['healthy'][b3[0]:b3[1], 0], feat['healthy'][b3[0]:b3[1], 1], s=6, label='healthy (test block)', color='tab:green')
    for fault, col in (('inner_race', 'tab:red'), ('ball', 'tab:orange'), ('outer_race', 'tab:purple')):
        ax[0].scatter(feat[fault][:, 0], feat[fault][:, 1], s=6, label=fault, color=col, alpha=0.6)
    ax[0].set_xlabel('RMS drive-end sensor (g)')
    ax[0].set_ylabel('RMS fan-end sensor (g)')
    ax[0].set_title('Each dot = one 32 ms frame (full band)')
    ax[0].legend()
    sub = [r for r in results if r['path'] == 'MPU-like 1 kHz' and r['detector'] == 'DE+FE']
    for fault, col in (('inner_race', 'tab:red'), ('ball', 'tab:orange'), ('outer_race', 'tab:purple')):
        xs = NOISE_LEVELS
        ys = [[r['auc'] for r in sub if r['noise'] == n and r['fault'] == fault][0] for n in xs]
        ax[1].plot(xs, ys, marker='o', label=fault, color=col)
    ax[1].set_xlabel('added sensor noise (multiple of healthy signal RMS)')
    ax[1].set_ylabel('AUC')
    ax[1].set_ylim(0.4, 1.02)
    ax[1].set_title('Detection vs sensor noise (MPU-like path)')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'vibration_v2_overview.png'), dpi=110)
    print('saved', csv_path)


if __name__ == '__main__':
    main()
