"""
Shared metric definitions, used by BOTH the batch evaluation (final_evaluation.py) and the interface (dashboard_gui.py),
so the numbers shown on screen are computed exactly like the numbers in the results files.

Test set: Hitachi's split of the new machine = N healthy clips + all N faulty clips (balanced).
Decision rule on the device: clip score > alert level  ->  alert.  alert level = mean + 3*std of healthy calibration scores.
"""
import time
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import features as F


def full_metrics(sn, sa, alert):
    """sn = scores of healthy test clips, sa = scores of faulty test clips, alert = alert level."""
    y = np.r_[np.zeros(len(sn)), np.ones(len(sa))]
    s = np.r_[sn, sa]
    tp = int((sa > alert).sum())
    fn = len(sa) - tp
    fp = int((sn > alert).sum())
    tn = len(sn) - fp
    recall = tp / (tp + fn) if tp + fn else float('nan')
    spec = tn / (tn + fp) if tn + fp else float('nan')
    prec = tp / (tp + fp) if tp + fp else float('nan')
    f1 = 2 * prec * recall / (prec + recall) if (tp + fp) and (prec + recall) > 0 else 0.0
    return dict(auc=float(roc_auc_score(y, s)), precision=prec, recall=recall, f1=f1,
                accuracy=(tp + tn) / (tp + tn + fp + fn), fpr=fp / (fp + tn) if fp + tn else 0.0,
                specificity=spec, balanced_accuracy=(recall + spec) / 2, tp=tp, fp=fp, tn=tn, fn=fn)


def roc_points(sn, sa):
    y = np.r_[np.zeros(len(sn)), np.ones(len(sa))]
    fpr, tpr, _ = roc_curve(y, np.r_[sn, sa])
    return fpr, tpr


def precision_at_prevalence(recall, fpr, p):
    """Precision you would see if only a fraction p of the clips are faulty (real factories: faults are rare)."""
    den = recall * p + fpr * (1 - p)
    return recall * p / den if den > 0 else float('nan')


def linear_layers(model):
    return [m for m in model if isinstance(m, nn.Linear)]


def model_card(model, mels, frames, repeats=2000):
    """Size and cost of one model. 'MACs' = multiply-accumulate operations per input vector (one 32 ms frame)."""
    lins = linear_layers(model)
    params = sum(p.numel() for p in model.parameters())
    macs = sum(l.in_features * l.out_features for l in lins)
    widest = max(max(l.in_features, l.out_features) for l in lins)
    # estimated memory for the chip (not measured): weights + two activation buffers (32-bit) + feature buffers
    ram_acts = 2 * widest * 4
    # features per frame: Hann window table + frame buffer (32-bit) + one log-mel row per stacked frame
    ram_features = (1024 + 1024 + 513 * 2 + mels * frames) * 4
    # operations to make one feature frame (standard operation counts, estimates)
    mel_w = F.librosa.filters.mel(sr=16000, n_fft=1024, n_mels=mels)
    mel_nnz = int((mel_w > 0).sum())
    feat_ops = 1024 + 25600 + 513 + mel_nnz + mels        # window + real FFT (~2.5 N log2 N) + power + mel + log
    # laptop inference time (measured here; NOT the chip)
    x = torch.randn(1, lins[0].in_features)
    model.eval()
    with torch.no_grad():
        for _ in range(200):
            model(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        dt = (time.perf_counter() - t0) / repeats
    return dict(params=params, macs=macs, size_f32_kb=params * 4 / 1024, size_int8_kb=params / 1024,
                ram_activations_bytes=ram_acts, ram_features_bytes=ram_features, mel_nonzero_weights=mel_nnz,
                feature_ops=feat_ops, total_ops_per_frame=feat_ops + macs, laptop_ms_per_frame=dt * 1000)
