"""
Hitachi MIMII baseline re-implementation (PyTorch).
Dense AE 320-64-64-8-64-64-320, Adam(1e-3), 50 epochs, batch 512, last 10% of
training vectors held out as validation (monitoring only, final epoch is used).
Split (identical to Hitachi baseline.py): sorted normal files, the first
len(abnormal) normals + all abnormals = eval set, remaining normals = train.
Score per file = mean over its frames of the per-vector MSE; metric = ROC-AUC.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import features as F


def build_model(dim=320):
    layers = []
    sizes = [dim, 64, 64, 8, 64, 64, dim]
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.xavier_uniform_(lin.weight)   # Keras glorot_uniform default
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def stack_all(logmels):
    """(n_files, 64, T) -> (n_files, T-4, 320)"""
    return np.stack([F.stack_frames(lm) for lm in logmels])


def split_files(normal, abnormal):
    n_abn = len(abnormal)
    eval_normal = normal[:n_abn]
    train = normal[n_abn:]
    return train, eval_normal


def per_file_scores(model, vecs):
    """vecs (n_files, n_vec, 320) -> mean MSE per file"""
    model.eval()
    out = []
    with torch.no_grad():
        for v in vecs:
            x = torch.from_numpy(v)
            out.append(torch.mean((model(x) - x) ** 2).item())
    return np.array(out)


def train_model(train_vecs, epochs=50, batch=512, seed=0, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = train_vecs.reshape(-1, train_vecs.shape[-1])
    n_val = int(0.1 * len(X))
    Xtr, Xva = torch.from_numpy(X[:len(X) - n_val]), torch.from_numpy(X[len(X) - n_val:])
    model = build_model(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(Xtr)))
        tot, cnt = 0.0, 0
        for i in range(0, len(Xtr), batch):
            xb = Xtr[perm[i:i + batch]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), Xva).item()
        hist.append((tot / cnt, vl))
        if log and (ep % 10 == 9 or ep == 0):
            log(f'    epoch {ep + 1}/{epochs} loss {tot / cnt:.3f} val {vl:.3f}')
    return model, hist


def run_one(snr, mtype, mid, seed=0, epochs=50, log=print):
    t0 = time.time()
    normal = F.load_logmels(snr, mtype, mid, 'normal')
    abnormal = F.load_logmels(snr, mtype, mid, 'abnormal')
    train_lm, eval_normal_lm = split_files(normal, abnormal)
    train_vecs = stack_all(train_lm)
    model, hist = train_model(train_vecs, epochs=epochs, seed=seed, log=log)
    s_norm = per_file_scores(model, stack_all(eval_normal_lm))
    s_abn = per_file_scores(model, stack_all(abnormal))
    y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_abn))]
    auc = roc_auc_score(y, np.r_[s_norm, s_abn])
    return dict(snr=snr, machine=mtype, model_id=mid, seed=seed, auc=float(auc),
                n_train_files=len(train_lm), n_eval_normal=len(s_norm), n_eval_abnormal=len(s_abn),
                final_train_loss=hist[-1][0], final_val_loss=hist[-1][1],
                mean_score_normal=float(s_norm.mean()), mean_score_abnormal=float(s_abn.mean()),
                seconds=round(time.time() - t0, 1))
