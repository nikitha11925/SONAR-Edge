"""
LIVE DEMO (simulation): recorded clips are replayed as if they came from the microphone.

What it shows, step by step:
  1. A small model is trained ONLY on healthy sound of the OTHER machines of the same type.
  2. The "new machine" (held-out ID) is calibrated: the device learns that machine's normal levels from healthy
     clips and sets its own alert level (average + 3 x spread).
  3. Test clips (mixed normal and faulty, shuffled) are read from the original WAV files, turned into features
     (channel 1, log-mel) and judged: score above the alert level = RED (alert), otherwise GREEN.
  4. A summary and a picture are produced.

IMPORTANT: this runs on the laptop, not on the microcontroller. Training and calibration use stored features of
recordings made on the same dataset; the test clips go through the real audio -> features -> model path.

Usage:  python demo.py --type fan --held id_06 --clips 16
        python demo.py --type fan --held id_06 --clips 16 --delay 1.5     (one verdict every 1.5 s, feels live)
        python demo.py --type slider --held id_06 --clips 16 --play 3     (plays the first 3 clips on your speakers)
        add --no-show to stop the picture from opening automatically
"""
import os
import csv
import argparse
import time
import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import features as F
import model_selection as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'demo_output')
MODEL_DIR = os.path.join(HERE, 'demo_models')
K = 30


def chosen_design(mtype):
    rows = {r['machine']: r for r in csv.DictReader(open(os.path.join(HERE, 'results', 'model_selection_choice_final.csv')))}
    r = rows[mtype]
    return (r['design'], int(r['frames']), r['depth'], int(r['hidden']), int(r['bottleneck'])), int(r['mels'])


def feature_root(mels):
    return M.FEATURE_ROOT_64 if mels == 64 else rf'N:\dev\mimii_features_b{mels}'


def get_model(mtype, held, snr, design, mels, retrain):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f'{mtype}_{held}_{snr}dB.pt')
    name, frames, depth, hidden, bott = design
    in_dim = mels * frames
    model = M.build(in_dim, depth, hidden, bott)
    if os.path.exists(path) and not retrain:
        model.load_state_dict(torch.load(path))
        return model, False
    F.FEATURE_ROOT = feature_root(mels)
    train_ids = [i for i in F.IDS if i != held]
    parts = []
    for i in train_ids:
        n = F.load_logmels(snr, mtype, i, 'normal')
        a = F.load_logmels(snr, mtype, i, 'abnormal')
        v = M.stack(n[len(a):], frames, M.TRAIN_STRIDE)
        mu, sd = M.stats(v)
        parts.append(((v.reshape(-1, v.shape[-1]) - mu) / sd).astype(np.float32))
    torch.set_num_threads(8)
    M.train(model, np.concatenate(parts), 30, 0)
    torch.save(model.state_dict(), path)
    return model, True


def clip_features(wav_path, mels, frames):
    y = F.load_channel0(wav_path)
    lm = F.logmel(y, n_mels=mels)
    return F.stack_frames(lm, frames=frames)[None]       # (1, n_vec, dim)


def play(wav_path):
    import winsound
    y = F.load_channel0(wav_path)
    tmp = os.path.join(OUT_DIR, '_play.wav')
    sf.write(tmp, y, 16000, subtype='PCM_16')
    winsound.PlaySound(tmp, winsound.SND_FILENAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', default='fan', choices=F.TYPES)
    ap.add_argument('--held', default='id_06', choices=F.IDS)
    ap.add_argument('--snr', default='6')
    ap.add_argument('--clips', type=int, default=16)
    ap.add_argument('--play', type=int, default=0, help='play the first N test clips on the speakers (Windows)')
    ap.add_argument('--retrain', action='store_true')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--delay', type=float, default=0.0, help='seconds to pause between clips (makes the demo feel live)')
    ap.add_argument('--no-show', action='store_true', help='do not open the picture at the end')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    design, mels = chosen_design(a.type)
    name, frames, depth, hidden, bott = design

    print('=' * 72)
    print(' SIMULATION - recorded clips replayed as if from the microphone')
    print(f' machine type: {a.type}   new machine: {a.held}   noise level: {a.snr} dB')
    print(f' model: {name} with {mels} mel bands (single frame)')
    print('=' * 72)

    t0 = time.time()
    model, trained = get_model(a.type, a.held, a.snr, design, mels, a.retrain)
    n_par = M.n_params(model)
    print(f'[1] Model {"trained" if trained else "loaded"} on healthy sound of the OTHER {a.type}s only '
          f'({n_par:,} parameters, {time.time() - t0:.0f}s)')

    # ---- calibration on the new machine (healthy clips only) ----
    F.FEATURE_ROOT = feature_root(mels)
    nh = F.load_logmels(a.snr, a.type, a.held, 'normal')
    ah_files = F.list_wavs(a.snr, a.type, a.held, 'abnormal')
    nh_files = F.list_wavs(a.snr, a.type, a.held, 'normal')
    k = len(ah_files)
    pool = nh[k:]
    stats_vecs = M.stack(pool[:K], frames)
    mu, sd = M.stats(stats_vecs)
    thr_vecs = M.stack(pool[K:2 * K], frames)
    s_thr = M.file_scores(model, thr_vecs, mu, sd)
    alert = M.tau(s_thr)
    print(f'[2] Calibrated on {2 * K} healthy clips of {a.held} (= {2 * K * 10 // 60} min of sound): '
          f'learned its normal levels, alert level = {alert:.3f}')

    # ---- test clips straight from the WAV files ----
    rng = np.random.default_rng(a.seed)
    half = a.clips // 2
    normal_idx = rng.choice(k, half, replace=False)                  # evaluation normals are the first k files
    abn_idx = rng.choice(len(ah_files), a.clips - half, replace=False)
    tests = [(nh_files[i], 'normal') for i in normal_idx] + [(ah_files[i], 'faulty') for i in abn_idx]
    order = rng.permutation(len(tests))
    tests = [tests[i] for i in order]

    print(f'[3] Listening to {len(tests)} new clips (shuffled, the device is not told which are faulty):')
    print('    clip  score   alert   verdict            truth     result')
    scores, truths = [], []
    ok = fa = miss = det = 0
    for n, (wav, truth) in enumerate(tests, 1):
        if n <= a.play:
            print(f'    (playing clip {n} ...)')
            play(wav)
        v = clip_features(wav, mels, frames)
        s = float(M.file_scores(model, v, mu, sd)[0])
        verdict = 'RED   (alert)' if s > alert else 'GREEN (normal)'
        flagged = s > alert
        if truth == 'normal':
            res = 'FALSE ALARM' if flagged else 'ok'
            fa += flagged
            ok += (not flagged)
        else:
            res = 'ok (caught)' if flagged else 'MISSED'
            det += flagged
            miss += (not flagged)
            ok += flagged
        scores.append(s)
        truths.append(truth)
        print(f'    {n:4d}  {s:6.3f}  {alert:6.3f}  {verdict:17s}  {truth:8s}  {res}', flush=True)
        if a.delay > 0:
            time.sleep(a.delay)

    nn_ = sum(1 for t in truths if t == 'normal')
    nf = len(truths) - nn_
    print('-' * 72)
    print(f'    correct: {ok}/{len(tests)}   caught faults: {det}/{nf}   false alarms: {fa}/{nn_}   missed faults: {miss}/{nf}')
    print('    (small sample; the full tests use hundreds of clips - see the results tables)')

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (s, t) in enumerate(zip(scores, truths), 1):
        ax.scatter(i, s, c='tab:red' if t == 'faulty' else 'tab:green', s=70, edgecolors='k')
    ax.axhline(alert, color='k', linestyle='--', label=f'alert level ({alert:.2f})')
    ax.scatter([], [], c='tab:green', edgecolors='k', label='healthy clip')
    ax.scatter([], [], c='tab:red', edgecolors='k', label='faulty clip')
    ax.set_xlabel('clip number (replayed in this order)')
    ax.set_ylabel('how strange the sound is (error score)')
    ax.set_title(f'{a.type} {a.held}: calibrated on this machine\'s healthy sound, then tested ({a.snr} dB)')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'demo_{a.type}_{a.held}_{a.snr}dB.png')
    fig.savefig(out, dpi=120)
    print(f'    picture saved: {out}')
    if not a.no_show:
        try:
            os.startfile(out)            # opens the picture in the default viewer (Windows)
        except Exception as e:           # pragma: no cover
            print(f'    (could not open the picture automatically: {e})')


if __name__ == '__main__':
    main()
