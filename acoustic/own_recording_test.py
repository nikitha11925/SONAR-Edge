"""
TEST WITH YOUR OWN RECORDINGS (different microphone, different machine).

    python own_recording_test.py --healthy <folder or file> --faulty <folder or file> [--type fan]
    python own_recording_test.py --healthy <rec A> --healthy-test <rec B> --faulty <rec C>     (fairest: separate healthy session)

(The same logic is used by the "5. My recordings" tab of dashboard_gui.py.)

Steps (same as the dashboard / final evaluation, only the source of the sound changes):
  1. Read every audio file in the folder (or the single file). WAV, MP3, M4A (also with a .mpeg suffix added by WhatsApp).
     Mono: first channel. Resampled to 16 kHz. Long recordings are cut into 10-second clips (leftover < 10 s is dropped).
  2. Healthy clips, in time order:  first K -> this machine's feature numbers (mean, spread)
                                    next K  -> alert level = mean + 3*std of their scores
                                    rest    -> healthy TEST clips (never used for calibration), unless --healthy-test is given
  3. Faulty clips are all TEST clips.
  4. The model is the one trained on MIMII machines (NOT on your machine): for 'fan' the model trained on fans 00, 02, 04.
"""
import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
import demo as D
import features as F
import model_selection as M
import evaluation_lib as E

CLIP = 10 * 16000
EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.mp4', '.mpeg')


def read_audio(f):
    """(samples, sample rate), first channel only. M4A/AAC goes through PyAV, the rest through soundfile."""
    if f.lower().endswith(('.m4a', '.aac', '.mp4', '.mpeg')):
        import av
        chunks, sr = [], None
        with av.open(f) as c:
            for fr in c.decode(audio=0):
                a = fr.to_ndarray().astype(np.float32)
                a = a[0] if fr.format.is_planar else a.reshape(-1, len(fr.layout.channels))[:, 0]
                chunks.append(a)
                sr = fr.sample_rate
        y = np.concatenate(chunks)
        if np.abs(y).max() > 2.0:           # integer formats come back unscaled
            y = y / 32768.0
        return y, sr
    y, sr = sf.read(f, dtype='float32', always_2d=True)
    return y[:, 0], sr


def cut_clips(y, sr):
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return [y[i * CLIP:(i + 1) * CLIP] for i in range(len(y) // CLIP)]


def read_clips(path):
    files = [path] if os.path.isfile(path) else sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(EXTS))
    if not files:
        raise ValueError(f'No audio files found in {path}')
    clips = []
    for f in files:
        clips += cut_clips(*read_audio(f))
    return clips


def analyse(healthy, faulty, mtype='fan', k=D.K, min_test=10, healthy_test=None):
    """Calibrate on `healthy`, test on healthy_test (or the leftover healthy clips) and `faulty`. Raises ValueError on bad input."""
    hc, fc = read_clips(healthy), read_clips(faulty)
    htc = read_clips(healthy_test) if healthy_test else None
    warnings = []
    need_test = 0 if htc else min_test
    k0 = k
    if len(hc) < 2 * k + need_test:
        k = (len(hc) - need_test) // 2
        if k < 5:
            raise ValueError(f'Need at least {2 * 5 + need_test} healthy clips of 10 s for calibration (you have {len(hc)}).')
        warnings.append(f'Not enough healthy clips for K={k0}; using K={k}. Results are less reliable.')
    if len(fc) < 5:
        raise ValueError(f'Need at least 5 faulty clips of 10 s (you have {len(fc)}).')
    design, mels = D.chosen_design(mtype)
    _, frames, _, _, _ = design
    model, _ = D.get_model(mtype, 'id_06', '6', design, mels, False)
    to_lm = lambda clips: [F.logmel(c, n_mels=mels) for c in clips]
    lh, lf = to_lm(hc), to_lm(fc)
    mu, sd = M.stats(M.stack(lh[:k], frames))
    alert = M.tau(M.file_scores(model, M.stack(lh[k:2 * k], frames), mu, sd))
    test_h = to_lm(htc) if htc else lh[2 * k:]
    sn = M.file_scores(model, M.stack(test_h, frames), mu, sd)
    sa = M.file_scores(model, M.stack(lf, frames), mu, sd)
    met = E.full_metrics(sn, sa, alert)
    fpr, tpr = E.roc_points(sn, sa)
    return dict(met=met, sn=sn, sa=sa, alert=alert, k=k, n_cal_clips=len(hc), mtype=mtype,
                params=M.n_params(model), roc=(fpr, tpr), warnings=warnings, separate_healthy_session=bool(htc),
                ctx=dict(model=model, frames=frames, mels=mels, mu=mu, sd=sd, alert=alert))


def score_recording(ctx, path):
    """Scores of every 10-second clip of one recording, using an already calibrated context."""
    clips = read_clips(path)
    if not clips:
        raise ValueError('The recording is shorter than 10 seconds.')
    lms = [F.logmel(c, n_mels=ctx['mels']) for c in clips]
    return M.file_scores(ctx['model'], M.stack(lms, ctx['frames']), ctx['mu'], ctx['sd'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--healthy', required=True)
    ap.add_argument('--faulty', required=True)
    ap.add_argument('--type', default='fan', choices=F.TYPES)
    ap.add_argument('--k', type=int, default=D.K)
    ap.add_argument('--min-test', type=int, default=10, help='healthy clips kept for testing')
    ap.add_argument('--healthy-test', default=None,
                    help='optional: a SEPARATE healthy recording (another session) used as the healthy test clips; '
                         'then every clip of --healthy is used for calibration only')
    a = ap.parse_args()
    try:
        r = analyse(a.healthy, a.faulty, a.type, a.k, a.min_test, a.healthy_test)
    except ValueError as e:
        sys.exit(str(e))
    m = r['met']
    for w in r['warnings']:
        print('WARNING:', w)
    print(f'model: {a.type}, trained on MIMII machines other than id_06 ({r["params"]:,} parameters)')
    print(f'calibration: {r["k"]} clips for feature numbers + {r["k"]} clips for the alert level; alert level = {r["alert"]:.4f}')
    print(f'test: {len(r["sn"])} healthy + {len(r["sa"])} faulty clips (not used for calibration)\n')
    print(f'AUC              {m["auc"]:.3f}   (0.5 = guessing, 1.0 = perfect)')
    print(f'recall           {m["recall"]:.3f}   (faults caught)')
    print(f'precision        {m["precision"]:.3f}')
    print(f'false-alarm rate {m["fpr"]:.3f}   (healthy clips wrongly flagged)')
    print(f'accuracy         {m["accuracy"]:.3f}')
    print(f'confusion: faults caught {m["tp"]}, faults missed {m["fn"]}, false alarms {m["fp"]}, healthy passed {m["tn"]}')
    print(f'\nscores: healthy median {np.median(r["sn"]):.4f}, faulty median {np.median(r["sa"]):.4f}, alert level {r["alert"]:.4f}')


if __name__ == '__main__':
    main()
