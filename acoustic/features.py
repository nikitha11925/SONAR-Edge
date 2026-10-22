"""
MIMII log-mel feature extraction (Hitachi baseline-compatible).
Channel 1 only, n_fft=1024, hop=512, 64 mel bands, power=2,
log-mel = 20/power * log10(mel + eps), 5 consecutive frames stacked -> 320 dims.
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa

DATA_ROOT = r'N:\dev\Sonar-acoustic-dataset'
FEATURE_ROOT = r'N:\dev\mimii_features'

N_MELS = 64
N_FFT = 1024
HOP = 512
POWER = 2.0
FRAMES = 5

TYPES = ['fan', 'pump', 'slider', 'valve']
IDS = ['id_00', 'id_02', 'id_04', 'id_06']
SNRS = ['-6', '0', '6']


def class_dir(snr, mtype, mid, label, root=DATA_ROOT):
    return os.path.join(root, f'{snr}_dB_{mtype}', mtype, mid, label)


def list_wavs(snr, mtype, mid, label, root=DATA_ROOT):
    d = class_dir(snr, mtype, mid, label, root)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.wav'))


def load_channel0(path):
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    assert sr == 16000, sr
    return y[:, 0]


def logmel(y, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=n_mels, power=POWER)
    return (20.0 / POWER * np.log10(mel + sys.float_info.epsilon)).astype(np.float32)


def stack_frames(lm, frames=FRAMES):
    """(n_mels, T) -> (T-frames+1, n_mels*frames), Hitachi-style layout."""
    n_mels, T = lm.shape
    n = T - frames + 1
    out = np.zeros((n, n_mels * frames), dtype=np.float32)
    for t in range(frames):
        out[:, n_mels * t:n_mels * (t + 1)] = lm[:, t:t + n].T
    return out


def feature_path(snr, mtype, mid, label):
    return os.path.join(FEATURE_ROOT, f'{snr}_{mtype}_{mid}_{label}.npy')


def load_logmels(snr, mtype, mid, label):
    """(n_files, 64, T) float32 log-mel array, files in sorted-name order."""
    return np.load(feature_path(snr, mtype, mid, label))
