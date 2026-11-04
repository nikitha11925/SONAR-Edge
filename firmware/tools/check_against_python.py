"""Compiles the C pipeline on the laptop and compares its log-mel output with acoustic/features.py (librosa) on a test frame."""
import os, subprocess, sys, tempfile
import numpy as np
import librosa

FW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mels = int(sys.argv[1]) if len(sys.argv) > 1 else 64
exe = os.path.join(tempfile.gettempdir(), 'ae_host_test.exe')
subprocess.check_call(['gcc', '-O2', '-I', f'{FW}/common', f'{FW}/host_test.c', f'{FW}/common/ae_pipeline.c',
                       f'{FW}/common/model_data.c', '-o', exe, '-lm'])

rng = np.random.default_rng(0)
t = np.arange(1024) / 16000
x = (6000 * np.sin(2 * np.pi * 120 * t) + 2000 * np.sin(2 * np.pi * 1000 * t) + rng.normal(0, 300, 1024)).astype(np.int16)
raw = os.path.join(tempfile.gettempdir(), 'frame.raw')
x.tofile(raw)
out = subprocess.check_output([exe, raw], text=True).split('\n')
c = np.array([float(v) for v in out[:mels]])

# same maths as features.logmel on exactly one frame (no centre padding)
y = x.astype(np.float32) / 32768
mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=512, n_mels=mels, power=2.0, center=False)
p = 10 * np.log10(mel[:, 0] + sys.float_info.epsilon)
print('max |C - Python| over', mels, 'bands: %.5f dB' % np.abs(c - p).max())
print(out[mels])
