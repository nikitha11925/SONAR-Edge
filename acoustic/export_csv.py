"""
Export the extracted features of ONE clip to CSV so they can be inspected in Excel.
  logmel  : 64 rows (mel bands) x 313 columns (time frames)   [the spectrogram]
  vectors : 309 rows (model inputs) x 320 columns (5 frames x 64 bands)
Usage: python export_csv.py --snr 6 --type fan --id id_00 --label normal --index 0
"""
import os
import argparse
import numpy as np
import features as F

ap = argparse.ArgumentParser()
ap.add_argument('--snr', default='6')
ap.add_argument('--type', default='fan')
ap.add_argument('--id', default='id_00')
ap.add_argument('--label', default='normal')
ap.add_argument('--index', type=int, default=0, help='which file (sorted order) in that folder')
ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv_samples'))
a = ap.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
lm = F.load_logmels(a.snr, a.type, a.id, a.label)[a.index]          # (64, 313)
name = os.path.basename(F.list_wavs(a.snr, a.type, a.id, a.label)[a.index])[:-4]
tag = f'{a.snr}dB_{a.type}_{a.id}_{a.label}_{name}'

# 1) the spectrogram: rows = mel bands, columns = time frames
p1 = os.path.join(a.out_dir, f'{tag}_logmel.csv')
np.savetxt(p1, lm, delimiter=',', fmt='%.3f',
           header=','.join(f'frame_{t}' for t in range(lm.shape[1])), comments='')

# 2) the model inputs: each row = 5 stacked frames = 320 attributes
v = F.stack_frames(lm)                                               # (309, 320)
cols = [f'f{t}_mel{b}' for t in range(F.FRAMES) for b in range(F.N_MELS)]
p2 = os.path.join(a.out_dir, f'{tag}_model_inputs.csv')
np.savetxt(p2, v, delimiter=',', fmt='%.3f', header=','.join(cols), comments='')
print('logmel  :', lm.shape, '->', p1)
print('vectors :', v.shape, '->', p2)
