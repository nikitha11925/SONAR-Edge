"""Summarise calibration v2: mean +/- spread over the 4 machine IDs (and seeds) per machine type, per method."""
import os
import csv
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, 'results', 'calibration_v2_results.csv')
K_SHOW = int(sys.argv[1]) if len(sys.argv) > 1 else 30
snr_show = sys.argv[2] if len(sys.argv) > 2 else '6'
rows = [r for r in csv.DictReader(open(path)) if r['snr'] == snr_show]
names = {'M0_no_calibration': 'no calibration', 'M1_threshold_only': 'threshold only',
         'M2_full_calibration': 'full calibration', 'S_simple_zscore': 'simple control (no network)',
         'O_oracle_per_machine': 'per-machine model (best case)'}
order = ['M0_no_calibration', 'M1_threshold_only', 'M2_full_calibration', 'S_simple_zscore', 'O_oracle_per_machine']
lines = [f'Calibration v2, {snr_show} dB, K={K_SHOW} calibration clips (mean over the 4 machines of a type; +/- = spread over seeds)', '',
         '| type | method | AUC | false alarms | faults detected |', '|---|---|---|---|---|']
for t in ['fan', 'pump', 'slider', 'valve']:
    for m in order:
        sel = [r for r in rows if r['machine'] == t and r['method'] == m and (int(r['K']) == K_SHOW or m == 'M0_no_calibration')]
        if not sel:
            continue
        # average over IDs first (per seed), then report mean and spread over seeds
        seeds = sorted({r['seed'] for r in sel})
        per_seed = []
        for s in seeds:
            ss = [r for r in sel if r['seed'] == s]
            per_seed.append([np.mean([float(r[c]) for r in ss]) for c in ('auc', 'fpr', 'tpr')])
        a = np.array(per_seed)
        pm = a.mean(0)
        sd = a.std(0) if len(a) > 1 else np.zeros(3)
        fmt = lambda i, pct=False: (f'{pm[i] * 100:.0f}% ±{sd[i] * 100:.0f}' if pct else f'{pm[i]:.2f} ±{sd[i]:.2f}')
        lines.append(f'| {t} | {names[m]} | {fmt(0)} | {fmt(1, True)} | {fmt(2, True)} |')
out = '\n'.join(lines)
print(out)
open(os.path.join(HERE, 'results', f'calibration_v2_summary_{snr_show}dB_K{K_SHOW}.md'), 'w', encoding='utf-8').write(out + '\n')
