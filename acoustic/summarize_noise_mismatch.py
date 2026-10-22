"""Summarise the noise-mismatch experiment (mean over the 4 machines of each type)."""
import os
import csv
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, 'results', 'noise_mismatch_results.csv'))))
model_show = sys.argv[1] if len(sys.argv) > 1 else 'trained@6dB'
SN = ['6', '0', '-6']


def avg(t, c, s, model):
    sel = [r for r in rows if r['machine'] == t and r['calib'] == c and r['test'] == s and r['model'] == model]
    if not sel:
        return None
    return np.mean([[float(r['auc']), float(r['fpr']), float(r['tpr'])] for r in sel], axis=0), len({r['held_out'] for r in sel})


out = [f'Noise mismatch - model {model_show}; rows = noise when the device was calibrated, columns = noise at test time',
       'Each cell: false-alarm rate / faults found  (AUC in brackets). Mean over the machines of the type.', '']
for t in ['fan', 'pump', 'slider', 'valve']:
    first = avg(t, '6', '6', model_show)
    if first is None:
        continue
    out += [f'### {t}  ({first[1]} machines)', '| calibrated at \\ tested at | 6 dB | 0 dB | -6 dB |', '|---|---|---|---|']
    for c in SN + ['mix']:
        cells = []
        for s in SN:
            v = avg(t, c, s, model_show)
            if v is None:
                cells.append('-')
                continue
            m = v[0]
            cells.append(f'{m[1]*100:.0f}% / {m[2]*100:.0f}%  ({m[0]:.2f})')
        label = 'mixed noise (10 clips each)' if c == 'mix' else f'{c} dB'
        out.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    out.append('')

out += ['### Matched vs mismatched (all types, both models), mean over the three types that work (fan, pump, slider)', '',
        '| model | calibration noise = test noise | calibrated quiet (6 dB), tested noisier | calibrated noisy (-6 dB), tested quieter | calibrated on mixed noise |',
        '|---|---|---|---|---|']
for model in ['trained@6dB', 'trained@all']:
    def agg(pairs):
        vals = []
        for t in ['fan', 'pump', 'slider']:
            for c, s in pairs:
                v = avg(t, c, s, model)
                if v is not None:
                    vals.append(v[0])
        if not vals:
            return '-'
        m = np.mean(vals, axis=0)
        return f'{m[1]*100:.0f}% / {m[2]*100:.0f}%  ({m[0]:.2f})'
    out.append(f'| {model} | {agg([("6","6"),("0","0"),("-6","-6")])} | {agg([("6","0"),("6","-6"),("0","-6")])} | '
               f'{agg([("-6","6"),("-6","0"),("0","6")])} | {agg([("mix","6"),("mix","0"),("mix","-6")])} |')
text = '\n'.join(out)
print(text)
open(os.path.join(HERE, 'results', 'noise_mismatch_summary.md'), 'w', encoding='utf-8').write(text + '\n')
