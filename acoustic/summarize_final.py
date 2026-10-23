"""Summarise results/final_evaluation_results.csv: mean over seeds per case, then per type / noise level, for two subsets."""
import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, 'results', 'final_evaluation_results.csv'))))
COLS = ['auc', 'precision', 'recall', 'f1', 'accuracy', 'fpr']


def agg(sel):
    out = {}
    for c in COLS:
        v = [float(r[c]) for r in sel]
        out[c] = np.nanmean(v) if v else float('nan')
    return out


lines = ['# Final evaluation summary', '',
         f'{len(rows)} rows = machine type x noise x held-out machine x 3 seeds. Values are means over seeds and machines.',
         'Sealed machine = ID 06 (never used to choose designs). Tuning machines = IDs 00, 02, 04 (choices made at 6 dB).', '']
for title, ids in [('All 4 machines', {'id_00', 'id_02', 'id_04', 'id_06'}), ('Sealed machine ID 06 only', {'id_06'}),
                   ('Tuning machines 00, 02, 04', {'id_00', 'id_02', 'id_04'})]:
    lines += [f'## {title}', '', '| type | noise | cases | AUC | precision | recall | F1 | accuracy | false-alarm rate |', '|---|---|---|---|---|---|---|---|---|']
    for t in ['fan', 'pump', 'slider', 'valve']:
        for s in ['6', '0', '-6']:
            sel = [r for r in rows if r['machine'] == t and r['snr'] == s and r['held_out'] in ids]
            if not sel:
                continue
            m = agg(sel)
            lines.append(f'| {t} | {s} dB | {len(sel) // 3} | ' + ' | '.join(f'{m[c]:.2f}' for c in COLS) + ' |')
    lines.append('')
open(os.path.join(HERE, 'results', 'final_summary.md'), 'w').write('\n'.join(lines))
print('\n'.join(lines))
