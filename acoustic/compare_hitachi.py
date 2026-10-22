"""Compare our baseline AUCs with Hitachi's Table 3 (IDs 00/02/04/06 only)."""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Hitachi (Purohit et al. 2019) Table 3: {type: {snr: {id: auc}}}
H = {
 'valve':  {'6': {'00': .68, '02': .66, '04': .64, '06': .70}, '0': {'00': .55, '02': .59, '04': .65, '06': .66}, '-6': {'00': .62, '02': .57, '04': .50, '06': .53}},
 'pump':   {'6': {'00': .84, '02': .45, '04': .99, '06': .94}, '0': {'00': .65, '02': .46, '04': .95, '06': .76}, '-6': {'00': .58, '02': .52, '04': .93, '06': .61}},
 'fan':    {'6': {'00': .75, '02': .99, '04': .92, '06': .99}, '0': {'00': .63, '02': .83, '04': .75, '06': .97}, '-6': {'00': .57, '02': .68, '04': .57, '06': .83}},
 'slider': {'6': {'00': .99, '02': .93, '04': .88, '06': .71}, '0': {'00': .99, '02': .79, '04': .78, '06': .56}, '-6': {'00': .93, '02': .74, '04': .61, '06': .52}},
}

rows = [r for r in csv.DictReader(open(os.path.join(HERE, 'results', 'baseline_results.csv'))) if r['seed'] == '0']
ours = {(r['machine'], r['snr'], r['model_id'][3:]): float(r['auc']) for r in rows}

lines = ['| type | SNR | ours (mean of IDs) | Hitachi (same 4 IDs) | diff |', '|---|---|---|---|---|']
fig, ax = plt.subplots(figsize=(7, 7))
allo, allh = [], []
for t in ['fan', 'pump', 'slider', 'valve']:
    for s in ['6', '0', '-6']:
        o = [ours[(t, s, i)] for i in ['00', '02', '04', '06'] if (t, s, i) in ours]
        h = [H[t][s][i] for i in ['00', '02', '04', '06'] if (t, s, i) in ours]
        if len(o) == 4:
            lines.append(f'| {t} | {s} dB | {np.mean(o):.3f} | {np.mean(h):.3f} | {np.mean(o) - np.mean(h):+.3f} |')
        allo += o
        allh += h
ax.scatter(allh, allo, c='tab:blue')
ax.plot([0.4, 1], [0.4, 1], 'k--', lw=1)
ax.set_xlabel("Hitachi Table 3 AUC (per ID)")
ax.set_ylabel('our reproduction AUC (per ID, 1 seed)')
ax.set_title('Baseline reproduction, per machine ID')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'plots', 'baseline_vs_hitachi.png'), dpi=110)
d = np.array(allo) - np.array(allh)
lines.append('')
lines.append(f'Per-ID cases: {len(d)}; mean diff {d.mean():+.3f}; mean abs diff {np.abs(d).mean():.3f}; '
             f'corr {np.corrcoef(allo, allh)[0, 1]:.3f}')
out = '\n'.join(lines)
print(out)
open(os.path.join(HERE, 'results', 'baseline_vs_hitachi.md'), 'w').write(out + '\n')

print('\nPer-ID detail:')
for t in ['fan', 'pump', 'slider', 'valve']:
    for s in ['6', '0', '-6']:
        print(t, s, ' '.join(f"{i}:{ours.get((t, s, i), float('nan')):.2f}/{H[t][s][i]:.2f}" for i in ['00', '02', '04', '06']))
