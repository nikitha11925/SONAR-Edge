"""
Apply the pre-set selection rule to the tuning results.

Rule (fixed before looking at results): per machine type, among designs whose mean tuning AUC is within
TOL = 0.02 of the best (0.02 ~ the seed-to-seed spread seen earlier), choose the SMALLEST model (fewest parameters).

  python summarize_selection.py stage1   -> uses the 64-band designs, writes results/model_selection_choice.csv
  python summarize_selection.py stage2   -> adds the 32/128-band variants of each stage-1 winner,
                                            writes results/model_selection_choice_final.csv
"""
import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TUNE = os.path.join(RES, 'model_selection_tune.csv')
TOL = 0.02
TYPES = ['fan', 'pump', 'slider', 'valve']
mode = sys.argv[1] if len(sys.argv) > 1 else 'stage1'

rows = list(csv.DictReader(open(TUNE)))
table = {}
for r in rows:
    key = (r['machine'], r['design'])
    table.setdefault(key, []).append(r)

summary = {}
for (mt, des), rs in table.items():
    folds = sorted({r['held_out'] for r in rs})
    per_fold = [np.mean([[float(r['auc']), float(r['fpr']), float(r['tpr'])] for r in rs if r['held_out'] == f], axis=0) for f in folds]
    m = np.mean(per_fold, axis=0)
    summary[(mt, des)] = dict(auc=m[0], fpr=m[1], tpr=m[2], params=int(rs[0]['params']), mels=int(rs[0]['mels']),
                              frames=int(rs[0]['frames']), depth=rs[0]['depth'], hidden=int(rs[0]['hidden']),
                              bottleneck=int(rs[0]['bottleneck']), n_folds=len(folds))


def choose(cands):
    best = max(c['auc'] for c in cands)
    ok = [c for c in cands if c['auc'] >= best - TOL]
    return min(ok, key=lambda c: (c['params'], -c['auc'])), best


lines = []
choices = []
if mode == 'stage1':
    out_csv = os.path.join(RES, 'model_selection_choice.csv')
    lines.append('Stage 1: model designs at 64 mel bands, tuning machines only (IDs 00/02/04 held out in turn, 6 dB, full calibration, 30 clips)')
    for mt in TYPES:
        cands = [dict(name=des, **v) for (m, des), v in summary.items() if m == mt and v['mels'] == 64 and '_m' not in des]
        if not cands:
            continue
        ch, best = choose(cands)
        lines += ['', f'### {mt}   (best mean AUC {best:.3f}; rule: smallest model within {TOL} -> chosen = {ch["name"]})',
                  '| design | parameters | mean AUC | false alarms | faults found |', '|---|---|---|---|---|']
        for c in sorted(cands, key=lambda c: -c['auc']):
            mark = '  <-- chosen' if c['name'] == ch['name'] else ''
            lines.append(f"| {c['name']} | {c['params']:,} | {c['auc']:.3f} | {c['fpr']*100:.0f}% | {c['tpr']*100:.0f}%{mark} |")
        if best < 0.6:
            lines.append(f'(No design works for {mt}: best AUC {best:.2f}. The choice is arbitrary; the smallest model within the band was taken.)')
        choices.append(dict(machine=mt, design=ch['name'], mels=64, frames=ch['frames'], depth=ch['depth'], hidden=ch['hidden'],
                            bottleneck=ch['bottleneck'], params=ch['params'], mean_auc=round(ch['auc'], 4)))
else:
    out_csv = os.path.join(RES, 'model_selection_choice_final.csv')
    st1 = {r['machine']: r for r in csv.DictReader(open(os.path.join(RES, 'model_selection_choice.csv')))}
    lines.append('Stage 2: the stage-1 winner of each type with 32 / 64 / 128 mel bands (same rule)')
    for mt in TYPES:
        if mt not in st1:
            continue
        base = st1[mt]['design']
        cands = []
        for (m, des), v in summary.items():
            if m == mt and (des == base or des == base + '_m32' or des == base + '_m128'):
                cands.append(dict(name=des, **v))
        ch, best = choose(cands)
        lines += ['', f'### {mt}   (structure {base}; chosen = {ch["name"]})', '| design | mel bands | parameters | mean AUC | false alarms | faults found |', '|---|---|---|---|---|---|']
        for c in sorted(cands, key=lambda c: c['mels']):
            mark = '  <-- chosen' if c['name'] == ch['name'] else ''
            lines.append(f"| {c['name']} | {c['mels']} | {c['params']:,} | {c['auc']:.3f} | {c['fpr']*100:.0f}% | {c['tpr']*100:.0f}%{mark} |")
        choices.append(dict(machine=mt, design=base, mels=ch['mels'], frames=ch['frames'], depth=ch['depth'], hidden=ch['hidden'],
                            bottleneck=ch['bottleneck'], params=ch['params'], mean_auc=round(ch['auc'], 4)))

with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, ['machine', 'design', 'mels', 'frames', 'depth', 'hidden', 'bottleneck', 'params', 'mean_auc'])
    w.writeheader()
    w.writerows(choices)
text = '\n'.join(lines)
print(text)
open(os.path.join(RES, f'model_selection_{mode}_summary.md'), 'w', encoding='utf-8').write(text + '\n')
