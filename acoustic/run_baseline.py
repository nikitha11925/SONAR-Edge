"""Run the Hitachi-style baseline for one case or the whole 48-case grid (resumable)."""
import os
import csv
import argparse
import multiprocessing as mp
import features as F

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'baseline_results.csv')
FIELDS = ['snr', 'machine', 'model_id', 'seed', 'auc', 'n_train_files', 'n_eval_normal', 'n_eval_abnormal',
          'final_train_loss', 'final_val_loss', 'mean_score_normal', 'mean_score_abnormal', 'seconds']


def done_keys():
    if not os.path.exists(RESULTS):
        return set()
    with open(RESULTS) as f:
        return {(r['snr'], r['machine'], r['model_id'], r['seed']) for r in csv.DictReader(f)}


def worker(job):
    import torch
    torch.set_num_threads(job['threads'])
    import baseline_ae as B
    r = B.run_one(job['snr'], job['machine'], job['model_id'], seed=job['seed'], epochs=job['epochs'], log=lambda m: None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--single', nargs=3, metavar=('SNR', 'TYPE', 'ID'))
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--snrs', nargs='*', default=F.SNRS)
    args = ap.parse_args()

    if args.single:
        cases = [tuple(args.single)]
    else:
        cases = [(s, t, i) for s in args.snrs for t in F.TYPES for i in F.IDS]
    done = done_keys()
    jobs = [dict(snr=s, machine=t, model_id=i, seed=args.seed, epochs=args.epochs, threads=args.threads)
            for s, t, i in cases if (s, t, i, str(args.seed)) not in done]
    print(f'{len(jobs)} jobs to run ({len(cases) - len(jobs)} already done)', flush=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, 'a', newline='') as f, mp.get_context('spawn').Pool(args.workers) as pool:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            w.writerow(r)
            f.flush()
            print(f"[{k}/{len(jobs)}] {r['snr']}dB {r['machine']} {r['model_id']}  AUC={r['auc']:.4f}  ({r['seconds']}s)", flush=True)


if __name__ == '__main__':
    main()
