"""
Machine Health Monitor - dashboard (SIMULATION on the laptop, recorded clips).

Tabs
  1. Live clip      : calibrate on a machine, choose a WAV file, get GREEN / RED  (demo_gui.App)
  2. Evaluate       : run the whole test set for one machine and noise level -> AUC, precision, recall, F1, accuracy,
                      false-alarm rate, confusion matrix, ROC curve, score histogram, and what precision would be
                      if faults are rare
  3. Results table  : the final evaluation over all machines, noise levels and 3 seeds (results file)
  4. Model card     : size and cost of the chosen model of each machine type
  5. My recordings  : calibrate on your own healthy recording, test on your own healthy / faulty recordings

Run:  python dashboard_gui.py
Needs the SSD with N:\\dev\\Sonar-acoustic-dataset and N:\\dev\\mimii_features* connected.
"""
import os
import csv
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import features as F
import model_selection as M
import demo as D
import demo_gui as G
import evaluation_lib as E
from own_recording_gui import OwnTab

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_CSV = os.path.join(HERE, 'results', 'final_evaluation_results.csv')
GREEN, RED = '#2e8b57', '#c0392b'


class EvalEngine:
    """Run one full evaluation (same pipeline as final_evaluation.py, seed 0)."""

    def run(self, mtype, held, snr):
        design, mels = D.chosen_design(mtype)
        name, frames, depth, hidden, bott = design
        model, trained = D.get_model(mtype, held, snr, design, mels, False)
        F.FEATURE_ROOT = D.feature_root(mels)
        nh = np.load(F.feature_path(snr, mtype, held, 'normal'), mmap_mode='r')
        ah = np.load(F.feature_path(snr, mtype, held, 'abnormal'), mmap_mode='r')
        k = len(ah)
        pool = nh[k:]
        mu, sd = M.stats(M.stack(pool[:G.K], frames))
        alert = M.tau(M.file_scores(model, M.stack(pool[G.K:2 * G.K], frames), mu, sd))
        sn = M.file_scores(model, M.stack(nh[:k], frames), mu, sd)
        sa = M.file_scores(model, M.stack(ah, frames), mu, sd)
        met = E.full_metrics(sn, sa, alert)
        fpr, tpr = E.roc_points(sn, sa)
        card = E.model_card(model, mels, frames)
        return dict(met=met, sn=sn, sa=sa, alert=alert, roc=(fpr, tpr), card=card, design=name, mels=mels,
                    params=M.n_params(model), mtype=mtype, held=held, snr=snr, trained_now=trained)


class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title('Machine Health Monitor  -  dashboard (simulation on a laptop, recorded clips)')
        root.geometry('1280x820')
        nb = ttk.Notebook(root)
        nb.pack(fill='both', expand=True)
        self.tab_live = ttk.Frame(nb)
        self.tab_eval = ttk.Frame(nb)
        self.tab_res = ttk.Frame(nb)
        self.tab_card = ttk.Frame(nb)
        nb.add(self.tab_live, text='1. Live clip')
        nb.add(self.tab_eval, text='2. Evaluate (all metrics)')
        nb.add(self.tab_res, text='3. Results table')
        nb.add(self.tab_card, text='4. Model card')
        self.tab_own = ttk.Frame(nb)
        nb.add(self.tab_own, text='5. My recordings')
        self.own = OwnTab(root, self.tab_own)
        self.live = G.App(root, parent=self.tab_live)
        self.build_eval()
        self.build_results()
        self.build_card()

    # ------------------------------------------------------------------ tab 2
    def build_eval(self):
        left = ttk.Frame(self.tab_eval, padding=10)
        left.pack(side='left', fill='y')
        right = ttk.Frame(self.tab_eval, padding=10)
        right.pack(side='right', fill='both', expand=True)
        ttk.Label(left, text='Evaluate on the whole test set', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        self.e_type = tk.StringVar(value='fan')
        self.e_held = tk.StringVar(value='id_06')
        self.e_snr = tk.StringVar(value='6')
        for label, var, vals in (('Machine type', self.e_type, F.TYPES), ('New machine (device installed on)', self.e_held, F.IDS),
                                 ('Noise level (dB; also the calibration noise)', self.e_snr, ['6', '0', '-6'])):
            ttk.Label(left, text=label).pack(anchor='w', pady=(6, 0))
            ttk.Combobox(left, textvariable=var, values=vals, state='readonly', width=22).pack(anchor='w')
        self.e_btn = ttk.Button(left, text='Run evaluation', command=self.on_eval)
        self.e_btn.pack(anchor='w', pady=10)
        self.e_status = tk.StringVar(value='Press "Run evaluation". The first time for a case it trains the model (about 20-60 s).')
        ttk.Label(left, textvariable=self.e_status, wraplength=330, foreground='#444').pack(anchor='w')
        self.e_text = tk.Text(left, width=48, height=30, font=('Consolas', 9), wrap='none')
        self.e_text.pack(anchor='w', pady=8)
        self.e_text.insert('1.0', 'Results appear here.')
        self.e_text.config(state='disabled')
        self.e_fig = Figure(figsize=(8.2, 7.2), dpi=100)
        self.ax_roc = self.e_fig.add_subplot(2, 2, 1)
        self.ax_hist = self.e_fig.add_subplot(2, 2, 2)
        self.ax_cm = self.e_fig.add_subplot(2, 2, 3)
        self.ax_info = self.e_fig.add_subplot(2, 2, 4)
        self.e_canvas = FigureCanvasTkAgg(self.e_fig, master=right)
        self.e_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.draw_eval_empty()

    def draw_eval_empty(self):
        for ax in (self.ax_roc, self.ax_hist, self.ax_cm, self.ax_info):
            ax.clear()
        self.ax_roc.set_title('ROC curve')
        self.ax_hist.set_title('Scores: healthy vs faulty')
        self.ax_cm.set_title('Confusion matrix')
        self.ax_info.axis('off')
        self.e_fig.tight_layout()
        self.e_canvas.draw()

    def on_eval(self):
        self.e_btn.config(state='disabled')
        self.e_status.set('Working...')
        t, h, s = self.e_type.get(), self.e_held.get(), self.e_snr.get()

        def work():
            try:
                res = EvalEngine().run(t, h, s)
                self.root.after(0, lambda: self.show_eval(res))
            except Exception as e:  # noqa
                self.root.after(0, lambda: self.eval_fail(str(e)))
        threading.Thread(target=work, daemon=True).start()

    def eval_fail(self, msg):
        self.e_btn.config(state='normal')
        self.e_status.set('Failed: ' + msg)
        messagebox.showerror('Problem', msg)

    def show_eval(self, r):
        m = r['met']
        n_norm, n_abn = m['tn'] + m['fp'], m['tp'] + m['fn']
        lines = [f"Machine: {r['mtype']} {r['held']}   noise {r['snr']} dB",
                 f"Test clips: {n_norm} healthy + {n_abn} faulty",
                 f"Design: {r['design']}  ({r['params']:,} parameters, {r['mels']} mel bands)",
                 f"Alert level (set by calibration): {r['alert']:.3f}", '',
                 f"AUC                  {m['auc']:.3f}   ranking quality (0.5 = chance)",
                 f"Precision            {m['precision']:.3f}   of the alerts, share that were real faults",
                 f"Recall (detection)   {m['recall']:.3f}   of the real faults, share caught",
                 f"F1 score             {m['f1']:.3f}",
                 f"Accuracy             {m['accuracy']:.3f}",
                 f"False-alarm rate     {m['fpr'] * 100:.1f}%",
                 f"Specificity          {m['specificity']:.3f}",
                 f"Balanced accuracy    {m['balanced_accuracy']:.3f}", '',
                 f"Confusion matrix (clips): TP {m['tp']}  FP {m['fp']}  TN {m['tn']}  FN {m['fn']}", '',
                 'Precision if faults are RARE in real use',
                 '(this test set is half faulty; real factories are not):']
        for p in (0.5, 0.1, 0.01):
            lines.append(f"   {p * 100:>4.0f}% of clips faulty -> precision {E.precision_at_prevalence(m['recall'], m['fpr'], p):.2f}")
        lines += ['', 'Seed 0, 30 calibration clips, alert = mean + 3 std.']
        self.e_text.config(state='normal')
        self.e_text.delete('1.0', 'end')
        self.e_text.insert('1.0', '\n'.join(lines))
        self.e_text.config(state='disabled')

        for ax in (self.ax_roc, self.ax_hist, self.ax_cm, self.ax_info):
            ax.clear()
        fpr, tpr = r['roc']
        self.ax_roc.plot(fpr, tpr, color='tab:blue')
        self.ax_roc.plot([0, 1], [0, 1], 'k:', lw=1)
        self.ax_roc.scatter([m['fpr']], [m['recall']], color=RED, zorder=5, label='device operating point')
        self.ax_roc.set_xlabel('false-alarm rate')
        self.ax_roc.set_ylabel('detection rate')
        self.ax_roc.set_title(f"ROC curve  (AUC {m['auc']:.3f})")
        self.ax_roc.legend(loc='lower right', fontsize=8)
        allv = np.r_[r['sn'], r['sa']]
        bins = np.linspace(allv.min(), np.percentile(allv, 99), 40)
        self.ax_hist.hist(np.clip(r['sn'], None, bins[-1]), bins=bins, alpha=0.7, color=GREEN, label='healthy')
        self.ax_hist.hist(np.clip(r['sa'], None, bins[-1]), bins=bins, alpha=0.7, color=RED, label='faulty')
        self.ax_hist.axvline(r['alert'], color='k', linestyle='--', label='alert level')
        self.ax_hist.set_xlabel('how strange the sound is (error score)')
        self.ax_hist.set_ylabel('number of clips')
        self.ax_hist.set_title('Scores: healthy vs faulty')
        self.ax_hist.legend(fontsize=8)
        cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        self.ax_cm.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                self.ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
        self.ax_cm.set_xticks([0, 1])
        self.ax_cm.set_xticklabels(['no alert', 'alert'])
        self.ax_cm.set_yticks([0, 1])
        self.ax_cm.set_yticklabels(['healthy', 'faulty'])
        self.ax_cm.set_xlabel('device said')
        self.ax_cm.set_ylabel('truth')
        self.ax_cm.set_title('Confusion matrix')
        c = r['card']
        self.ax_info.axis('off')
        self.ax_info.text(0, 1, 'Model card (this model)\n\n'
                          f"parameters            {c['params']:,}\n"
                          f"operations per frame  {c['macs']:,} multiply-adds (model)\n"
                          f"size, 32-bit          {c['size_f32_kb']:.1f} KB\n"
                          f"size, INT8 (est.)     {c['size_int8_kb']:.1f} KB\n"
                          f"laptop time/frame     {c['laptop_ms_per_frame']:.3f} ms (NOT the chip)\n"
                          '\nPower and speed on the chip:\nnot measured (no hardware yet).',
                          va='top', family='monospace', fontsize=9)
        self.e_fig.tight_layout()
        self.e_canvas.draw()
        self.e_btn.config(state='normal')
        self.e_status.set('Done.')

    # ------------------------------------------------------------------ tab 3
    def build_results(self):
        top = ttk.Frame(self.tab_res, padding=10)
        top.pack(fill='x')
        ttk.Label(top, text='Final evaluation (all noise levels, 3 seeds). Machines to include:').pack(side='left')
        self.r_subset = tk.StringVar(value='All 4 machines')
        cb = ttk.Combobox(top, textvariable=self.r_subset, state='readonly', width=34,
                          values=['All 4 machines', 'Sealed machine (ID 06) only', 'Tuning machines (ID 00, 02, 04)'])
        cb.pack(side='left', padx=8)
        cb.bind('<<ComboboxSelected>>', lambda e: self.load_results())
        ttk.Button(top, text='Reload file', command=self.load_results).pack(side='left')
        cols = ('type', 'noise', 'machines', 'AUC', 'precision', 'recall', 'F1', 'accuracy', 'false alarms')
        self.r_tree = ttk.Treeview(self.tab_res, columns=cols, show='headings', height=16)
        for c, w in zip(cols, (90, 70, 80, 110, 110, 110, 110, 110, 110)):
            self.r_tree.heading(c, text=c)
            self.r_tree.column(c, width=w, anchor='center')
        self.r_tree.pack(fill='both', expand=True, padx=10)
        self.r_note = tk.StringVar(value='')
        ttk.Label(self.tab_res, textvariable=self.r_note, wraplength=1150, justify='left', foreground='#444').pack(anchor='w', padx=10, pady=8)
        self.load_results()

    def load_results(self):
        self.r_tree.delete(*self.r_tree.get_children())
        if not os.path.exists(FINAL_CSV):
            self.r_note.set('results/final_evaluation_results.csv not found yet. Run:  python final_evaluation.py')
            return
        rows = list(csv.DictReader(open(FINAL_CSV)))
        sub = self.r_subset.get()
        keep = {'All 4 machines': F.IDS, 'Sealed machine (ID 06) only': ['id_06'],
                'Tuning machines (ID 00, 02, 04)': ['id_00', 'id_02', 'id_04']}[sub]
        keys = ('auc', 'precision', 'recall', 'f1', 'accuracy', 'fpr')
        n_cases = 0
        for t in F.TYPES:
            for s in ('6', '0', '-6'):
                sel = [r for r in rows if r['machine'] == t and r['snr'] == s and r['held_out'] in keep]
                if not sel:
                    continue
                seeds = sorted({r['seed'] for r in sel})
                per_seed = np.array([[np.nanmean([float(r[k]) for r in sel if r['seed'] == sd]) for k in keys] for sd in seeds])
                mean, std = per_seed.mean(0), per_seed.std(0)
                fmt = lambda i, pct=False: (f'{mean[i] * 100:.0f}% \u00b1{std[i] * 100:.0f}' if pct else f'{mean[i]:.2f} \u00b1{std[i]:.2f}')
                self.r_tree.insert('', 'end', values=(t, f'{s} dB', len({r['held_out'] for r in sel}), fmt(0), fmt(1), fmt(2), fmt(3), fmt(4), fmt(5, True)))
                n_cases += len(sel)
        self.r_note.set(f'{n_cases} evaluation runs loaded. Each row: mean over the machines, \u00b1 spread over seeds. '
                        'Alert level = mean + 3 std of 30 healthy calibration clips; test set is half healthy, half faulty. '
                        'Design choices were made on machines 00, 02, 04 at 6 dB; ID 06 was sealed, so its row is the fairest. '
                        'Valves do not work (AUC about 0.5).')

    # ------------------------------------------------------------------ tab 4
    def build_card(self):
        frame = ttk.Frame(self.tab_card, padding=10)
        frame.pack(fill='both', expand=True)
        cols = ('type', 'design', 'mel bands', 'parameters', 'model ops/frame', 'size 32-bit', 'size INT8 (est.)',
                'RAM activations', 'RAM features', 'ops/frame (features+model)', 'laptop ms/frame')
        tree = ttk.Treeview(frame, columns=cols, show='headings', height=6)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=105, anchor='center')
        tree.pack(fill='x')
        import csv as _csv
        rows = {r['machine']: r for r in _csv.DictReader(open(os.path.join(HERE, 'results', 'model_selection_choice_final.csv')))}
        for t in F.TYPES:
            r = rows[t]
            mels, frames = int(r['mels']), int(r['frames'])
            model = M.build(mels * frames, r['depth'], int(r['hidden']), int(r['bottleneck']))
            c = E.model_card(model, mels, frames)
            tree.insert('', 'end', values=(t, r['design'], mels, f"{c['params']:,}", f"{c['macs']:,}", f"{c['size_f32_kb']:.1f} KB",
                                           f"{c['size_int8_kb']:.1f} KB", f"{c['ram_activations_bytes']} B", f"{c['ram_features_bytes'] / 1024:.1f} KB",
                                           f"{c['total_ops_per_frame']:,}", f"{c['laptop_ms_per_frame']:.3f}"))
        txt = ('How to read this table\n'
               'MEASURED here: parameters, operations per frame (counted), laptop time per frame (this laptop, NOT the microcontroller).\n'
               'ESTIMATED: INT8 size (about 1 byte per parameter, quantization not done yet), RAM for activations and feature buffers, '
               'operations to compute one feature frame (standard operation counts).\n'
               'NOT MEASURED (needs hardware): real speed on the ESP32-S3, memory actually used, power consumption, noise and energy measurement.\n\n'
               'Time budget: a new frame arrives every 32 ms. At roughly 30,000-50,000 operations per frame (last column), even at 10 clock cycles per operation '
               'the work would take about 2 ms on a 240 MHz chip - a rough estimate only, to be confirmed on the board.\n'
               'Target in the proposal: ROC-AUC above 0.90 - met for fans and sliders at 6 dB (see Results table), not met in heavy noise or for valves.')
        ttk.Label(frame, text=txt, wraplength=1180, justify='left').pack(anchor='w', pady=14)


def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == '__main__':
    main()
