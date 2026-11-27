"""
Alert-level comparison dashboard (separate from dashboard_gui.py; changes nothing there).

Same model, same calibration clips, three ways to set the alert level from the 30 alert-calibration scores:
  1. mean + k*std                       (current rule)
  2. median + k*MAD                     (robust: one odd clip barely moves it; MAD scaled by 1.4826)
  3. drop up to N outliers, then mean + k*std
     (outlier = score above median + k*MAD; if MORE than N clips are high they are all kept,
      because that is real variation, not a one-off)

Sources
  * MIMII dataset : machine type / machine ID / noise level (needs the SSD feature files, like tab 2)
  * My folders    : calibration folder (first 30 clips -> feature numbers, next 30 -> alert level),
                    optional separate healthy test folder, faulty folder (like tab 5)

Only saved models are loaded (acoustic/demo_models); nothing is trained or overwritten.
Run:  python acoustic/alert_compare_gui.py
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import features as F
import model_selection as M
import demo as D
import evaluation_lib as E
import own_recording_test as T

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(os.path.dirname(HERE), 'own_data', 'clips')
GREEN, RED = '#2e8b57', '#c0392b'
COLORS = {'mean + k*std (current)': 'black', 'median + k*MAD': '#1f6fb2', 'drop outliers, then mean + k*std': '#d07a00'}


# ---------------------------------------------------------------- alert rules
def mad(s):
    return 1.4826 * float(np.median(np.abs(s - np.median(s)))) + 1e-9


def alert_mean_std(s, k):
    return float(s.mean() + k * s.std())


def alert_median_mad(s, k):
    return float(np.median(s) + k * mad(s))


def alert_drop(s, k, max_drop):
    high = s > np.median(s) + k * mad(s)
    n = int(high.sum())
    if 1 <= n <= max_drop:
        return alert_mean_std(s[~high], k), np.flatnonzero(high)
    return alert_mean_std(s, k), np.array([], dtype=int)


# ---------------------------------------------------------------- data
def load_saved_model(mtype, held, snr):
    design, mels = D.chosen_design(mtype)
    name, frames, depth, hidden, bott = design
    path = os.path.join(D.MODEL_DIR, f'{mtype}_{held}_{snr}dB.pt')
    if not os.path.exists(path):
        raise ValueError(f'No saved model {os.path.basename(path)}. This tool never trains; '
                         'open that case once in the main dashboard (tab 2) first.')
    model = M.build(mels * frames, depth, int(hidden), int(bott))
    model.load_state_dict(torch.load(path))
    return model, frames, mels


def run_mimii(mtype, held, snr):
    """Same split as dashboard tab 2: test = first k normal + all abnormal; calibration from the rest."""
    model, frames, mels = load_saved_model(mtype, held, snr)
    F.FEATURE_ROOT = D.feature_root(mels)
    nh = np.load(F.feature_path(snr, mtype, held, 'normal'), mmap_mode='r')
    ah = np.load(F.feature_path(snr, mtype, held, 'abnormal'), mmap_mode='r')
    k = len(ah)
    pool = nh[k:]
    K = D.K
    mu, sd = M.stats(M.stack(pool[:K], frames))
    cal = M.file_scores(model, M.stack(pool[K:2 * K], frames), mu, sd)
    sn = M.file_scores(model, M.stack(nh[:k], frames), mu, sd)
    sa = M.file_scores(model, M.stack(ah, frames), mu, sd)
    names = [f'normal clip {k + K + i + 1}' for i in range(K)]
    return dict(title=f'{mtype} {held}, {snr} dB (MIMII)', cal=np.asarray(cal), names=names, sn=sn, sa=sa)


def clips_with_names(path):
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(T.EXTS))
    out = []
    for f in files:
        cs = T.cut_clips(*T.read_audio(f))
        for j, c in enumerate(cs):
            out.append((os.path.basename(f) + (f' [{j + 1}]' if len(cs) > 1 else ''), c))
    return out


def run_folders(mtype, healthy, faulty, healthy_test):
    design, mels = D.chosen_design(mtype)
    model, frames, mels = load_saved_model(mtype, 'id_06', '6')     # same model as tab 5 / the board
    K = D.K
    hc = clips_with_names(healthy)
    htc = clips_with_names(healthy_test) if healthy_test else None
    fc = clips_with_names(faulty)
    need = 2 * K + (0 if htc else 5)
    if len(hc) < need:
        raise ValueError(f'The calibration folder needs at least {need} clips of 10 s (it has {len(hc)}).')
    if len(fc) < 1:
        raise ValueError('No faulty clips found.')
    lm = lambda cs: [F.logmel(c, n_mels=mels) for _, c in cs]
    lh = lm(hc)
    mu, sd = M.stats(M.stack(lh[:K], frames))
    cal = M.file_scores(model, M.stack(lh[K:2 * K], frames), mu, sd)
    test_h = lm(htc) if htc else lh[2 * K:]
    sn = M.file_scores(model, M.stack(test_h, frames), mu, sd)
    sa = M.file_scores(model, M.stack(lm(fc), frames), mu, sd)
    names = [n for n, _ in hc[K:2 * K]]
    return dict(title=f'{mtype}, my folders (model fan id_06 6 dB)' if mtype == 'fan' else f'{mtype}, my folders',
                cal=np.asarray(cal), names=names, sn=sn, sa=sa)


# ---------------------------------------------------------------- GUI
class App:
    def __init__(self, root):
        self.root = root
        root.title('Alert level comparison  -  mean+3std vs median+MAD vs drop outliers')
        root.geometry('1300x840')
        left = ttk.Frame(root, padding=10)
        left.pack(side='left', fill='y')
        right = ttk.Frame(root, padding=6)
        right.pack(side='right', fill='both', expand=True)

        ttk.Label(left, text='Compare alert-level rules', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        self.src = tk.StringVar(value='mimii')
        ttk.Radiobutton(left, text='MIMII dataset', variable=self.src, value='mimii').pack(anchor='w', pady=(6, 0))
        self.type_var, self.id_var, self.snr_var = tk.StringVar(value='fan'), tk.StringVar(value='id_00'), tk.StringVar(value='6')
        row = ttk.Frame(left)
        row.pack(anchor='w', padx=18)
        for lab, var, vals, w in (('type', self.type_var, F.TYPES, 8), ('machine', self.id_var, F.IDS, 7), ('noise dB', self.snr_var, ['6', '0', '-6'], 4)):
            ttk.Label(row, text=lab).pack(side='left')
            ttk.Combobox(row, textvariable=var, values=vals, state='readonly', width=w).pack(side='left', padx=(2, 8))

        ttk.Radiobutton(left, text='My folders of clips (machine type above selects the model)', variable=self.src,
                        value='folders').pack(anchor='w', pady=(10, 0))
        self.paths, self.labels = {}, {}
        for key, title in (('healthy', 'Healthy for calibration (first 30 = feature numbers, next 30 = alert level)'),
                           ('healthy_test', 'Separate healthy test folder (optional)'), ('faulty', 'Faulty clips')):
            ttk.Label(left, text=title, wraplength=360).pack(anchor='w', padx=18, pady=(4, 0))
            r = ttk.Frame(left)
            r.pack(anchor='w', padx=18)
            ttk.Button(r, text='Folder...', command=lambda k=key: self.pick(k)).pack(side='left')
            ttk.Button(r, text='Clear', command=lambda k=key: self.set_path(k, None)).pack(side='left', padx=4)
            self.labels[key] = tk.StringVar(value='(none)')
            ttk.Label(left, textvariable=self.labels[key], wraplength=360, foreground='#444').pack(anchor='w', padx=18)
        ttk.Button(left, text='Use my saved clips (sessions 1, 2, faulty)', command=self.use_saved).pack(anchor='w', padx=18, pady=4)

        par = ttk.Frame(left)
        par.pack(anchor='w', pady=(10, 0))
        ttk.Label(par, text='k (how many spreads above)').pack(side='left')
        self.k_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(par, from_=1, to=6, increment=0.5, textvariable=self.k_var, width=5).pack(side='left', padx=4)
        ttk.Label(par, text='  max clips to drop').pack(side='left')
        self.drop_var = tk.IntVar(value=2)
        ttk.Spinbox(par, from_=0, to=5, increment=1, textvariable=self.drop_var, width=4).pack(side='left', padx=4)

        self.btn = ttk.Button(left, text='Run comparison', command=self.on_run)
        self.btn.pack(anchor='w', pady=8)
        self.status = tk.StringVar(value='Choose a source and press "Run comparison".')
        ttk.Label(left, textvariable=self.status, wraplength=380, foreground='#444').pack(anchor='w')
        self.text = tk.Text(left, width=58, height=20, font=('Consolas', 9), wrap='none')
        self.text.pack(anchor='w', pady=6)

        self.fig = Figure(figsize=(8.4, 7.4), dpi=100)
        self.ax_c = self.fig.add_subplot(2, 1, 1)
        self.ax_h = self.fig.add_subplot(2, 1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.last = None
        for v in (self.k_var, self.drop_var):
            v.trace_add('write', lambda *a: self.redraw())

    def set_path(self, key, p):
        self.paths[key] = p
        self.labels[key].set(p or '(none)')

    def pick(self, key):
        p = filedialog.askdirectory(initialdir=CLIPS if os.path.isdir(CLIPS) else None)
        if p:
            self.set_path(key, p)
            self.src.set('folders')

    def use_saved(self):
        for k, v in (('healthy', 'healthy_session1'), ('healthy_test', 'healthy_session2'), ('faulty', 'faulty')):
            self.set_path(k, os.path.join(CLIPS, v))
        self.src.set('folders')

    def on_run(self):
        src, t = self.src.get(), self.type_var.get()
        if src == 'folders' and (not self.paths.get('healthy') or not self.paths.get('faulty')):
            messagebox.showinfo('Missing', 'Choose the calibration folder and the faulty folder.')
            return
        self.btn.config(state='disabled')
        self.status.set('Working... (MIMII: a few seconds; folders: about 30 s to 2 min)')
        args = (t, self.id_var.get(), self.snr_var.get()) if src == 'mimii' else \
            (t, self.paths['healthy'], self.paths['faulty'], self.paths.get('healthy_test'))

        def work():
            try:
                r = run_mimii(*args) if src == 'mimii' else run_folders(*args)
                self.root.after(0, lambda: self.done(r))
            except Exception as e:  # noqa
                msg = str(e)
                self.root.after(0, lambda: self.fail(msg))
        threading.Thread(target=work, daemon=True).start()

    def fail(self, msg):
        self.btn.config(state='normal')
        self.status.set('Problem: ' + msg)
        messagebox.showerror('Problem', msg)

    def done(self, r):
        self.btn.config(state='normal')
        self.status.set('Done. Change k or "max clips to drop" to see the effect instantly.')
        self.last = r
        self.redraw()

    def redraw(self):
        r = self.last
        if r is None:
            return
        try:
            k, nd = float(self.k_var.get()), int(self.drop_var.get())
        except (tk.TclError, ValueError):
            return
        cal, sn, sa = r['cal'], r['sn'], r['sa']
        a_drop, dropped = alert_drop(cal, k, nd)
        alerts = {'mean + k*std (current)': alert_mean_std(cal, k), 'median + k*MAD': alert_median_mad(cal, k),
                  'drop outliers, then mean + k*std': a_drop}
        lines = [r['title'], f'Alert-calibration clips: {len(cal)}   test: {len(sn)} healthy, {len(sa)} faulty', '',
                 f'Calibration scores: min {cal.min():.3f}  median {np.median(cal):.3f}  max {cal.max():.3f}',
                 f'                    mean {cal.mean():.3f}  std {cal.std():.3f}  MAD {mad(cal):.3f}', '',
                 f'{"rule (k = " + f"{k:g}" + ")":<34}{"alert":>7}{"false":>7}{"caught":>8}',
                 f'{"":<34}{"":>7}{"alarms":>7}{"faults":>8}']
        auc = None
        for name, a in alerts.items():
            m = E.full_metrics(sn, sa, a)
            auc = m['auc']
            lines.append(f'{name:<34}{a:>7.3f}{m["fpr"] * 100:>6.1f}%{m["recall"] * 100:>7.1f}%')
        lines += ['', f'AUC {auc:.3f} (the same for every rule: the rule only moves the line)', '']
        if len(dropped):
            lines.append(f'Dropped {len(dropped)} clip(s) as one-off highs:')
            lines += [f'   {r["names"][i]}  score {cal[i]:.3f}' for i in dropped]
        else:
            n_high = int((cal > np.median(cal) + k * mad(cal)).sum())
            lines.append('No clips dropped: ' + ('none are unusually high.' if n_high == 0 else
                                                  f'{n_high} are high (more than {nd}), so they are kept as real variation.'))
        top = np.argsort(cal)[::-1][:5]
        lines += ['', 'Highest calibration clips:'] + [f'   {r["names"][i]}  {cal[i]:.3f}' for i in top]
        self.text.config(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', '\n'.join(lines))
        self.text.config(state='disabled')

        self.ax_c.clear()
        self.ax_h.clear()
        x = np.arange(1, len(cal) + 1)
        keep = np.ones(len(cal), bool)
        keep[dropped] = False
        self.ax_c.plot(x[keep], cal[keep], 'o', color=GREEN, label='alert-calibration clips (healthy)')
        if len(dropped):
            self.ax_c.plot(x[~keep], cal[~keep], 'x', color=RED, ms=10, mew=2, label='dropped as one-off highs')
        for name, a in alerts.items():
            self.ax_c.axhline(a, color=COLORS[name], linestyle='--', label=f'{name}: {a:.3f}')
        self.ax_c.set_xlabel('calibration clip')
        self.ax_c.set_ylabel('error score')
        self.ax_c.set_title('The 30 clips that set the alert level, and where each rule puts the line')
        self.ax_c.legend(fontsize=8)
        allv = np.r_[sn, sa, cal]
        bins = np.linspace(allv.min(), np.percentile(allv, 99), 40)
        self.ax_h.hist(np.clip(sn, None, bins[-1]), bins=bins, alpha=0.7, color=GREEN, label='healthy test')
        self.ax_h.hist(np.clip(sa, None, bins[-1]), bins=bins, alpha=0.7, color=RED, label='faulty test')
        for name, a in alerts.items():
            self.ax_h.axvline(min(a, bins[-1]), color=COLORS[name], linestyle='--', lw=2)
        self.ax_h.set_xlabel('error score')
        self.ax_h.set_ylabel('number of clips')
        self.ax_h.set_title('Test clips and the three alert levels (left of a line = GREEN, right = RED)')
        self.ax_h.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
