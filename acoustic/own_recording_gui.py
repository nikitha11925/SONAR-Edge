"""
Tab "5. My recordings" of the dashboard.

  Calibrate on YOUR healthy recording, then
    * "Test results"  sub-tab : whole-set metrics (AUC, recall, false alarms ...) and plots
    * "Single clip"   sub-tab : pick ANY 10-second clip (or a random healthy / faulty one), get GREEN / RED,
                                spectrogram, score against the alert level, and play it
Inputs can be folders of 10-second clips (made by split_recordings.py, recommended) or long recordings.
All logic is in own_recording_test.py (same code as the command line).
"""
import os
import random
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import soundfile as sf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import features as F
import model_selection as M
import demo as D
import own_recording_test as T

GREEN, RED = '#2e8b57', '#c0392b'
AUDIO = [('Audio', '*.wav *.mp3 *.m4a *.mpeg *.flac *.ogg *.aac'), ('all files', '*.*')]
HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(os.path.dirname(HERE), 'own_data', 'clips')


def audio_files(path):
    if not path:
        return []
    if os.path.isfile(path):
        return [path]
    return sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(T.EXTS))


def clip_files(path):
    """Files of a folder of 10-second clips (each file one clip), else []."""
    fs = audio_files(path)
    if os.path.isdir(path or '') and fs:
        try:
            if sf.info(fs[0]).duration <= 11:
                return fs
        except Exception:  # noqa
            pass
    return []


class OwnTab:
    def __init__(self, root, parent):
        self.root = root
        self.ctx = None
        self.k = None
        self.cal_files = set()
        self.healthy_test_files, self.faulty_files = [], []
        self.last_path = None
        left = ttk.Frame(parent, padding=10)
        left.pack(side='left', fill='y')
        right = ttk.Frame(parent, padding=6)
        right.pack(side='right', fill='both', expand=True)

        ttk.Label(left, text='Test with your own recordings', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        ttk.Label(left, text='Machine type (selects the model)').pack(anchor='w', pady=(4, 0))
        self.type_var = tk.StringVar(value='fan')
        ttk.Combobox(left, textvariable=self.type_var, values=F.TYPES, state='readonly', width=22).pack(anchor='w')

        self.paths, self.labels = {}, {}
        for key, title in (('healthy', '1. Healthy, for CALIBRATION (folder of clips or a recording)\n    the first 30 clips set the feature numbers, the next 30 the alert level'),
                           ('healthy_test', '2. A SEPARATE healthy session to test on (optional, fairest)'),
                           ('faulty', '3. Faulty clips / recording')):
            ttk.Label(left, text=title, justify='left').pack(anchor='w', pady=(8, 0))
            row = ttk.Frame(left)
            row.pack(anchor='w')
            ttk.Button(row, text='Folder...', command=lambda k=key: self.pick_folder(k)).pack(side='left')
            ttk.Button(row, text='File...', command=lambda k=key: self.pick_file(k)).pack(side='left', padx=4)
            ttk.Button(row, text='Clear', command=lambda k=key: self.set_path(k, None)).pack(side='left')
            self.labels[key] = tk.StringVar(value='(none)')
            ttk.Label(left, textvariable=self.labels[key], wraplength=340, foreground='#444').pack(anchor='w')
        ttk.Button(left, text='Use my saved clips (sessions 1, 2, faulty)', command=self.use_saved).pack(anchor='w', pady=(8, 0))

        self.btn_run = ttk.Button(left, text='Calibrate and test', command=self.on_run)
        self.btn_run.pack(anchor='w', pady=8)
        self.status = tk.StringVar(value='Choose the clips, then press "Calibrate and test".')
        ttk.Label(left, textvariable=self.status, wraplength=340, foreground='#444').pack(anchor='w')
        self.text = tk.Text(left, width=52, height=15, font=('Consolas', 9), wrap='none')
        self.text.pack(anchor='w', pady=6)
        self.set_text('Results appear here.')

        sub = ttk.Notebook(right)
        sub.pack(fill='both', expand=True)
        self.sub = sub
        t_res, t_clip = ttk.Frame(sub), ttk.Frame(sub)
        sub.add(t_res, text='Test results')
        sub.add(t_clip, text='Single clip')

        self.fig = Figure(figsize=(8, 6.2), dpi=100)
        self.ax_t = self.fig.add_subplot(2, 1, 1)
        self.ax_h = self.fig.add_subplot(2, 1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=t_res)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        bar = ttk.Frame(t_clip, padding=6)
        bar.pack(fill='x')
        self.b_choose = ttk.Button(bar, text='Choose a clip...', command=self.on_choose, state='disabled')
        self.b_h = ttk.Button(bar, text='Random healthy test clip', command=lambda: self.on_random('healthy'), state='disabled')
        self.b_f = ttk.Button(bar, text='Random faulty clip', command=lambda: self.on_random('faulty'), state='disabled')
        self.b_play = ttk.Button(bar, text='Play this clip', command=self.on_play, state='disabled')
        for b in (self.b_choose, self.b_h, self.b_f, self.b_play):
            b.pack(side='left', padx=3)
        self.drop = tk.Label(t_clip, text='DROP an audio file here (WAV, MP3, M4A) to judge it\n(calibrate first; only the first 10 s of the file are judged)',
                             bg='#eef3fb', fg='#244', relief='groove', bd=2, height=3)
        self.drop.pack(fill='x', padx=6, pady=(2, 4))
        self.setup_drop()
        info = ttk.Frame(t_clip, padding=6)
        info.pack(fill='x')
        self.verdict = tk.Label(info, text='-', font=('Segoe UI', 26, 'bold'), width=12, height=2, bg='#dddddd')
        self.verdict.pack(side='left')
        self.detail = tk.StringVar(value='Calibrate first, then pick a clip.')
        ttk.Label(info, textvariable=self.detail, wraplength=700, justify='left').pack(side='left', padx=12)
        self.cfig = Figure(figsize=(8, 4.8), dpi=100)
        self.ax_spec = self.cfig.add_subplot(2, 1, 1)
        self.ax_bar = self.cfig.add_subplot(2, 1, 2)
        self.ccanvas = FigureCanvasTkAgg(self.cfig, master=t_clip)
        self.ccanvas.get_tk_widget().pack(fill='both', expand=True)
        self.draw_empty()

    # ---- helpers
    def set_text(self, s):
        self.text.config(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', s)
        self.text.config(state='disabled')

    def set_path(self, key, p):
        self.paths[key] = p
        self.labels[key].set(p if p else '(none)')

    def pick_file(self, key):
        p = filedialog.askopenfilename(title='Choose a recording', filetypes=AUDIO, initialdir=CLIPS if os.path.isdir(CLIPS) else None)
        if p:
            self.set_path(key, p)

    def pick_folder(self, key):
        p = filedialog.askdirectory(title='Choose a folder', initialdir=CLIPS if os.path.isdir(CLIPS) else None)
        if p:
            self.set_path(key, p)

    def use_saved(self):
        d = {k: os.path.join(CLIPS, v) for k, v in (('healthy', 'healthy_session1'), ('healthy_test', 'healthy_session2'), ('faulty', 'faulty'))}
        missing = [v for v in d.values() if not os.path.isdir(v)]
        if missing:
            messagebox.showinfo('Not found', 'These folders do not exist yet (run split_recordings.py):\n' + '\n'.join(missing))
            return
        for k, v in d.items():
            self.set_path(k, v)

    def draw_empty(self):
        for ax, t in ((self.ax_t, 'Score of every test clip (appears here)'), (self.ax_h, 'Scores: healthy vs faulty')):
            ax.clear()
            ax.set_title(t)
        self.fig.tight_layout()
        self.canvas.draw()
        self.ax_spec.clear()
        self.ax_bar.clear()
        self.ax_spec.set_title('Spectrogram of the clip (appears here)')
        self.ax_bar.set_title('Score against the alert level')
        self.cfig.tight_layout()
        self.ccanvas.draw()

    # ---- calibrate and test
    def on_run(self):
        if not self.paths.get('healthy') or not self.paths.get('faulty'):
            messagebox.showinfo('Missing', 'Choose at least healthy clips (1) and faulty clips (3).')
            return
        for b in (self.btn_run, self.b_choose, self.b_h, self.b_f, self.b_play):
            b.config(state='disabled')
        self.status.set('Working... reading the clips and calibrating (about 30 s to 2 min).')
        a = (self.paths['healthy'], self.paths['faulty'], self.type_var.get(), self.paths.get('healthy_test'))

        def work():
            try:
                r = T.analyse(a[0], a[1], a[2], healthy_test=a[3])
                self.root.after(0, lambda: self.show(r))
            except Exception as e:  # noqa
                msg = str(e)
                self.root.after(0, lambda: self.fail(msg))
        threading.Thread(target=work, daemon=True).start()

    def fail(self, msg):
        self.btn_run.config(state='normal')
        self.status.set('Problem: ' + msg)
        messagebox.showerror('Problem', msg)

    def show(self, r):
        self.ctx, self.k = r['ctx'], r['k']
        hc = clip_files(self.paths['healthy'])
        self.cal_files = set(hc[:2 * self.k])
        if self.paths.get('healthy_test'):
            self.healthy_test_files = clip_files(self.paths['healthy_test'])
        else:
            self.healthy_test_files = hc[2 * self.k:]
        self.faulty_files = clip_files(self.paths['faulty'])
        m, sn, sa, alert = r['met'], r['sn'], r['sa'], r['alert']
        src = 'a separate healthy session' if r['separate_healthy_session'] else 'the leftover clips of the calibration recording (less strict)'
        lines = [f"Model: {r['mtype']}, {r['params']:,} parameters, trained on MIMII machines only",
                 f"Calibration: {r['k']} + {r['k']} clips of 10 s; alert level {alert:.3f}",
                 f"Test: {len(sn)} healthy clips from {src}", f"      {len(sa)} faulty clips", '',
                 f"AUC               {m['auc']:.3f}   (0.5 = chance, 1.0 = perfect)",
                 f"Recall            {m['recall']:.3f}   faults caught",
                 f"Precision         {m['precision']:.3f}",
                 f"False-alarm rate  {m['fpr'] * 100:.1f}%",
                 f"Accuracy          {m['accuracy']:.3f}",
                 f"TP {m['tp']}  FN {m['fn']}  FP {m['fp']}  TN {m['tn']}", '',
                 f"Median score: healthy {np.median(sn):.3f}, faulty {np.median(sa):.3f}"]
        for w in r['warnings']:
            lines += ['', 'WARNING: ' + w]
        lines += ['', 'One machine, one microphone, few clips:', 'a first check, not proof.']
        self.set_text('\n'.join(lines))

        self.ax_t.clear()
        self.ax_h.clear()
        self.ax_t.plot(range(len(sn)), sn, 'o-', color=GREEN, label='healthy test clips')
        self.ax_t.plot(range(len(sn), len(sn) + len(sa)), sa, 'o-', color=RED, label='faulty test clips')
        self.ax_t.axhline(alert, color='k', linestyle='--', label='alert level')
        self.ax_t.set_xlabel('clip number (10 s each)')
        self.ax_t.set_ylabel('error score')
        self.ax_t.set_title('Score of every test clip')
        self.ax_t.legend(fontsize=8)
        allv = np.r_[sn, sa]
        bins = np.linspace(allv.min(), np.percentile(allv, 99), 30)
        self.ax_h.hist(np.clip(sn, None, bins[-1]), bins=bins, alpha=0.7, color=GREEN, label='healthy')
        self.ax_h.hist(np.clip(sa, None, bins[-1]), bins=bins, alpha=0.7, color=RED, label='faulty')
        self.ax_h.axvline(alert, color='k', linestyle='--', label='alert level')
        self.ax_h.set_xlabel('error score')
        self.ax_h.set_ylabel('number of clips')
        self.ax_h.set_title(f"Scores: healthy vs faulty   (AUC {m['auc']:.3f})")
        self.ax_h.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

        have_clips = bool(self.faulty_files)
        self.b_choose.config(state='normal')
        self.b_h.config(state='normal' if self.healthy_test_files else 'disabled')
        self.b_f.config(state='normal' if have_clips else 'disabled')
        hint = '' if (self.healthy_test_files and have_clips) else \
            ' (Random buttons need folders of 10-second clips; use split_recordings.py or "Use my saved clips".)'
        self.status.set('Done. Open the "Single clip" sub-tab on the right to judge individual clips.' + hint)
        self.btn_run.config(state='normal')
        self.sub.select(1)

    # ---- single clip
    def setup_drop(self):
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES
            TkinterDnD._require(self.root)
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind('<<Drop>>', lambda e: self.on_drop(e.data))
        except Exception:  # noqa
            self.drop.config(text='Drag and drop is not available here (pip install tkinterdnd2).\nUse "Choose a clip..." instead.')

    def on_drop(self, data):
        if self.ctx is None:
            messagebox.showinfo('Calibrate first', 'Press "Calibrate and test" first (use the left panel), then drop a file.')
            return
        files = [f for f in self.root.tk.splitlist(data) if os.path.isfile(f)]
        if not files:
            messagebox.showinfo('Drop', 'Drop an audio file (not a folder).')
            return
        self.judge(files[0])

    def on_choose(self):
        p = filedialog.askopenfilename(title='Choose a clip', filetypes=AUDIO, initialdir=CLIPS if os.path.isdir(CLIPS) else None)
        if p:
            self.judge(p)

    def on_random(self, kind):
        files = self.healthy_test_files if kind == 'healthy' else self.faulty_files
        if files:
            self.judge(random.choice(files))

    def judge(self, path):
        try:
            y, sr = T.read_audio(path)
            clips = T.cut_clips(y, sr)
            if not clips:
                raise ValueError('This file is shorter than 10 seconds.')
            clip = clips[0]
            lm = F.logmel(clip, n_mels=self.ctx['mels'])
            s = float(M.file_scores(self.ctx['model'], M.stack([lm], self.ctx['frames']), self.ctx['mu'], self.ctx['sd'])[0])
        except Exception as e:  # noqa
            messagebox.showerror('Problem', f'Could not judge this file: {e}')
            return
        self.last_path = path
        alert = self.ctx['alert']
        red = s > alert
        self.verdict.config(text='RED\nALERT' if red else 'GREEN\nNORMAL', bg=RED if red else GREEN, fg='white')
        notes = [f'{os.path.basename(path)}', f'Score {s:.3f}  vs alert level {alert:.3f}']
        if len(clips) > 1:
            notes.append(f'(file is longer than 10 s; only the first 10 s were judged)')
        folder = os.path.basename(os.path.dirname(path))
        if path in self.cal_files:
            notes.append('Note: this healthy clip was used for CALIBRATION. Pick another for a fair test.')
        notes.append(f'Folder: {folder}')
        self.detail.set('\n'.join(notes))
        self.ax_spec.clear()
        self.ax_bar.clear()
        self.ax_spec.imshow(lm, origin='lower', aspect='auto', cmap='magma', vmin=-80, vmax=-5, extent=[0, 10, 0, lm.shape[0]])
        self.ax_spec.set_title(os.path.basename(path))
        self.ax_spec.set_xlabel('time (s)')
        self.ax_spec.set_ylabel('mel band (low to high pitch)')
        self.ax_bar.barh([0], [s], color=RED if red else GREEN, height=0.5)
        self.ax_bar.axvline(alert, color='k', linestyle='--')
        self.ax_bar.text(alert, 0.42, ' alert level', va='bottom')
        self.ax_bar.set_yticks([])
        self.ax_bar.set_xlim(0, max(s, alert) * 1.25)
        self.ax_bar.set_xlabel('how strange the sound is (error score)')
        self.ax_bar.set_title('Score against the alert level')
        self.cfig.tight_layout()
        self.ccanvas.draw()
        self.b_play.config(state='normal')

    def on_play(self):
        if not self.last_path:
            return
        p = self.last_path

        def work():
            try:
                y, sr = T.read_audio(p)
                clip = T.cut_clips(y, sr)[0]
                import soundfile as _sf
                import winsound
                tmp = os.path.join(D.OUT_DIR, '_play.wav')
                _sf.write(tmp, np.clip(clip, -1, 1), 16000, subtype='PCM_16')
                winsound.PlaySound(tmp, winsound.SND_FILENAME)
            except Exception as ex:  # noqa
                self.root.after(0, lambda: messagebox.showinfo('Sound', f'Could not play: {ex}'))
        threading.Thread(target=work, daemon=True).start()
