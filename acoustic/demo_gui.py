"""
Machine Health Monitor - small interface (SIMULATION on the laptop).

  1. Choose the machine type and the machine the device is "installed on", press  Calibrate device.
     (The device learns that machine's normal levels from healthy clips and sets its own alert level.
      The model was trained only on healthy sound of the OTHER machines of the same type.)
  2. Press  Choose WAV file...  (or a random healthy / faulty clip) -> verdict GREEN / RED, score vs alert level,
     and the clip's spectrogram.

Run:  python demo_gui.py
Needs the SSD with N:\\dev\\Sonar-acoustic-dataset and N:\\dev\\mimii_features* connected.
"""
import os
import re
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import soundfile as sf
import librosa
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import features as F
import model_selection as M
import demo as D

K = D.K
SNR = '6'
GREEN, RED = '#2e8b57', '#c0392b'


class Engine:
    """All the logic, no window: prepare (train/load + calibrate) and judge one WAV file."""

    def __init__(self):
        self.ready = False

    def prepare(self, mtype, held, snr=SNR):
        design, mels = D.chosen_design(mtype)
        name, frames, depth, hidden, bott = design
        model, trained = D.get_model(mtype, held, snr, design, mels, False)
        F.FEATURE_ROOT = D.feature_root(mels)
        nh = F.load_logmels(snr, mtype, held, 'normal')
        self.abn_files = F.list_wavs(snr, mtype, held, 'abnormal')
        self.norm_files = F.list_wavs(snr, mtype, held, 'normal')
        k = len(self.abn_files)
        pool = nh[k:]
        self.mu, self.sd = M.stats(M.stack(pool[:K], frames))
        self.alert = M.tau(M.file_scores(model, M.stack(pool[K:2 * K], frames), self.mu, self.sd))
        self.calib_names = {os.path.basename(p) for p in self.norm_files[k:k + 2 * K]}
        self.calib_range = (os.path.basename(self.norm_files[k]), os.path.basename(self.norm_files[k + 2 * K - 1]))
        self.model, self.frames, self.mels = model, frames, mels
        self.design_name, self.n_params = name, M.n_params(model)
        self.mtype, self.held, self.trained_now = mtype, held, trained
        self.eval_normals = self.norm_files[:k]
        self.ready = True

    def judge(self, wav_path):
        y, sr = sf.read(wav_path, dtype='float32', always_2d=True)
        y = y[:, 0]
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        lm = F.logmel(y, n_mels=self.mels)
        vec = F.stack_frames(lm, frames=self.frames)[None]
        score = float(M.file_scores(self.model, vec, self.mu, self.sd)[0])
        low = wav_path.lower().replace('\\', '/')
        folder_truth = 'faulty' if '/abnormal/' in low else ('healthy' if '/normal/' in low else 'unknown')
        m = re.search(r'/(fan|pump|slider|valve)/(id_\d\d)/', low)
        clip_machine = (m.group(1), m.group(2)) if m else None
        other = clip_machine is not None and clip_machine != (self.mtype, self.held)
        return dict(score=score, alert=self.alert, flagged=score > self.alert, logmel=lm, seconds=len(y) / 16000,
                    folder_truth=folder_truth, clip_machine=clip_machine, other_machine=other,
                    used_for_calibration=(not other) and os.path.basename(wav_path) in self.calib_names
                    and folder_truth == 'healthy')


class App:
    def __init__(self, root):
        self.root = root
        self.engine = Engine()
        root.title('Machine Health Monitor  -  simulation (recorded clips, laptop)')
        root.geometry('1180x760')
        left = ttk.Frame(root, padding=10)
        left.pack(side='left', fill='y')
        right = ttk.Frame(root, padding=10)
        right.pack(side='right', fill='both', expand=True)

        ttk.Label(left, text='1. Choose the machine', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        ttk.Label(left, text='Machine type').pack(anchor='w', pady=(6, 0))
        self.type_var = tk.StringVar(value='fan')
        ttk.Combobox(left, textvariable=self.type_var, values=F.TYPES, state='readonly', width=18).pack(anchor='w')
        ttk.Label(left, text='Machine the device is installed on').pack(anchor='w', pady=(6, 0))
        self.held_var = tk.StringVar(value='id_06')
        ttk.Combobox(left, textvariable=self.held_var, values=F.IDS, state='readonly', width=18).pack(anchor='w')
        ttk.Label(left, text='2. Calibrate', font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(14, 0))
        self.btn_cal = ttk.Button(left, text='Calibrate device', command=self.on_calibrate)
        self.btn_cal.pack(anchor='w', pady=4)
        self.status = tk.StringVar(value='Not calibrated yet.')
        ttk.Label(left, textvariable=self.status, wraplength=300, foreground='#444').pack(anchor='w')

        ttk.Label(left, text='3. Give it a recording', font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(14, 0))
        self.btn_file = ttk.Button(left, text='Choose WAV file...', command=self.on_choose, state='disabled')
        self.btn_file.pack(anchor='w', pady=2)
        self.btn_h = ttk.Button(left, text='Random healthy clip', command=lambda: self.on_random('healthy'), state='disabled')
        self.btn_h.pack(anchor='w', pady=2)
        self.btn_f = ttk.Button(left, text='Random faulty clip', command=lambda: self.on_random('faulty'), state='disabled')
        self.btn_f.pack(anchor='w', pady=2)
        self.btn_play = ttk.Button(left, text='Play this clip', command=self.on_play, state='disabled')
        self.btn_play.pack(anchor='w', pady=2)

        self.verdict = tk.Label(left, text='-', font=('Segoe UI', 30, 'bold'), width=14, height=2, bg='#dddddd')
        self.verdict.pack(anchor='w', pady=(16, 4))
        self.detail = tk.StringVar(value='')
        ttk.Label(left, textvariable=self.detail, wraplength=300, justify='left').pack(anchor='w')
        self.note = tk.StringVar(value='')
        ttk.Label(left, textvariable=self.note, wraplength=300, justify='left', foreground='#8a4b00').pack(anchor='w', pady=(6, 0))
        ttk.Label(left, text='Simulation on a laptop with recorded clips.\nThe folder name is shown only for reference;\nthe device itself does not see it.',
                  foreground='#666').pack(side='bottom', anchor='w')

        self.fig = Figure(figsize=(7.5, 5.2), dpi=100)
        self.ax_spec = self.fig.add_subplot(2, 1, 1)
        self.ax_bar = self.fig.add_subplot(2, 1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        cols = ('clip', 'score', 'alert', 'verdict', 'folder says')
        self.tree = ttk.Treeview(right, columns=cols, show='headings', height=6)
        for c, w in zip(cols, (300, 70, 70, 90, 90)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w)
        self.tree.pack(fill='x', pady=(6, 0))
        self.last_path = None
        self.draw_empty()

    # ----- drawing -----
    def draw_empty(self):
        self.ax_spec.clear()
        self.ax_bar.clear()
        self.ax_spec.set_title('Spectrogram of the clip (appears here)')
        self.ax_bar.set_title('Score against the alert level')
        self.fig.tight_layout()
        self.canvas.draw()

    def draw_result(self, res, name):
        self.ax_spec.clear()
        self.ax_bar.clear()
        lm = res['logmel']
        self.ax_spec.imshow(lm, origin='lower', aspect='auto', cmap='magma', vmin=-80, vmax=-5,
                            extent=[0, res['seconds'], 0, lm.shape[0]])
        self.ax_spec.set_title(f'{name}   ({res["seconds"]:.0f} s)')
        self.ax_spec.set_xlabel('time (s)')
        self.ax_spec.set_ylabel('mel band (low to high pitch)')
        col = RED if res['flagged'] else GREEN
        self.ax_bar.barh([0], [res['score']], color=col, height=0.5)
        self.ax_bar.axvline(res['alert'], color='k', linestyle='--')
        self.ax_bar.text(res['alert'], 0.42, ' alert level', va='bottom')
        self.ax_bar.set_yticks([])
        self.ax_bar.set_xlim(0, max(res['score'], res['alert']) * 1.25)
        self.ax_bar.set_xlabel('how strange the sound is (error score)')
        self.ax_bar.set_title('Score against the alert level')
        self.fig.tight_layout()
        self.canvas.draw()

    # ----- actions -----
    def set_busy(self, text):
        self.status.set(text)
        self.root.update_idletasks()

    def on_calibrate(self):
        self.btn_cal.config(state='disabled')
        for b in (self.btn_file, self.btn_h, self.btn_f, self.btn_play):
            b.config(state='disabled')
        self.set_busy('Working... (the first time for a machine type the model is trained, about 20 seconds)')
        t, h = self.type_var.get(), self.held_var.get()

        def work():
            try:
                self.engine.prepare(t, h)
                self.root.after(0, self.after_calibrate)
            except Exception as e:  # noqa
                self.root.after(0, lambda: self.fail(f'Calibration failed: {e}'))
        threading.Thread(target=work, daemon=True).start()

    def after_calibrate(self):
        e = self.engine
        self.status.set(f'Ready. Model "{e.design_name}" ({e.n_params:,} parameters), trained on the OTHER {e.mtype}s only.\n'
                        f'Calibrated on {2 * K} healthy clips of {e.held} ({2 * K * 10 // 60} min of sound): '
                        f'alert level = {e.alert:.3f}.\n'
                        f'(Calibration used healthy clips {e.calib_range[0]} to {e.calib_range[1]}.)')
        self.btn_cal.config(state='normal')
        for b in (self.btn_file, self.btn_h, self.btn_f):
            b.config(state='normal')
        self.verdict.config(text='-', bg='#dddddd')
        self.detail.set('')
        self.note.set('')
        self.draw_empty()

    def fail(self, msg):
        self.btn_cal.config(state='normal')
        messagebox.showerror('Problem', msg)
        self.status.set(msg)

    def on_choose(self):
        e = self.engine
        start = rf'N:\dev\Sonar-acoustic-dataset\{SNR}_dB_{e.mtype}\{e.mtype}\{e.held}'
        path = filedialog.askopenfilename(title='Choose a WAV file', initialdir=start if os.path.isdir(start) else None,
                                          filetypes=[('WAV audio', '*.wav'), ('all files', '*.*')])
        if path:
            self.judge_path(path)

    def on_random(self, kind):
        e = self.engine
        files = e.eval_normals if kind == 'healthy' else e.abn_files
        self.judge_path(files[np.random.randint(len(files))])

    def judge_path(self, path):
        self.set_busy('Listening...')
        try:
            res = self.engine.judge(path)
        except Exception as ex:  # noqa
            messagebox.showerror('Problem', f'Could not read this file: {ex}')
            return
        self.show_result(res, path)

    def show_result(self, res, path):
        self.last_path = path
        name = os.path.basename(path)
        self.verdict.config(text='RED\nALERT' if res['flagged'] else 'GREEN\nNORMAL', bg=RED if res['flagged'] else GREEN, fg='white')
        self.detail.set(f'Score {res["score"]:.3f}  vs alert level {res["alert"]:.3f}\n'
                        f'Folder says: {res["folder_truth"]}')
        notes = []
        if res['other_machine']:
            ct, ci = res['clip_machine']
            notes.append(f'This clip is from {ct} {ci}, but the device is calibrated on {self.engine.mtype} {self.engine.held}. '
                         f'That is a DIFFERENT machine, so an alert is expected until you press "Calibrate device" for it.')
        if res['used_for_calibration']:
            notes.append('Note: this healthy clip was used for calibration. Pick another for a strictly fair test.')
        if res['folder_truth'] == 'unknown':
            notes.append('This file is not from the dataset; the model only knows this machine type.')
        self.note.set('\n'.join(notes))
        self.draw_result(res, name)
        self.tree.insert('', 0, values=(name, f'{res["score"]:.3f}', f'{res["alert"]:.3f}',
                                         'RED' if res['flagged'] else 'GREEN', res['folder_truth']))
        self.btn_play.config(state='normal')

    def on_play(self):
        if not self.last_path:
            return

        def work():
            try:
                D.play(self.last_path)
            except Exception as ex:  # noqa
                self.root.after(0, lambda: messagebox.showinfo('Sound', f'Could not play: {ex}'))
        threading.Thread(target=work, daemon=True).start()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
