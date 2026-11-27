"""
BOARD TEST WITH REAL RECORDINGS: the laptop sends recorded sound to the ESP32-S3 over USB, the board scores it,
and the same sound is scored in Python, so the two can be compared.

    python firmware/tools/stream_test.py                         (defaults: your own recordings in own_data/clips)
    python firmware/tools/stream_test.py --port COM7 --normal <wav> --faulty <wav>

Needs the board program in firmware/esp32_stream flashed on the board, connected by its "USB" port.

Steps (same as acoustic/own_recording_test.py, only the device doing the maths changes):
  1. Calibration, feature numbers: the first K healthy clips of --cal-dir are sent; the BOARD learns the mean/spread.
  2. Calibration, alert level:     the next K healthy clips are scored by the board; alert = mean + 3 x spread
                                   of those clip scores (computed here on the laptop from the board's scores).
  3. Test: the --normal and --faulty clips are scored by the board. Clip score = average of its frame scores.
  Python does steps 1-3 on the same samples with the same maths (log-mel without centre padding, as on the board).
Clip WAV files must be 16 kHz, 16-bit mono (own_data/clips already are).
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import soundfile as sf
import librosa
import serial
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'acoustic'))
import demo as D                 # noqa: E402
import model_selection as M      # noqa: E402

N_FFT, HOP, SR = 1024, 512, 16000
WINDOW = 3                       # frames in flight at once; the board's 8 KB input buffer drops data if overfilled
CLIPS = os.path.join(ROOT, 'own_data', 'clips')


# ---------- sound ----------
def read_int16(path):
    y, sr = sf.read(path, dtype='int16', always_2d=True)
    if sr != SR:
        sys.exit(f'{path}: sample rate {sr}, expected {SR}')
    return y[:, 0]


def frames_of(y):
    n = (len(y) - N_FFT) // HOP + 1
    return np.stack([y[i * HOP:i * HOP + N_FFT] for i in range(n)]).astype('<i2')


# ---------- Python reference (same maths as the board) ----------
def py_logmel(y, mels):
    mel = librosa.feature.melspectrogram(y=y.astype(np.float32) / 32768.0, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=mels, power=2.0, center=False)
    return (10.0 * np.log10(mel + np.finfo(np.float64).eps)).T.astype(np.float32)      # (frames, mels)


def py_frame_scores(model, lm, mu, sd):
    with torch.no_grad():
        x = torch.from_numpy(((lm - mu) / sd).astype(np.float32))
        return ((model(x) - x) ** 2).mean(1).numpy()


# ---------- board ----------
class Board:
    def __init__(self, port):
        self.port = port
        self.s = None

    def open(self, wait=10.0):
        t0 = time.time()
        while time.time() - t0 < wait:
            try:
                s = serial.Serial()
                s.port, s.baudrate, s.timeout = self.port, 115200, 2.0
                s.dtr = False            # keep both lines off: on this port they can reset the chip
                s.rts = False
                s.open()
                self.s = s
                return
            except serial.SerialException:
                time.sleep(0.2)
        sys.exit(f'Could not open {self.port}. Is the board plugged into its "USB" port?')

    def reply(self, tag, n):
        """Skip anything (start-up text) until 0xA5 + tag, then return n data bytes."""
        t0 = time.time()
        prev = b''
        while time.time() - t0 < 10:
            b = self.s.read(1)
            if not b:
                continue
            if prev == b'\xa5' and b == tag:
                data = self.s.read(n) if n else b''
                if len(data) != n:
                    break
                return data
            prev = b
        sys.exit(f'No reply from the board (waiting for {tag!r}). Press RST on the board and try again.')

    def restart(self):
        """Restart the board so its calibration starts empty, then wait until it answers."""
        self.s.write(b'R')
        self.s.flush()
        self.s.close()
        time.sleep(2.0)
        self.open()
        for _ in range(10):
            self.s.reset_input_buffer()
            self.s.write(b'P')
            self.s.timeout = 1.0
            try:
                info = self.reply(b'P', 8)
                self.s.timeout = 2.0
                time.sleep(0.3)
                self.s.reset_input_buffer()       # drop replies to repeated pings
                return np.frombuffer(info, '<u4')
            except SystemExit:
                time.sleep(0.5)
        sys.exit('The board did not answer after restarting. Is the esp32_stream program flashed?')

    def send_frames(self, cmd, fr, n_reply):
        """Send frames with at most WINDOW waiting for a reply; return the reply data of each frame."""
        out, sent = [], 0
        while len(out) < len(fr):
            while sent < len(fr) and sent - len(out) < WINDOW:
                self.s.write(cmd + fr[sent].tobytes())
                sent += 1
            out.append(self.reply(cmd, n_reply))
        return out

    def cal_add(self, fr):
        self.send_frames(b'S', fr, 0)

    def cal_done(self):
        self.s.write(b'D')
        self.reply(b'D', 0)

    def led(self, colour):
        """Board RGB LED: 0 off, 1 green, 2 red, 3 blue."""
        self.s.write(b'L' + bytes([colour]))
        self.reply(b'L', 0)

    def scores(self, fr):
        d = self.send_frames(b'F', fr, 8)
        sc = np.array([np.frombuffer(x[:4], '<f4')[0] for x in d], np.float64)
        us = np.array([np.frombuffer(x[4:], '<u4')[0] for x in d])
        if not np.all(np.isfinite(sc)):
            sys.exit('The board returned invalid scores (calibration missing?). Press RST and try again.')
        return sc, us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM7')
    ap.add_argument('--cal-dir', default=os.path.join(CLIPS, 'healthy_session1'))
    ap.add_argument('--normal', default=os.path.join(CLIPS, 'healthy_session2', 'healthy_session2_001.wav'))
    ap.add_argument('--faulty', default=os.path.join(CLIPS, 'faulty', 'faulty_001.wav'))
    ap.add_argument('--type', default='fan')
    ap.add_argument('--k', type=int, default=D.K)
    ap.add_argument('--out', default=os.path.join(ROOT, 'firmware', 'results', 'board_stream_test.csv'))
    a = ap.parse_args()

    cal = sorted(os.path.join(a.cal_dir, f) for f in os.listdir(a.cal_dir) if f.lower().endswith('.wav'))
    if len(cal) < 2 * a.k:
        sys.exit(f'Need {2 * a.k} healthy clips in {a.cal_dir} (found {len(cal)}).')
    stats_clips, alert_clips = cal[:a.k], cal[a.k:2 * a.k]
    tests = [('normal', a.normal), ('faulty', a.faulty)]

    design, mels = D.chosen_design(a.type)
    model, _ = D.get_model(a.type, 'id_06', '6', design, mels, False)
    model.eval()

    board = Board(a.port)
    board.open()
    info = board.restart()
    print(f'board ready on {a.port}: {info[0]} mel bands, {info[1]:,} model parameters')
    if info[0] != mels:
        sys.exit(f'The board uses {info[0]} mel bands but the Python model uses {mels}. Re-export and re-flash.')

    # 1. feature numbers
    t0 = time.time()
    print(f'calibration 1/2: sending {a.k} healthy clips (feature numbers)...')
    py_lm = []
    for f in stats_clips:
        y = read_int16(f)
        board.cal_add(frames_of(y))
        py_lm.append(py_logmel(y, mels))
    board.cal_done()
    X = np.concatenate(py_lm)
    mu, sd = X.mean(0), X.std(0) + 1e-6

    # 2. alert level
    print(f'calibration 2/2: scoring {a.k} more healthy clips (alert level)...')
    b_alert_sc, p_alert_sc = [], []
    for f in alert_clips:
        y = read_int16(f)
        b_alert_sc.append(board.scores(frames_of(y))[0].mean())
        p_alert_sc.append(py_frame_scores(model, py_logmel(y, mels), mu, sd).mean())
    b_alert, p_alert = M.tau(np.array(b_alert_sc)), M.tau(np.array(p_alert_sc))

    # 3. test clips
    rows, all_us = [], []
    for label, f in tests:
        y = read_int16(f)
        bs, us = board.scores(frames_of(y))
        ps = py_frame_scores(model, py_logmel(y, mels), mu, sd)
        all_us.append(us)
        rows.append(dict(clip=os.path.relpath(f, ROOT), label=label,
                         board_score=bs.mean(), python_score=ps.mean(),
                         board_alert=b_alert, python_alert=p_alert,
                         board_verdict='FAULT' if bs.mean() > b_alert else 'OK',
                         python_verdict='FAULT' if ps.mean() > p_alert else 'OK',
                         max_frame_diff=float(np.abs(bs - ps).max()), frames=len(bs)))
    board.s.close()
    us = np.concatenate(all_us)

    print(f'\ndone in {time.time() - t0:.0f} s\n')
    print(f'alert level        board {b_alert:.4f}   python {p_alert:.4f}')
    print(f'{"clip":<46}{"label":<8}{"board":>9}{"python":>9}{"diff %":>8}   board verdict / python verdict')
    for r in rows:
        diff = 100 * abs(r['board_score'] - r['python_score']) / r['python_score']
        print(f'{r["clip"]:<46}{r["label"]:<8}{r["board_score"]:>9.4f}{r["python_score"]:>9.4f}{diff:>8.3f}'
              f'   {r["board_verdict"]} / {r["python_verdict"]}')
    print(f'\nbiggest single-frame difference: {max(r["max_frame_diff"] for r in rows):.2e}')
    print(f'time on the chip per frame (features + model): median {np.median(us) / 1000:.2f} ms '
          f'= {100 * np.median(us) / 32000:.1f} % of the 32 ms budget')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ['median_chip_us_per_frame'])
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'median_chip_us_per_frame': float(np.median(us))})
    print('saved', os.path.relpath(a.out, ROOT))


if __name__ == '__main__':
    main()
