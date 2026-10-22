"""Step 1: look at the real audio. Normal vs abnormal, and the same clip at 3 noise levels."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import features as F

OUT = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT, exist_ok=True)


def lm_of(snr, mtype, mid, label, idx):
    return F.logmel(F.load_channel0(F.list_wavs(snr, mtype, mid, label)[idx]))


def show(ax, lm, title):
    t = lm.shape[1] * F.HOP / 16000
    ax.imshow(lm, origin='lower', aspect='auto', cmap='magma', vmin=-80, vmax=-5,
              extent=[0, t, 0, 64])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('mel band (0 = low freq, 63 = ~8 kHz)')


# 1) normal vs abnormal at 6 dB, all four machine types (id_00)
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for c, m in enumerate(F.TYPES):
    show(axes[0, c], lm_of('6', m, 'id_00', 'normal', 0), f'{m} id_00 NORMAL (6 dB)')
    show(axes[1, c], lm_of('6', m, 'id_00', 'abnormal', 0), f'{m} id_00 ABNORMAL (6 dB)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'normal_vs_abnormal_6dB.png'), dpi=110)
plt.close(fig)

# 2) the same clip (fan and valve, id_00, normal #0) at 6 / 0 / -6 dB
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for r, m in enumerate(['fan', 'valve']):
    for c, s in enumerate(['6', '0', '-6']):
        show(axes[r, c], lm_of(s, m, 'id_00', 'normal', 0), f'{m} id_00 normal clip 0 @ {s} dB SNR')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'same_clip_three_noise_levels.png'), dpi=110)
plt.close(fig)

# 3) average spectrum (mean over time) normal vs abnormal, 6 dB, to see where they differ
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for c, m in enumerate(F.TYPES):
    for label, col in (('normal', 'tab:blue'), ('abnormal', 'tab:red')):
        files = F.list_wavs('6', m, 'id_00', label)[:40]
        spec = np.mean([F.logmel(F.load_channel0(p)).mean(axis=1) for p in files], axis=0)
        axes[c].plot(spec, color=col, label=label)
    axes[c].set_title(f'{m} id_00: mean log-mel, 40 clips, 6 dB')
    axes[c].set_xlabel('mel band')
    axes[c].set_ylabel('log-mel (dB-like)')
    axes[c].legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'mean_spectrum_normal_vs_abnormal_6dB.png'), dpi=110)
print('saved to', OUT)
