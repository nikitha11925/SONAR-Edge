import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Configuration: 48 kHz sampling rate, 2048 sample window
FS = 48000
WINDOW_SIZE = 2048
t = np.arange(WINDOW_SIZE) / FS

# Simulated bearing vibration window with baseline rotation and fault shocks
baseline_rumble = 0.5 * np.sin(2 * np.pi * 30 * t)
impact_bursts = 1.2 * np.sin(2 * np.pi * 3000 * t) * np.exp(-np.tile(np.linspace(0, 5, 256), 8))
raw_window = baseline_rumble + impact_bursts

# 1. Compute Hilbert Transform & Envelope
analytic_signal = hilbert(raw_window)
envelope = np.abs(analytic_signal)

# 2. Compute Frequency Spectra (FFT)
freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1/FS)
raw_fft = np.abs(np.fft.rfft(raw_window))
env_fft = np.abs(np.fft.rfft(envelope - np.mean(envelope)))

# 3. Plot Comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

# Time domain: Raw vs Envelope
axes[0, 0].plot(t * 1000, raw_window, color='gray', alpha=0.7, label='Raw Signal')
axes[0, 0].plot(t * 1000, envelope, color='crimson', linewidth=1.5, label='Hilbert Envelope')
axes[0, 0].set_title('Time Domain: Raw Signal vs. Extracted Envelope')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Zoomed-in transient
axes[0, 1].plot(t[:256] * 1000, raw_window[:256], color='gray', alpha=0.7)
axes[0, 1].plot(t[:256] * 1000, envelope[:256], color='crimson', linewidth=1.5)
axes[0, 1].set_title('Zoomed View: Single Impact Pulse')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].grid(True, alpha=0.3)

# Frequency domain: Raw Spectrum
axes[1, 0].plot(freqs, raw_fft, color='steelblue')
axes[1, 0].set_title('Raw Spectrum (Carrier Noise Dominates)')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Magnitude')
axes[1, 0].set_xlim(0, 5000)
axes[1, 0].grid(True, alpha=0.3)

# Frequency domain: Envelope Spectrum
axes[1, 1].plot(freqs, env_fft, color='darkorange')
axes[1, 1].set_title('Envelope Spectrum (Fault Repetition Peak Isolated)')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_xlim(0, 500)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hilbert_envelope_results.png', dpi=300)
print("Saved plot to hilbert_envelope_results.png")