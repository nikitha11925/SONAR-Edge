# Implementation Plan: CWRU Bearing Vibration Dataset Cleaning & Preprocessing Pipeline

This plan outlines the end-to-end data cleaning, signal conditioning, framing, and feature extraction pipeline for the Case Western Reserve University (CWRU) vibration dataset (`dataset/cwru/*.mat`), preparing clean training/testing matrices for the 67-feature Edge Autoencoder.

---

## User Review Required

> [!IMPORTANT]
> **Channel Selection & Resampling**: The CWRU dataset contains Drive End (`DE`), Fan End (`FE`), and Base Accelerometer (`BA`) channels sampled at 12,000 Hz. If fusing directly with 16 kHz audio (from MIMII), we can either:
> 1. Resample CWRU to 16 kHz (standardized rate).
> 2. Process vibration windows at 12 kHz with an identical time window length (e.g., 32 ms = 384 samples at 12 kHz vs 512 samples at 16 kHz).

> [!NOTE]
> **Baseline Partitioning**: In accordance with one-class unsupervised anomaly detection, file `97.mat` (healthy baseline) will be split into 80% train / 20% validation, while fault files (`105.mat`, `118.mat`, `130.mat`) will be held out exclusively for testing.

---

## Open Questions

1. **Window Size**: Do you prefer a **32 ms window** (matches MIMII audio frame length) or a **64 ms window** (provides higher frequency resolution for vibration harmonics)?
2. **Channel Output**: Should we extract RMS across all 3 available accelerometer sensors (`DE`, `FE`, `BA`), or synthesize/map them to simulated 3-axis $(X, Y, Z)$ format?

---

## Proposed Changes

### Data Pipeline & Preprocessing Module

#### [NEW] [clean_cwru_dataset.py](file:///c:/dev/capstone/clean_cwru_dataset.py)
* **MAT Parser**: Dynamically extracts Drive End (`DE`), Fan End (`FE`), and Base Accelerometer (`BA`) channels from `.mat` dictionaries regardless of key naming quirks (`X097_DE_time` vs `X105_DE_time`).
* **DC Bias Removal**: Subtracts channel mean $\mu$ to remove static acceleration bias and sensor zero-point drift.
* **4th-Order Butterworth Bandpass Filter**:
  * Passband: $10\text{ Hz}$ (removes building sway/static tilt) to $5000\text{ Hz}$ (anti-aliasing).
  * Implemented using `scipy.signal.butter` with zero-phase filtering (`scipy.signal.filtfilt`).
* **Outlier Filtering & Clipping**:
  * Applies a Hampel / $3\sigma$ filter to replace transient sensor spikes and dropouts with local median values.
* **Sliding Window Framing**:
  * Splits continuous signals into overlapping frames (e.g., 512 samples, 50% overlap / 256 sample hop).
  * Applies Hanning window function to eliminate boundary spectral leakage.
* **Feature Extraction (Time & RMS Domain)**:
  * Computes Root Mean Square (RMS) acceleration: $a_{\text{RMS}} = \sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}$.
  * Computes Peak-to-Peak, Kurtosis, and Crest Factor for diagnostic verification.
  * Formats the final 3-channel vibration vector for concatenation into the 67-feature multi-modal vector.
* **Dataset Splitting & Export**:
  * Exports preprocessed healthy training splits, validation splits, and labeled fault test splits to `dataset/cwru/processed/` in `.npy` and `.csv` formats.

---

## Verification Plan

### Automated Tests
1. **Signal Cleanliness & Filter Verification**:
   * Verify that mean post-filtering is $\approx 0.0$ ($\pm 10^{-6}$).
   * Verify signal PSD (Power Spectral Density) attenuates frequencies $<10\text{ Hz}$ and $>5000\text{ Hz}$.
2. **Frame Consistency & NaN Checks**:
   * Ensure 0 NaN / Inf values across all framed tensors.
   * Verify generated array shapes: `[N_frames, 3]` for vibration RMS features.
3. **Execution Command**:
   ```bash
   python clean_cwru_dataset.py --input_dir dataset/cwru --output_dir dataset/cwru/processed
   ```

### Manual Verification
* Inspect diagnostic plots comparing raw vs. cleaned vibration signals (time domain and FFT spectrum).
* Validate fault condition separation between healthy (`97.mat`) and faulty (`105.mat`, `118.mat`, `130.mat`) RMS energy distributions.
