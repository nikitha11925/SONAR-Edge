@'
# Self-Calibrating Multi-Modal Edge AI Node for Industrial Machine Anomaly Detection

**Phase-I Verification: Vibration Signal Processing & Adaptive Thresholding**  
**Target Hardware:** ESP32-S3 Microcontroller (Dual-Core)  

---

## 1. Executive Summary
This repository contains the Phase-I validation scripts, verified console outputs, benchmark metrics, and visual artifacts for edge deployment on the ESP32-S3:
1. **Hilbert Transform Envelope Demodulation (`test_hilbert.py`):** Strips away high-frequency structural motor resonance (~3,000 Hz) to isolate the exact physical bearing fault frequency (~188 Hz BPFO), enabling a lightweight 1D CNN architecture (~6.8 KB quantized) that fits inside ESP32-S3 internal SRAM.
2. **Median Absolute Deviation (MAD) Thresholding (`test_threshold.py`):** Replaces rigid Gaussian ($\mu + 3\sigma$) alert boundaries with a non-parametric median-based calibration threshold, improving fault recall from **94.0% to 98.0%** on skewed error distributions without retraining.

---

## 2. Verified Console Output Logs

Running both verification scripts produces the following exact terminal output:

```text
(venv) PS E:\SONAR> python test_hilbert.py
Saved plot to hilbert_envelope_results.png

(venv) PS E:\SONAR> python test_threshold.py
Plot saved to threshold_comparison.png
Old Threshold: 0.881 | FA: 4.0% | Recall: 94.0%
MAD Threshold: 0.726 | FA: 8.0% | Recall: 98.0%

## 2. Quick Setup & Verification

### Install Dependencies
```bash
pip install -r requirements.txt