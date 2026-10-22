Noise mismatch - model trained@6dB; rows = noise when the device was calibrated, columns = noise at test time
Each cell: false-alarm rate / faults found  (AUC in brackets). Mean over the machines of the type.

### fan  (4 machines)
| calibrated at \ tested at | 6 dB | 0 dB | -6 dB |
|---|---|---|---|
| 6 dB | 5% / 63%  (0.92) | 42% / 74%  (0.77) | 94% / 95%  (0.64) |
| 0 dB | 1% / 43%  (0.89) | 4% / 42%  (0.81) | 23% / 49%  (0.68) |
| -6 dB | 7% / 36%  (0.84) | 1% / 28%  (0.78) | 3% / 17%  (0.68) |
| mixed noise (10 clips each) | 0% / 35%  (0.89) | 1% / 31%  (0.82) | 8% / 29%  (0.68) |

### pump  (4 machines)
| calibrated at \ tested at | 6 dB | 0 dB | -6 dB |
|---|---|---|---|
| 6 dB | 35% / 87%  (0.95) | 46% / 84%  (0.87) | 87% / 94%  (0.79) |
| 0 dB | 26% / 69%  (0.93) | 19% / 59%  (0.85) | 24% / 63%  (0.78) |
| -6 dB | 13% / 50%  (0.85) | 4% / 30%  (0.81) | 3% / 20%  (0.73) |
| mixed noise (10 clips each) | 31% / 67%  (0.90) | 23% / 56%  (0.83) | 23% / 57%  (0.75) |

### slider  (4 machines)
| calibrated at \ tested at | 6 dB | 0 dB | -6 dB |
|---|---|---|---|
| 6 dB | 2% / 51%  (0.94) | 9% / 66%  (0.91) | 64% / 98%  (0.84) |
| 0 dB | 5% / 43%  (0.82) | 2% / 29%  (0.84) | 3% / 29%  (0.84) |
| -6 dB | 36% / 47%  (0.64) | 7% / 11%  (0.75) | 2% / 4%  (0.75) |
| mixed noise (10 clips each) | 3% / 46%  (0.85) | 2% / 30%  (0.84) | 2% / 17%  (0.81) |

### valve  (4 machines)
| calibrated at \ tested at | 6 dB | 0 dB | -6 dB |
|---|---|---|---|
| 6 dB | 2% / 1%  (0.56) | 5% / 5%  (0.56) | 41% / 46%  (0.54) |
| 0 dB | 4% / 3%  (0.48) | 3% / 3%  (0.51) | 3% / 2%  (0.52) |
| -6 dB | 26% / 22%  (0.43) | 6% / 4%  (0.48) | 2% / 3%  (0.49) |
| mixed noise (10 clips each) | 4% / 3%  (0.48) | 4% / 3%  (0.51) | 3% / 2%  (0.52) |

### Matched vs mismatched (all types, both models), mean over the three types that work (fan, pump, slider)

| model | calibration noise = test noise | calibrated quiet (6 dB), tested noisier | calibrated noisy (-6 dB), tested quieter | calibrated on mixed noise |
|---|---|---|---|---|
| trained@6dB | 8% / 41%  (0.83) | 44% / 73%  (0.79) | 11% / 40%  (0.81) | 11% / 41%  (0.82) |
| trained@all | 11% / 44%  (0.84) | 49% / 73%  (0.81) | 12% / 44%  (0.81) | 12% / 42%  (0.82) |
