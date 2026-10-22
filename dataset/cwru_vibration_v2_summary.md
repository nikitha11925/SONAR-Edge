CWRU vibration v2 - clean pipeline (no fabricated channel, time-ordered split, no neural network)

Health check on the healthy recording: AUC between its block 2 and block 3 (0.5 = no drift)
  full band, DE+FE: 0.55;  MPU-like, DE+FE: 0.58

Detector DE+FE: AUC per fault, with false-alarm rate (healthy test block) and detection rate
| sensor path | added noise (x healthy RMS) | inner race | ball | outer race | false alarms | detected (avg of 3 faults) |
|---|---|---|---|---|---|---|
| full 12 kHz | 0 | 1.000 | 1.000 | 1.000 | 3% | 100% |
| MPU-like 1 kHz | 0 | 0.994 | 0.997 | 0.995 | 6% | 100% |
| MPU-like 1 kHz | 0.5 | 0.976 | 0.985 | 0.979 | 7% | 98% |
| MPU-like 1 kHz | 1 | 0.854 | 0.889 | 0.862 | 5% | 36% |
| MPU-like 1 kHz | 2 | 0.553 | 0.566 | 0.561 | 4% | 5% |
| MPU-like 1 kHz | 4 | 0.493 | 0.490 | 0.499 | 4% | 2% |

Single-channel detectors, MPU-like path, no added noise (AUC inner/ball/outer):
  DE only: 0.974 / 0.992 / 0.981
  FE only: 0.994 / 0.995 / 0.994
