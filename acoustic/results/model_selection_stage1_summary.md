Stage 1: model designs at 64 mel bands, tuning machines only (IDs 00/02/04 held out in turn, 6 dB, full calibration, 30 clips)

### fan   (best mean AUC 0.909; rule: smallest model within 0.02 -> chosen = D_F1_H32_b8)
| design | parameters | mean AUC | false alarms | faults found |
|---|---|---|---|---|
| D_F3_H64_b8 | 34,248 | 0.909 | 5% | 60% |
| D_F3_H32_b8 | 15,176 | 0.908 | 4% | 56% |
| D_F5_H64_b8 | 50,760 | 0.902 | 4% | 57% |
| D_F5_H32_b8 | 23,496 | 0.897 | 4% | 58% |
| D_F5_H64_b16 | 51,792 | 0.893 | 5% | 58% |
| D_F1_H32_b8 | 6,856 | 0.891 | 4% | 54%  <-- chosen |
| D_F1_H64_b8 | 17,736 | 0.889 | 4% | 54% |
| D_F3_H32_b16 | 15,696 | 0.884 | 4% | 58% |
| S_F5_b16 | 21,904 | 0.879 | 4% | 53% |
| S_F3_b16 | 13,584 | 0.876 | 5% | 53% |
| S_F1_b16 | 5,264 | 0.868 | 5% | 52% |
| T_F1_b8 | 2,408 | 0.861 | 4% | 54% |

### pump   (best mean AUC 0.954; rule: smallest model within 0.02 -> chosen = T_F1_b8)
| design | parameters | mean AUC | false alarms | faults found |
|---|---|---|---|---|
| T_F1_b8 | 2,408 | 0.954 | 32% | 89%  <-- chosen |
| D_F1_H32_b8 | 6,856 | 0.952 | 36% | 88% |
| D_F1_H64_b8 | 17,736 | 0.951 | 38% | 89% |
| D_F3_H64_b8 | 34,248 | 0.950 | 34% | 88% |
| S_F1_b16 | 5,264 | 0.939 | 43% | 90% |
| D_F5_H64_b8 | 50,760 | 0.936 | 40% | 88% |
| D_F5_H32_b8 | 23,496 | 0.934 | 30% | 85% |
| S_F3_b16 | 13,584 | 0.932 | 43% | 91% |
| D_F3_H32_b8 | 15,176 | 0.931 | 34% | 84% |
| D_F3_H32_b16 | 15,696 | 0.923 | 41% | 89% |
| D_F5_H64_b16 | 51,792 | 0.922 | 46% | 85% |
| S_F5_b16 | 21,904 | 0.913 | 44% | 89% |

### slider   (best mean AUC 0.965; rule: smallest model within 0.02 -> chosen = S_F1_b16)
| design | parameters | mean AUC | false alarms | faults found |
|---|---|---|---|---|
| D_F3_H32_b8 | 15,176 | 0.965 | 4% | 70% |
| D_F3_H64_b8 | 34,248 | 0.964 | 2% | 70% |
| S_F5_b16 | 21,904 | 0.964 | 4% | 75% |
| S_F3_b16 | 13,584 | 0.954 | 3% | 69% |
| D_F1_H64_b8 | 17,736 | 0.953 | 3% | 67% |
| D_F5_H64_b16 | 51,792 | 0.953 | 4% | 72% |
| D_F5_H32_b8 | 23,496 | 0.948 | 3% | 68% |
| D_F5_H64_b8 | 50,760 | 0.947 | 4% | 71% |
| D_F3_H32_b16 | 15,696 | 0.946 | 4% | 68% |
| S_F1_b16 | 5,264 | 0.945 | 3% | 63%  <-- chosen |
| T_F1_b8 | 2,408 | 0.928 | 2% | 54% |
| D_F1_H32_b8 | 6,856 | 0.927 | 3% | 61% |

### valve   (best mean AUC 0.571; rule: smallest model within 0.02 -> chosen = S_F1_b16)
| design | parameters | mean AUC | false alarms | faults found |
|---|---|---|---|---|
| S_F1_b16 | 5,264 | 0.571 | 3% | 2%  <-- chosen |
| D_F1_H64_b8 | 17,736 | 0.555 | 2% | 2% |
| D_F3_H32_b16 | 15,696 | 0.552 | 2% | 2% |
| D_F1_H32_b8 | 6,856 | 0.547 | 1% | 1% |
| D_F3_H64_b8 | 34,248 | 0.545 | 2% | 2% |
| D_F5_H64_b16 | 51,792 | 0.537 | 1% | 2% |
| T_F1_b8 | 2,408 | 0.536 | 3% | 3% |
| S_F5_b16 | 21,904 | 0.536 | 2% | 2% |
| D_F5_H64_b8 | 50,760 | 0.535 | 1% | 2% |
| S_F3_b16 | 13,584 | 0.534 | 2% | 1% |
| D_F3_H32_b8 | 15,176 | 0.524 | 2% | 2% |
| D_F5_H32_b8 | 23,496 | 0.524 | 2% | 2% |
(No design works for valve: best AUC 0.57. The choice is arbitrary; the smallest model within the band was taken.)
