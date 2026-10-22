Stage 2: the stage-1 winner of each type with 32 / 64 / 128 mel bands (same rule)

### fan   (structure D_F1_H32_b8; chosen = D_F1_H32_b8)
| design | mel bands | parameters | mean AUC | false alarms | faults found |
|---|---|---|---|---|---|
| D_F1_H32_b8_m32 | 32 | 4,776 | 0.861 | 5% | 48% |
| D_F1_H32_b8 | 64 | 6,856 | 0.891 | 4% | 54%  <-- chosen |
| D_F1_H32_b8_m128 | 128 | 11,016 | 0.896 | 5% | 61% |

### pump   (structure T_F1_b8; chosen = T_F1_b8)
| design | mel bands | parameters | mean AUC | false alarms | faults found |
|---|---|---|---|---|---|
| T_F1_b8_m32 | 32 | 1,352 | 0.925 | 33% | 80% |
| T_F1_b8 | 64 | 2,408 | 0.954 | 32% | 89%  <-- chosen |
| T_F1_b8_m128 | 128 | 4,520 | 0.971 | 29% | 94% |

### slider   (structure S_F1_b16; chosen = S_F1_b16_m32)
| design | mel bands | parameters | mean AUC | false alarms | faults found |
|---|---|---|---|---|---|
| S_F1_b16_m32 | 32 | 3,184 | 0.938 | 6% | 67%  <-- chosen |
| S_F1_b16 | 64 | 5,264 | 0.945 | 3% | 63% |
| S_F1_b16_m128 | 128 | 9,424 | 0.954 | 2% | 62% |

### valve   (structure S_F1_b16; chosen = S_F1_b16)
| design | mel bands | parameters | mean AUC | false alarms | faults found |
|---|---|---|---|---|---|
| S_F1_b16_m32 | 32 | 3,184 | 0.540 | 3% | 2% |
| S_F1_b16 | 64 | 5,264 | 0.571 | 3% | 2%  <-- chosen |
| S_F1_b16_m128 | 128 | 9,424 | 0.573 | 2% | 1% |
