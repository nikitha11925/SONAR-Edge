/* Portable sound pipeline: 1024 samples -> log-mel -> autoencoder score. Plain C, no OS or HAL needed. */
#ifndef AE_PIPELINE_H_
#define AE_PIPELINE_H_
#include <stdint.h>
#include "model_data.h"

/* log-mel of one 1024-sample frame (same maths as acoustic/features.py, no centre padding) */
void  ae_logmel(const int16_t pcm[N_FFT], float out[N_MELS]);

/* calibration on healthy frames: pass 1 learns mean/std, pass 2 learns the alert level (mean + 3 std of scores) */
void  ae_cal_stats_add(const float lm[N_MELS]);
void  ae_cal_stats_done(void);
void  ae_cal_thr_add(const float lm[N_MELS]);
void  ae_cal_thr_done(void);

/* mean squared reconstruction error of one frame (uses the calibration mean/std) */
float ae_score(const float lm[N_MELS]);
float ae_alert_level(void);

/* RAM bytes used by this module's static buffers (for the memory report) */
uint32_t ae_static_ram_bytes(void);
#endif
