/* pdm_mic.h – PDM-to-PCM microphone driver for STM32F407-Discovery (MP45DT02)
 *
 * Uses I2S2 in master receive mode + BSP PDM filter library.
 * Collects exactly N_FFT=1024 PCM samples per frame (16 kHz, mono).
 *
 * API:
 *   pdm_mic_init()          – call once after MX_I2S2_Init()
 *   pdm_mic_frame_ready()   – returns 1 when a full frame is available
 *   pdm_mic_get_frame(buf)  – copies frame into caller's int16_t[N_FFT] buffer
 */
#ifndef PDM_MIC_H_
#define PDM_MIC_H_

#include <stdint.h>
#include "model_data.h"   /* for N_FFT */

void pdm_mic_init(void);
int  pdm_mic_frame_ready(void);
void pdm_mic_get_frame(int16_t out[N_FFT]);

/* Called from I2S DMA half/complete callbacks – do not call manually */
void pdm_mic_dma_half_cb(void);
void pdm_mic_dma_full_cb(void);

#endif /* PDM_MIC_H_ */

