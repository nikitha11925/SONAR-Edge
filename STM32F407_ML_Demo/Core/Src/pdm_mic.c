/* pdm_mic.c – PDM microphone driver for STM32F407-Discovery
 *
 * The MP45DT02 on the Discovery board is wired to I2S2 (see main.h).
 * It outputs PDM at 1.024 MHz (oversampling = 64 × 16 kHz).
 * We read 16-bit PDM words via DMA, then decimate with the STM32 BSP
 * PDM-to-PCM filter (order-5 CIC + compensation FIR, 64× decimation).
 *
 * DMA double-buffer layout
 * ─────────────────────────
 *   Each DMA transfer moves PDM_HALF_WORDS 16-bit words.
 *   At 64× decimation one PCM sample needs 64 PDM bits = 4 × 16-bit words.
 *   So PDM_HALF_WORDS = PCM_FRAME_SAMPLES * 4 / 2 … adjusted for stereo
 *   packing in the I2S frame (left/right words interleaved).
 *
 *   PDM_HALF_WORDS = N_FFT / 2 * 4   (half of the double buffer)
 *   Full DMA buffer = 2 × PDM_HALF_WORDS
 *
 * When DMA half-complete fires  → process first  half → first  N_FFT/2 PCM samples
 * When DMA full-complete fires  → process second half → second N_FFT/2 PCM samples
 * After two callbacks the pcm_buf[N_FFT] is complete and frame_ready is set.
 *
 * NOTE: The BSP PDM library header is  Drivers/BSP/STM32F4xx-Discovery/stm32f4_discovery_audio.h
 *       and the C source is            Drivers/BSP/STM32F4xx-Discovery/stm32f4_discovery_audio.c
 *       STM32CubeIDE's BSP pack installs them automatically when you select the Discovery board.
 *       If they are absent, enable them in Project Manager > Middleware > BSP.
 */

#include <string.h>
#include "main.h"
#include "pdm_mic.h"

/* ---- BSP PDM filter ---- */
/* The BSP provides PDM_Filter_Init / PDM_Filter_64_LSB.
   Include the header that ships with the STM32F4-Discovery BSP pack.      */
#include "pdm_filter.h"   /* Drivers/BSP/STM32F4xx-Discovery/pdm_filter.h  */

/* ---- sizes ---- */
#define PCM_SAMPLES_PER_HALF  (N_FFT / 2)          /* 512 PCM samples per DMA half */
#define PDM_BYTES_PER_HALF    (PCM_SAMPLES_PER_HALF * 64 / 8)  /* 64× decimation, bits→bytes */
#define PDM_HW_PER_HALF       (PDM_BYTES_PER_HALF / 2)         /* 16-bit DMA words            */
#define PDM_BUF_HW            (PDM_HW_PER_HALF * 2)            /* full double buffer           */

/* ---- DMA buffer (must be in AHB-reachable SRAM) ---- */
static uint16_t s_pdm_buf[PDM_BUF_HW];

/* ---- PCM accumulation buffer ---- */
static int16_t  s_pcm_buf[N_FFT];
static volatile int s_frame_ready = 0;
static int s_half_done = 0;         /* 0 = waiting first half, 1 = waiting second half */

/* ---- PDM filter state ---- */
static PDMFilter_InitStruct s_filter;

/* ===== public API ===== */

void pdm_mic_init(void)
{
    /* Configure PDM filter */
    s_filter.Fs                  = 16000;
    s_filter.HP_HZ               = 10;    /* high-pass cut-off (removes DC) */
    s_filter.In_MicChannels      = 1;
    s_filter.Out_MicChannels     = 1;
    PDM_Filter_Init((PDMFilter_InitStruct *)&s_filter);

    s_frame_ready = 0;
    s_half_done   = 0;

    /* Start I2S2 DMA receive (circular – never stops) */
    extern I2S_HandleTypeDef hi2s2;
    HAL_I2S_Receive_DMA(&hi2s2, s_pdm_buf, PDM_BUF_HW);
}

int pdm_mic_frame_ready(void)
{
    return s_frame_ready;
}

void pdm_mic_get_frame(int16_t out[N_FFT])
{
    memcpy(out, s_pcm_buf, sizeof(s_pcm_buf));
    s_frame_ready = 0;
}

/* ---- internal helper ---- */
static void process_half(const uint16_t *pdm_hw, int16_t *pcm_out)
{
    /* PDM_Filter_64_LSB: convert PDM_BYTES_PER_HALF bytes of PDM into
       PCM_SAMPLES_PER_HALF int16 samples.
       Volume = 64 (matches BSP audio demo default). */
    PDM_Filter_64_LSB((uint8_t *)pdm_hw, (uint16_t *)pcm_out,
                      PCM_SAMPLES_PER_HALF, (PDMFilter_InitStruct *)&s_filter);
}

/* ===== DMA callbacks (called from HAL_I2S_RxHalfCpltCallback / HAL_I2S_RxCpltCallback) ===== */

void pdm_mic_dma_half_cb(void)
{
    /* First half of DMA buffer is ready */
    process_half(s_pdm_buf, s_pcm_buf);             /* → s_pcm_buf[0..511]   */
    s_half_done = 1;
}

void pdm_mic_dma_full_cb(void)
{
    /* Second half of DMA buffer is ready */
    process_half(s_pdm_buf + PDM_HW_PER_HALF,
                 s_pcm_buf + PCM_SAMPLES_PER_HALF); /* → s_pcm_buf[512..1023] */
    if (s_half_done && !s_frame_ready) {
        s_frame_ready = 1;
    }
    s_half_done = 0;
}

