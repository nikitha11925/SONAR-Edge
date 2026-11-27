/* ESP32-S3 (ESP-IDF) memory + speed test of the sound pipeline. Output goes to the serial monitor. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_app_desc.h"
#include "ae_pipeline.h"

static void heap_line(const char *name, uint32_t caps)
{
    printf("  %-22s total %7u  free %7u  lowest-free %7u  largest-block %7u\n", name,
           (unsigned)heap_caps_get_total_size(caps), (unsigned)heap_caps_get_free_size(caps),
           (unsigned)heap_caps_get_minimum_free_size(caps), (unsigned)heap_caps_get_largest_free_block(caps));
}

static void report(const char *when)
{
    printf("\n=== MEMORY (%s) ===\n", when);
    heap_line("internal RAM", MALLOC_CAP_INTERNAL);
    heap_line("external PSRAM", MALLOC_CAP_SPIRAM);
    printf("  main task stack never-used (high-water): %u bytes\n",
           (unsigned)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t)));
}

static int16_t pcm[N_FFT];
static float lm[N_MELS];

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(3000)); /* wait so the laptop's serial monitor can reconnect after a reset over USB */
    printf("\nModel: %d parameters (%d KB float32), %d mel bands\n", MODEL_PARAMS, MODEL_PARAMS * 4 / 1024, N_MELS);
    printf("Pipeline static RAM (.bss, counted in 'internal RAM' above): %u bytes\n", (unsigned)ae_static_ram_bytes());
    printf("Flash tables: weights+biases ~%u B, hann+twiddle %u B, mel %u B\n",
           (unsigned)(MODEL_PARAMS * 4), (unsigned)(2 * N_FFT * 4), (unsigned)(MEL_W_COUNT * 4 + N_MELS * 4));
    report("at start");

    /* fake machine sound: a 120 Hz tone + 1 kHz tone, so the maths has real work to do */
    for (int i = 0; i < N_FFT; i++)
        pcm[i] = (int16_t)(6000 * sinf(2 * 3.14159265f * 120 * i / 16000) + 2000 * sinf(2 * 3.14159265f * 1000 * i / 16000));

    for (int i = 0; i < 30; i++) { ae_logmel(pcm, lm); ae_cal_stats_add(lm); }
    ae_cal_stats_done();
    for (int i = 0; i < 30; i++) { ae_logmel(pcm, lm); ae_cal_thr_add(lm); }
    ae_cal_thr_done();

    const int N = 200;
    int64_t t_feat = 0, t_model = 0;
    float s = 0;
    for (int i = 0; i < N; i++) {
        int64_t t0 = esp_timer_get_time();
        ae_logmel(pcm, lm);
        int64_t t1 = esp_timer_get_time();
        s = ae_score(lm);
        int64_t t2 = esp_timer_get_time();
        t_feat += t1 - t0; t_model += t2 - t1;
    }
    printf("\n=== SPEED (per 32 ms frame, %d runs) ===\n", N);
    printf("  features: %.0f us   model: %.0f us   total: %.1f %% of the 32 ms frame budget\n",
           (double)t_feat / N, (double)t_model / N, 100.0 * (t_feat + t_model) / N / 32000.0);
    printf("  last score %.4f, alert level %.4f\n", s, ae_alert_level());

    report("after running");
    printf("\nFlash/RAM of the whole program: run  idf.py size  (and  idf.py size-components)\n");

    while (1) vTaskDelay(pdMS_TO_TICKS(5000));
}
