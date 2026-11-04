#include <math.h>
#include "ae_pipeline.h"

/* ---- static RAM (this is what the memory report counts) ---- */
static float s_re[N_FFT], s_im[N_FFT];            /* FFT work buffers      */
static float s_buf_a[MAX_DIM], s_buf_b[MAX_DIM];  /* layer ping-pong       */
static float s_x[N_MELS];                         /* normalised input      */
static float s_mu[N_MELS], s_m2[N_MELS], s_sd[N_MELS]; /* calibration stats */
static float s_alert, s_smean, s_sm2;
static uint32_t s_n_stats, s_n_thr;

uint32_t ae_static_ram_bytes(void)
{
    return sizeof(s_re) + sizeof(s_im) + sizeof(s_buf_a) + sizeof(s_buf_b) + sizeof(s_x) +
           sizeof(s_mu) + sizeof(s_m2) + sizeof(s_sd) + 3 * sizeof(float) + 2 * sizeof(uint32_t);
}

/* ---- feature extraction ---- */
static void fft1024(void)
{
    /* bit reversal */
    for (uint32_t i = 1, j = 0; i < N_FFT; i++) {
        uint32_t bit = N_FFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = s_re[i]; s_re[i] = s_re[j]; s_re[j] = t;
            t = s_im[i]; s_im[i] = s_im[j]; s_im[j] = t;
        }
    }
    for (uint32_t len = 2; len <= N_FFT; len <<= 1) {
        uint32_t step = N_FFT / len;
        for (uint32_t i = 0; i < N_FFT; i += len) {
            for (uint32_t k = 0; k < len / 2; k++) {
                float wr = TWIDDLE[2 * k * step], wi = TWIDDLE[2 * k * step + 1];
                uint32_t a = i + k, b = i + k + len / 2;
                float tr = s_re[b] * wr - s_im[b] * wi;
                float ti = s_re[b] * wi + s_im[b] * wr;
                s_re[b] = s_re[a] - tr; s_im[b] = s_im[a] - ti;
                s_re[a] += tr;          s_im[a] += ti;
            }
        }
    }
}

void ae_logmel(const int16_t pcm[N_FFT], float out[N_MELS])
{
    for (int i = 0; i < N_FFT; i++) { s_re[i] = (pcm[i] * (1.0f / 32768.0f)) * HANN[i]; s_im[i] = 0.0f; }
    fft1024();
    /* power spectrum (bins 0..512) stored back into s_re */
    for (int k = 0; k < N_BINS; k++) s_re[k] = s_re[k] * s_re[k] + s_im[k] * s_im[k];
    const float *w = MEL_W;
    for (int m = 0; m < N_MELS; m++) {
        float acc = 0.0f;
        for (int j = 0; j < MEL_COUNT[m]; j++) acc += w[j] * s_re[MEL_FIRST[m] + j];
        w += MEL_COUNT[m];
        out[m] = 10.0f * log10f(acc + 2.220446e-16f);
    }
}

/* ---- model ---- */
static float run_model(const float *x_in)     /* x_in = normalised input; returns MSE */
{
    const float *w = MODEL_W, *b = MODEL_B;
    const float *in = x_in;
    float *out = s_buf_a;
    for (int l = 0; l < N_LAYERS; l++) {
        int ni = LAYER_DIMS[l], no = LAYER_DIMS[l + 1];
        for (int o = 0; o < no; o++) {
            float s = b[o];
            for (int i = 0; i < ni; i++) s += w[i] * in[i];
            w += ni;
            out[o] = (l < N_LAYERS - 1 && s < 0.0f) ? 0.0f : s;
        }
        b += no;
        in = out;
        out = (out == s_buf_a) ? s_buf_b : s_buf_a;
    }
    float e = 0.0f;
    for (int i = 0; i < N_MELS; i++) { float d = in[i] - x_in[i]; e += d * d; }
    return e / N_MELS;
}

float ae_score(const float lm[N_MELS])
{
    for (int i = 0; i < N_MELS; i++) s_x[i] = (lm[i] - s_mu[i]) / s_sd[i];
    return run_model(s_x);
}

/* ---- calibration (Welford running mean / variance) ---- */
void ae_cal_stats_add(const float lm[N_MELS])
{
    s_n_stats++;
    for (int i = 0; i < N_MELS; i++) {
        float d = lm[i] - s_mu[i];
        s_mu[i] += d / s_n_stats;
        s_m2[i] += d * (lm[i] - s_mu[i]);
    }
}

void ae_cal_stats_done(void)
{
    for (int i = 0; i < N_MELS; i++)
        s_sd[i] = (s_n_stats > 1 ? sqrtf(s_m2[i] / s_n_stats) : 0.0f) + 1e-6f;
}

void ae_cal_thr_add(const float lm[N_MELS])
{
    float s = ae_score(lm);
    s_n_thr++;
    float d = s - s_smean;
    s_smean += d / s_n_thr;
    s_sm2 += d * (s - s_smean);
}

void ae_cal_thr_done(void)
{
    s_alert = s_smean + 3.0f * (s_n_thr > 1 ? sqrtf(s_sm2 / s_n_thr) : 0.0f);
}

float ae_alert_level(void) { return s_alert; }
