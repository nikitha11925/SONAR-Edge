/* Laptop check: reads raw int16 samples (1024 per frame) from a file, prints log-mel and score.
 * Used by firmware/tools/check_against_python.py to confirm the C code matches the Python features. */
#include <stdio.h>
#include "ae_pipeline.h"

int main(int argc, char **argv)
{
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    static int16_t pcm[N_FFT];
    static float lm[N_MELS];
    if (!f || fread(pcm, 2, N_FFT, f) != N_FFT) return 2;
    ae_logmel(pcm, lm);
    for (int i = 0; i < N_MELS; i++) printf("%.6f\n", lm[i]);
    fclose(f);
    printf("RAM %u\n", (unsigned)ae_static_ram_bytes());
    return 0;
}
