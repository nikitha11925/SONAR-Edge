/* STM32CubeIDE memory + speed test of the sound pipeline.
 *
 * Add to your CubeIDE project:
 *   1. Copy firmware/common/{ae_pipeline.c,ae_pipeline.h,model_data.c,model_data.h} and this file into Core/Src (headers: Core/Inc).
 *   2. In main.c add   #include "mem_test.h"-style prototype:  void mem_test_run(void);
 *      and call  mem_test_run();  once, after the MX_xxx_Init() calls (before the while(1) loop).
 *   3. Project Properties > C/C++ Build > Settings > MCU/MCU GCC Linker > Miscellaneous: add  -lm  (Libraries: m).
 *   4. Printing: either SWV/ITM (Debug config > enable Serial Wire Viewer; open SWV ITM Data Console, enable port 0)
 *      or UART (define MEM_TEST_UART and set the handle below). For SWV the _write below sends bytes to ITM.
 *   5. Hardware floating point: Properties > Settings > MCU Settings > Floating-point unit = FPv4/FPv5 (if your chip has one).
 *
 * Static flash/RAM use needs no code: build, then open the "Build Analyzer" tab (Memory Regions / Memory Details).
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "main.h"              /* pulls in the HAL for your chip */
#include "ae_pipeline.h"

/* ---- printing ---- */
#ifdef MEM_TEST_UART
extern UART_HandleTypeDef huart2;   /* change to your UART handle */
int _write(int fd, char *p, int n) { (void)fd; HAL_UART_Transmit(&huart2, (uint8_t *)p, n, 1000); return n; }
#else
int _write(int fd, char *p, int n) { (void)fd; for (int i = 0; i < n; i++) ITM_SendChar((uint32_t)p[i]); return n; }
#endif

/* ---- symbols from the CubeIDE linker script (STM32xxxx_FLASH.ld) ---- */
extern uint32_t _sdata, _edata, _sbss, _ebss, _estack;
extern uint32_t _Min_Heap_Size, _Min_Stack_Size;   /* these are *values*: use their addresses */
extern uint8_t  end;                               /* first byte after .bss = start of heap */
extern uint32_t _sidata;

#define STACK_PAINT 0xA5A5A5A5u

/* Paint the unused RAM between the heap start and the current stack pointer, so the high-water mark can be found later. */
static void stack_paint(void)
{
    uint32_t *p = (uint32_t *)&end;
    uint32_t *sp; __asm volatile("mov %0, sp" : "=r"(sp));
    for (; p < sp - 16; p++) *p = STACK_PAINT;
}

static uint32_t stack_used_peak(void)
{
    uint32_t *p = (uint32_t *)&end;
    uint32_t *top = &_estack;
    while (p < top && *p == STACK_PAINT) p++;
    return (uint32_t)((uint8_t *)top - (uint8_t *)p);
}

static void mem_report(const char *when)
{
    uint32_t data = (uint32_t)((uint8_t *)&_edata - (uint8_t *)&_sdata);
    uint32_t bss  = (uint32_t)((uint8_t *)&_ebss  - (uint8_t *)&_sbss);
    uint32_t ram_total = (uint32_t)((uint8_t *)&_estack - (uint8_t *)&_sdata);   /* RAM start .. top of stack */
    printf("\r\n=== MEMORY (%s) ===\r\n", when);
    printf("  RAM region (this linker script): %lu bytes\r\n", (unsigned long)ram_total);
    printf("  .data (initialised globals): %lu   .bss (zeroed globals): %lu   -> static RAM %lu\r\n",
           (unsigned long)data, (unsigned long)bss, (unsigned long)(data + bss));
    printf("  of which the sound pipeline buffers: %lu bytes\r\n", (unsigned long)ae_static_ram_bytes());
    printf("  reserved heap: %lu   reserved stack: %lu\r\n", (unsigned long)(uint32_t)&_Min_Heap_Size,
           (unsigned long)(uint32_t)&_Min_Stack_Size);
    printf("  stack peak so far: %lu bytes   RAM never touched: %lu bytes\r\n",
           (unsigned long)stack_used_peak(),
           (unsigned long)(ram_total - data - bss - stack_used_peak()));
}

/* ---- cycle counter for timing (Cortex-M3/M4/M7/M33; not M0/M0+) ---- */
static void cyc_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
#define CYC() (DWT->CYCCNT)

static int16_t pcm[N_FFT];
static float lm[N_MELS];

void mem_test_run(void)
{
    stack_paint();
    cyc_init();
    printf("\r\nModel: %d parameters (%d KB float32), %d mel bands\r\n", MODEL_PARAMS, MODEL_PARAMS * 4 / 1024, N_MELS);
    printf("Flash tables: weights+biases ~%u B, hann+twiddle %u B, mel %u B (also see Build Analyzer)\r\n",
           (unsigned)(MODEL_PARAMS * 4), (unsigned)(2 * N_FFT * 4), (unsigned)(MEL_W_COUNT * 4 + N_MELS * 4));
    mem_report("at start");

    for (int i = 0; i < N_FFT; i++)
        pcm[i] = (int16_t)(6000 * sinf(2 * 3.14159265f * 120 * i / 16000) + 2000 * sinf(2 * 3.14159265f * 1000 * i / 16000));

    for (int i = 0; i < 30; i++) { ae_logmel(pcm, lm); ae_cal_stats_add(lm); }
    ae_cal_stats_done();
    for (int i = 0; i < 30; i++) { ae_logmel(pcm, lm); ae_cal_thr_add(lm); }
    ae_cal_thr_done();

    const int N = 50;
    uint32_t cf = 0, cm = 0;
    float s = 0;
    for (int i = 0; i < N; i++) {
        uint32_t t0 = CYC();
        ae_logmel(pcm, lm);
        uint32_t t1 = CYC();
        s = ae_score(lm);
        uint32_t t2 = CYC();
        cf += t1 - t0; cm += t2 - t1;
    }
    float us_per_cyc = 1e6f / (float)SystemCoreClock;
    printf("\r\n=== SPEED (per 32 ms frame, %d runs, clock %lu Hz) ===\r\n", N, (unsigned long)SystemCoreClock);
    printf("  features: %lu us   model: %lu us   total: %lu %% of the 32 ms frame budget\r\n",
           (unsigned long)(cf / N * us_per_cyc), (unsigned long)(cm / N * us_per_cyc),
           (unsigned long)((cf + cm) / N * us_per_cyc / 320));
    printf("  last score x1000: %ld, alert level x1000: %ld\r\n", (long)(s * 1000), (long)(ae_alert_level() * 1000));
    mem_report("after running");
}
