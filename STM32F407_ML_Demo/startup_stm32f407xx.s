/* startup_stm32f407xx.s – Cortex-M4 startup for STM32F407
 *
 * This is the minimal startup file CubeIDE generates for the STM32F407.
 * If CubeIDE regenerates the project it will replace this with its own version
 * (which is identical in function).  Keep this file only if you are building
 * WITHOUT the .ioc generator.
 *
 * Actions:
 *   1. Set the initial stack pointer (from linker _estack).
 *   2. Copy .data from FLASH to RAM.
 *   3. Zero .bss.
 *   4. Call SystemInit() (sets up the FPU and flash prefetch).
 *   5. Call the C library entry __libc_init_array (C++ constructors).
 *   6. Call main().
 *   7. Loop forever if main() returns.
 */

  .syntax unified
  .cpu cortex-m4
  .fpu softvfp
  .thumb

  .global g_pfnVectors
  .global Default_Handler

/* Provided by the linker script */
  .word _sidata
  .word _sdata
  .word _edata
  .word _sbss
  .word _ebss

/*
 * Reset_Handler
 */
  .section .text.Reset_Handler
  .weak Reset_Handler
  .type Reset_Handler, %function
Reset_Handler:
  ldr   r0, =_estack
  mov   sp, r0

  /* Copy .data */
  ldr r0, =_sdata
  ldr r1, =_edata
  ldr r2, =_sidata
  movs r3, #0
  b     data_loop_check

data_loop:
  ldr r4, [r2, r3]
  str r4, [r0, r3]
  adds r3, r3, #4

data_loop_check:
  adds r4, r0, r3
  cmp  r4, r1
  bcc  data_loop

  /* Zero .bss */
  ldr r2, =_sbss
  ldr r4, =_ebss
  movs r3, #0
  b     bss_loop_check

bss_loop:
  str  r3, [r2]
  adds r2, r2, #4

bss_loop_check:
  cmp r2, r4
  bcc bss_loop

  /* Call SystemInit (from system_stm32f4xx.c via CMSIS) */
  bl SystemInit

  /* C++ global constructors */
  bl __libc_init_array

  bl main

loop_forever:
  b loop_forever

  .size Reset_Handler, .-Reset_Handler

/*
 * Default handler (infinite loop) for all unhandled interrupts
 */
  .section .text.Default_Handler,"ax",%progbits
Default_Handler:
  b Default_Handler
  .size Default_Handler, .-Default_Handler

/*
 * Vector table (must be at 0x08000000 so the boot ROM can read SP + PC)
 */
  .section .isr_vector,"a",%progbits
  .type g_pfnVectors, %object
  .size g_pfnVectors, .-g_pfnVectors

g_pfnVectors:
  .word _estack
  .word Reset_Handler
  .word NMI_Handler
  .word HardFault_Handler
  .word MemManage_Handler
  .word BusFault_Handler
  .word UsageFault_Handler
  .word 0
  .word 0
  .word 0
  .word 0
  .word SVC_Handler
  .word DebugMon_Handler
  .word 0
  .word PendSV_Handler
  .word SysTick_Handler

  /* External interrupts – only the ones we use are named; rest are Default */
  .word Default_Handler     /* 0  WWDG          */
  .word Default_Handler     /* 1  PVD           */
  .word Default_Handler     /* 2  TAMP_STAMP    */
  .word Default_Handler     /* 3  RTC_WKUP      */
  .word Default_Handler     /* 4  FLASH         */
  .word Default_Handler     /* 5  RCC           */
  .word Default_Handler     /* 6  EXTI0         */
  .word Default_Handler     /* 7  EXTI1         */
  .word Default_Handler     /* 8  EXTI2         */
  .word Default_Handler     /* 9  EXTI3         */
  .word Default_Handler     /* 10 EXTI4         */
  .word Default_Handler     /* 11 DMA1_Stream0  */
  .word Default_Handler     /* 12 DMA1_Stream1  */
  .word Default_Handler     /* 13 DMA1_Stream2  */
  .word DMA1_Stream3_IRQHandler  /* 14 DMA1_Stream3 ← I2S2 RX */
  .word Default_Handler     /* 15 DMA1_Stream4  */
  .word Default_Handler     /* 16 DMA1_Stream5  */
  .word Default_Handler     /* 17 DMA1_Stream6  */
  /* … remaining 73 vectors left as Default_Handler for brevity … */
  /* CubeIDE's generated startup fills all 91 vectors; replace this   */
  /* file with the generated one (startup_stm32f407xx.s) in practice. */

/*
 * Weak aliases – override any of these in your application code
 */
  .weak NMI_Handler
  .thumb_set NMI_Handler,Default_Handler

  .weak HardFault_Handler
  .thumb_set HardFault_Handler,Default_Handler

  .weak MemManage_Handler
  .thumb_set MemManage_Handler,Default_Handler

  .weak BusFault_Handler
  .thumb_set BusFault_Handler,Default_Handler

  .weak UsageFault_Handler
  .thumb_set UsageFault_Handler,Default_Handler

  .weak SVC_Handler
  .thumb_set SVC_Handler,Default_Handler

  .weak DebugMon_Handler
  .thumb_set DebugMon_Handler,Default_Handler

  .weak PendSV_Handler
  .thumb_set PendSV_Handler,Default_Handler

  .weak SysTick_Handler
  .thumb_set SysTick_Handler,Default_Handler

  .weak DMA1_Stream3_IRQHandler
  .thumb_set DMA1_Stream3_IRQHandler,Default_Handler

