/* main.h – STM32F407-Discovery ML anomaly demo */
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"

/* ---- On-board LEDs (STM32F407-Discovery) ---- */
/* PD12 green, PD13 orange, PD14 red, PD15 blue   */
#define LED_GREEN_PIN   GPIO_PIN_12
#define LED_ORANGE_PIN  GPIO_PIN_13
#define LED_RED_PIN     GPIO_PIN_14
#define LED_BLUE_PIN    GPIO_PIN_15
#define LED_GPIO_PORT   GPIOD

/* ---- On-board microphone (MP45DT02 via I2S2) ---- */
/* I2S2: SCK=PB13, SD=PC3, WS=PB12                   */
/* The MP45DT02 outputs PDM; we use the BSP PDM filter */

/* ---- I2S2 handles (defined in main.c) ---- */
extern I2S_HandleTypeDef hi2s2;
extern DMA_HandleTypeDef hdma_i2s2_ext_rx;

/* ---- error handler ---- */
void Error_Handler(void);

#ifdef __cplusplus
}
#endif
#endif /* __MAIN_H */

