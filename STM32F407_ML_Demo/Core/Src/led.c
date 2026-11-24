/* led.c – LED driver for STM32F407-Discovery
 *
 * On-board LEDs:
 *   PD12 = GREEN
 *   PD13 = ORANGE
 *   PD14 = RED
 *   PD15 = BLUE
 *
 * We only use GREEN (normal) and RED (anomaly).
 * ORANGE is used during the calibration phase.
 * All LEDs share GPIOD; the GPIO clock must already be enabled
 * (MX_GPIO_Init does this automatically in CubeIDE).
 */
#include "main.h"
#include "led.h"

static inline void set(uint16_t on, uint16_t off)
{
    HAL_GPIO_WritePin(LED_GPIO_PORT, on,  GPIO_PIN_SET);
    HAL_GPIO_WritePin(LED_GPIO_PORT, off, GPIO_PIN_RESET);
}

void led_init(void)
{
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_GPIOD_CLK_ENABLE();
    g.Pin   = LED_GREEN_PIN | LED_ORANGE_PIN | LED_RED_PIN | LED_BLUE_PIN;
    g.Mode  = GPIO_MODE_OUTPUT_PP;
    g.Pull  = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LED_GPIO_PORT, &g);
    led_all_off();
}

void led_green(void)
{
    set(LED_GREEN_PIN,
        LED_ORANGE_PIN | LED_RED_PIN | LED_BLUE_PIN);
}

void led_red(void)
{
    set(LED_RED_PIN,
        LED_GREEN_PIN | LED_ORANGE_PIN | LED_BLUE_PIN);
}

void led_orange(void)
{
    set(LED_ORANGE_PIN,
        LED_GREEN_PIN | LED_RED_PIN | LED_BLUE_PIN);
}

void led_all_off(void)
{
    HAL_GPIO_WritePin(LED_GPIO_PORT,
        LED_GREEN_PIN | LED_ORANGE_PIN | LED_RED_PIN | LED_BLUE_PIN,
        GPIO_PIN_RESET);
}

