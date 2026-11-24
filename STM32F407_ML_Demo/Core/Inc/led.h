/* led.h – Green / Red LED abstraction for STM32F407-Discovery */
#ifndef LED_H_
#define LED_H_

void led_init(void);
void led_green(void);   /* solid green  = NORMAL  */
void led_red(void);     /* solid red    = ANOMALY  */
void led_orange(void);  /* solid orange = calibrating */
void led_all_off(void);

#endif

