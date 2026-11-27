/* ESP32-S3 (ESP-IDF): score recorded sound sent from the laptop over the "USB" port.
 * The laptop script firmware/tools/stream_test.py sends raw sound one frame (1024 samples) at a time;
 * the board turns it into log-mel features and runs the model, exactly as it would with the microphone.
 *
 * Commands from the laptop (1 byte, then data):
 *   'P'                 -> reply: mel bands + model parameters (checks the board is ready)
 *   'S' + 1024 int16    -> calibration: add this healthy frame to the feature mean/spread (short reply)
 *   'D'                 -> calibration done (reply)
 *   'F' + 1024 int16    -> reply: score of this frame (float) + time taken on the chip (microseconds)
 *   'L' + 1 byte        -> set the RGB LED: 0 off, 1 green (healthy), 2 red (fault), 3 blue (reply)
 *   'R'                 -> restart the board (clears the calibration)
 * Every reply starts with the byte 0xA5 and the command letter, so start-up text can be skipped.
 * The laptop must wait for replies and keep only a few frames in flight: the USB driver drops incoming
 * bytes when its buffer is full, and lost bytes would make the board misread sound as commands. */
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/usb_serial_jtag.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "led_strip.h"
#include "ae_pipeline.h"

#define LED_GPIO 48            /* the board's RGB LED (WS2812); some boards use 38 */

static int16_t pcm[N_FFT];
static led_strip_handle_t led;

static void led_set(uint8_t c)     /* 0 off, 1 green, 2 red, 3 blue (kept dim) */
{
    if (!led) return;
    static const uint8_t rgb[4][3] = {{0, 0, 0}, {0, 40, 0}, {40, 0, 0}, {0, 0, 40}};
    if (c > 3) c = 0;
    led_strip_set_pixel(led, 0, rgb[c][0], rgb[c][1], rgb[c][2]);
    led_strip_refresh(led);
}

static void led_init(void)
{
    led_strip_config_t sc = {
        .strip_gpio_num = LED_GPIO,
        .max_leds = 1,
        .led_model = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
    };
    led_strip_rmt_config_t rc = {.resolution_hz = 10 * 1000 * 1000};
    if (led_strip_new_rmt_device(&sc, &rc, &led) != ESP_OK) led = NULL;
    led_set(0);
}
static float lm[N_MELS];

static void read_exact(void *buf, uint32_t n)
{
    uint8_t *p = buf;
    while (n) {
        int r = usb_serial_jtag_read_bytes(p, n, portMAX_DELAY);
        if (r > 0) { p += r; n -= r; }
    }
}

static void reply(uint8_t tag, const void *data, uint32_t n)
{
    uint8_t hdr[2] = {0xA5, tag};
    usb_serial_jtag_write_bytes(hdr, 2, portMAX_DELAY);
    if (n) usb_serial_jtag_write_bytes(data, n, portMAX_DELAY);
}

void app_main(void)
{
    usb_serial_jtag_driver_config_t cfg = {.tx_buffer_size = 1024, .rx_buffer_size = 8192};
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));
    led_init();

    for (;;) {
        uint8_t cmd;
        read_exact(&cmd, 1);
        switch (cmd) {
        case 'P': {
            uint32_t info[2] = {N_MELS, MODEL_PARAMS};
            reply('P', info, sizeof info);
            break;
        }
        case 'S':
            read_exact(pcm, sizeof pcm);
            ae_logmel(pcm, lm);
            ae_cal_stats_add(lm);
            reply('S', NULL, 0);
            break;
        case 'D':
            ae_cal_stats_done();
            reply('D', NULL, 0);
            break;
        case 'F': {
            read_exact(pcm, sizeof pcm);
            int64_t t0 = esp_timer_get_time();
            ae_logmel(pcm, lm);
            float s = ae_score(lm);
            uint32_t us = (uint32_t)(esp_timer_get_time() - t0);
            uint8_t out[8];
            memcpy(out, &s, 4);
            memcpy(out + 4, &us, 4);
            reply('F', out, sizeof out);
            break;
        }
        case 'L': {                     /* LED colour: 1 byte, 0 off 1 green 2 red 3 blue */
            uint8_t c;
            read_exact(&c, 1);
            led_set(c);
            reply('L', NULL, 0);
            break;
        }
        case 'R':
            led_set(0);
            vTaskDelay(pdMS_TO_TICKS(50));
            esp_restart();
            break;
        default:
            break;   /* ignore anything else */
        }
    }
}
