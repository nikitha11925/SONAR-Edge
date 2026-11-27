idf_component_register(SRCS "main.c"
                            "../../common/ae_pipeline.c"
                            "../../common/model_data.c"
                       INCLUDE_DIRS "../../common"
                       PRIV_REQUIRES esp_driver_usb_serial_jtag esp_timer espressif__led_strip)
