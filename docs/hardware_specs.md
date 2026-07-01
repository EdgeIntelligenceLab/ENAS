# Hardware Platform Specifications

All values are vendor-published. SRAM and Flash are device capacities;
MACC budget is the per-inference allowance computed from typical clock
speed and target inference latency.

| Platform | SRAM | Flash | MACC budget | Clock | Family |
|---|:---:|:---:|:---:|:---:|:---:|
| STM32L010RBT6     | 20 KB | 128 KB | 0.75 M | 32 MHz | STM32L0 |
| NUCLEO-L010RB     | 20 KB | 64 KB  | 0.75 M | 32 MHz | STM32L0 |
| Arduino Nano33IoT | 32 KB | 256 KB | 1.20 M | 48 MHz | SAMD21 |
| NUCLEO-L412KB     | 64 KB | 128 KB | 3.20 M | 80 MHz | STM32L4 |
| Raspberry Pi Pico | 264 KB | 2 MB  | 3.00 M | 133 MHz | RP2040 |
| Arduino Nano33BLE | 256 KB | 1 MB  | 4.00 M | 64 MHz | nRF52840 |
| Arduino Nicla Vision | 1 MB | 2 MB | 8.00 M | 480 MHz | STM32H7 |
| STM32H743ZI       | 1 MB | 2 MB   | 15.0 M | 480 MHz | STM32H7 |

## Cores

- **Cortex-M0+**: STM32L010, Nano33IoT, RP Pico
- **Cortex-M4F**: NUCLEO-L412, Nano33BLE
- **Cortex-M7**:  Nicla Vision (+ M4 second core), STM32H743ZI

## Memory mapping

For all listed platforms, SRAM is fully available to the user application
when running TFLite-Micro. Flash holds both the model weights and the
application code; the MACC budget assumes a typical inference latency
budget of 100 ms per frame.
