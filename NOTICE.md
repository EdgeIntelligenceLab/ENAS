# Attribution

ENAS is built on top of **NanoNAS** by Andrea Mattia Garavagno et al.
(MIT License), https://github.com/AndreaMattiaGaravagno/NanoNAS.

- The NanoNAS baseline (`nanonas/nanonas.py`) is adapted from the upstream `NanoNAS.py`.
- The `stm32tflm` Flash/RAM measurement tool is from STMicroelectronics' X-CUBE-AI
  package and is redistributed in the upstream NanoNAS repository. It is required only
  for measured resource validation; obtain it from the NanoNAS repo or from
  https://www.st.com/en/embedded-software/x-cube-ai.html and place it at the repo root.

Please cite NanoNAS (Garavagno et al., IEEE Sensors Letters 2024) in addition to ENAS
when using the baseline. The original NanoNAS MIT license terms are preserved.
