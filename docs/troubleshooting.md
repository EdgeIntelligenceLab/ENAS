# Troubleshooting

## Issue: `/tmp` disk full during TFLite conversion

Symptom: `OSError: No space left on device`

Cause: TensorFlow Lite converter writes intermediate files to `/tmp`.

Fix: Set a custom TMPDIR with sufficient space:
```bash
export TMPDIR=/path/to/large/disk/tmp
```

The ENAS quantization module also redirects converter temp files automatically.

## Issue: `stm32tflm: not found`

Cause: The stm32tflm binary is required only for ground-truth hardware
validation, not for the main ENAS pipeline.

Fix: Download from STMicroelectronics X-CUBE-AI Linux package and place
at the repository root.

## Issue: VWW dataset generator runs out of memory

Cause: Loading the full COCO 2014 annotation JSON requires ~3 GB RAM.

Fix: Run on a machine with ≥8 GB RAM or process splits sequentially.

## Issue: Long search times even with ENAS

Cause: First run on a new (hardware, input size) combination builds the
cache from scratch.

Fix: Cache is persistent across runs. The second and subsequent runs at
the same (hardware, size) combination will be significantly faster.

## Issue: Inter-run variance higher than expected

Cause: Random sampling in Stage 1 may visit different regions of the
search space.

Fix: Increase `n_random_stage` from 30 to 50 in the config, or pass
`--seed` to fix the random state.
