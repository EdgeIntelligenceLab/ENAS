# Reproducibility Guide

## Compute Budget

The full paper sweep (636 trained models) requires approximately:
- 297 CPU-hours total on a 16-core node
- ~5–7 days wall-clock on a single 16-core CPU server
- ~36 hours wall-clock distributed across 4 nodes

No GPU is required.

## Inter-Run Variance

Expected accuracy variance across runs (paper Table 3):
- VWW:    std ≈ 2.58 pp (NanoNAS),  3.33 pp (ENAS)
- Cancer: std ≈ 1.26 pp (NanoNAS),  1.52 pp (ENAS)

ENAS has slightly higher variance because the larger search space exposes
more architectural variation. Mean accuracy converges over 3 runs.

## Random Seeds

Per-run seeds: **11, 42, 137** (runs 1, 2, 3 respectively).

## Determinism

TensorFlow CPU operations are not fully deterministic due to floating-point
summation order. To approximate determinism:

```bash
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
```

## Validation

The analytical RAM/Flash/MACC estimators have been validated against
stm32tflm ground-truth measurements. Mean absolute error:
- RAM:   < 8 %
- Flash: < 5 %
- MACC:  < 2 %
