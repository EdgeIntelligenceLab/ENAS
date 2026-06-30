# ENAS — Efficient Hardware-Aware Neural Architecture Search for TinyML

> **Anonymous repository for double-blind review at SuRE @ IJCAI 2026.**
> All identifying information has been removed.
> This repository accompanies the paper *"ENAS: An Efficient Hardware-Aware Neural Architecture Search Framework for TinyML on Resource-Constrained Microcontrollers"*.

> 📎 Alternative anonymous mirror: <https://anonymous.4open.science/r/ENAS-F882/>

---

## Table of Contents

- [Overview](#overview)
- [TinyML Motivation](#tinyml-motivation)
- [ENAS Framework Summary](#enas-framework-summary)
- [Supported Hardware](#supported-hardware)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Experiments](#reproducing-experiments)
- [Reproducing Figures and Tables](#reproducing-figures-and-tables)
- [Repository Structure](#repository-structure)
- [Reproducibility Notes](#reproducibility-notes)

---

## Overview

We present **ENAS**, a hardware-aware Neural Architecture Search framework that combines:

1. **Static analytical feasibility check** — eliminates expensive late-stage TFLite conversion during candidate evaluation
2. **Cell-based search space** — supports standard, depthwise-separable, and bottleneck blocks with optional skip connections
3. **Three-stage hybrid search** — random → top-K → mutation, with persistent cross-run caching
4. **CPU-only operation** — no GPU acceleration required, suitable for resource-constrained development environments

### Headline Results (from the paper)

- **2.41× mean search-time speedup** on Visual Wake Words
- **1.70× mean search-time speedup** on Melanoma Cancer
- **79.4% TFLite INT8 accuracy** on STM32H743ZI @ 80×80 (best result; +10.6 pp over NanoNAS)
- **636 fully trained models** across 8 MCUs × 9 input resolutions × 3 runs × 2 methods × 2 datasets
- **Zero deployment pipeline failures**
- **~297 CPU-hours** total compute (no GPU)

---

## TinyML Motivation

Deploying deep neural networks on microcontrollers requires models that simultaneously satisfy three hardware constraints:

- **RAM** — peak inference memory (typically 20 KB – 1 MB)
- **Flash** — model weight storage (64 KB – 2 MB)
- **MACC** — multiply-accumulate operations per inference (0.75 M – 15 M)

Recent TinyML NAS frameworks such as MCUNet and MicroNAS achieve strong accuracy but require substantial GPU resources (~300 GPU-hours for MCUNet on a single VWW search). CPU-only baselines such as NanoNAS improve accessibility but suffer from three limitations:

| Limitation | Impact |
|---|---|
| **L1 — Search-time bottleneck** | Repeated TFLite conversion consumes 60–76% of total runtime |
| **L2 — Narrow search space** | The 2-D (k, c) parameterisation excludes depthwise-separable, bottleneck, skip connections, stride control |
| **L3 — Greedy instability** | Per-board accuracy varies by up to 13 pp between runs on STM32H743 @ 96×96 |

ENAS addresses all three limitations while remaining CPU-only.

---

## ENAS Framework Summary

### Pipeline (4 phases)

```
┌──────────┐   ┌──────────────────────────┐   ┌──────────┐   ┌──────────┐
│ Phase 0  │   │ Phase 1: 3-Stage Search  │   │ Phase 2  │   │ Phase 3  │
│Feasibility│ → │ Random → Top-K → Mutate │ → │   Full   │ → │ INT8 PTQ │
│ (Eq. 3)  │   │  30  →  8(K)   → 40     │   │ Training │   │  + Eval  │
└──────────┘   └──────────────────────────┘   └──────────┘   └──────────┘
```

### Cell-Based Search Space

Each architecture is described by:
```
A = (k, [c₁, c₂, …, cₙ])
```
where `k ∈ [1, 12]` is the stem filter count, `n ∈ [1, 5]` is the number of cells, and each cell `cᵢ = (b, k, s, g, a, e)` has six parameters:

| Parameter | Values | Role |
|---|---|---|
| `b` (block type) | standard, depthwise-separable, bottleneck | |
| `k` (kernel) | 3, 5 | spatial kernel |
| `s` (stride) | 1, 2 | replaces explicit pooling |
| `g` (skip) | true, false | residual shortcut |
| `a` (activation) | relu, relu6 | INT8-friendly choice |
| `e` (expansion) | 1, 2 | bottleneck width multiplier |

**Total search space: ~5 × 10¹¹ candidate architectures.**

### Scoring Function (Paper Eq. 6)

```
s = w_acc · v_acc
  + w_eff · (1 - cube_root(r/R_max · f/F_max · m/M_max))
  + w_hr  · h
```

with `h = 1` if all three ratios ≤ 0.8, else 0.

Calibrated values from the paper: `(w_acc, w_eff, w_hr) = (0.80, 0.15, 0.05)` with `E_p = 3` proxy epochs on 20% of training data.

### Calibration Result (Paper Table 1)

| Proxy configuration | Mean NAS acc. | Rank correlation ρ |
|---|:---:|:---:|
| E_p=1, 20% data | 60.4% | 0.18 (not significant) |
| + E_p=3 | 65.2% | 0.59 (p<.05) |
| **E_p=3 + w_acc=0.80** | **65.2%** | **0.71 (p<.001)** |
| + 30% data (probe) | 66.8% | 0.74 (p<.001) |

Raising proxy epochs from 1 to 3 is the dominant calibration fix.

---

## Supported Hardware

The paper evaluates on 8 MCU platforms spanning three resource tiers (Table 2 in the paper):

| Platform | RAM | Flash | MACC | Tier |
|---|:---:|:---:|:---:|:---:|
| STM32L010RBT6 | 20 KB | 128 KB | 0.75 M | Ultra-constrained |
| NUCLEO-L010RB | 20 KB | 64 KB | 0.75 M | Ultra-constrained |
| Arduino Nano 33 IoT | 32 KB | 256 KB | 1.20 M | Constrained |
| NUCLEO-L412KB | 64 KB | 128 KB | 3.20 M | Moderate |
| Raspberry Pi Pico | 264 KB | 2 MB | 3.00 M | Moderate |
| Arduino Nano 33 BLE | 256 KB | 1 MB | 4.00 M | Capable |
| Arduino Nicla Vision | 1 MB | 2 MB | 8.00 M | High-capability |
| STM32H743ZI | 1 MB | 2 MB | 15.0 M | High-capability |

Configuration files: `configs/hardware/*.yaml`. Of 8 × 9 = 72 (hardware, resolution) cells, **53 are feasible** and 19 are correctly rejected by the Phase-0 analytical check (Eq. 3).

Additional platforms (Raspberry Pi 5, Jetson Nano) are included for optional transfer evaluation but are not part of the main paper table.

---

## Datasets

This repository supports the two TinyML benchmarks used in the paper:

### Visual Wake Words (VWW)
Binary classification (person / no person) derived from MS-COCO 2014. Baseline accuracy ~73% on the NanoNAS architecture family.

Generate the dataset from raw COCO 2014:
```bash
python datasets/generate_vww_dataset.py \
    --coco-root /path/to/coco \
    --output datasets/vww/
```

### Melanoma Cancer
Binary classification (benign / malignant) of dermoscopic images from the ISIC archive. Baseline accuracy ~89% (stronger class signal than VWW).

Acquisition instructions in `datasets/README.md`. Configuration files: `configs/datasets/{vww,melanoma}.yaml`.

---

## Installation

### Option 1 — pip

```bash
git clone <anonymous-url>
cd enas-tinyml
pip install -r requirements.txt
```

### Option 2 — conda

```bash
conda env create -f environment.yml
conda activate enas-tinyml
```

### Option 3 — Docker (recommended for review)

```bash
docker build -t enas-tinyml .
docker run -it -v $PWD:/workspace enas-tinyml
```

### Verify installation

```bash
python -c "from enas import ENAS; print('ENAS installed successfully')"
python scripts/run_smoke_test.py
```

---

## Quick Start

Single experiment on one hardware platform at one input size:

```bash
python scripts/run_single_experiment.py \
    --hardware STM32H743ZI \
    --size 80 \
    --dataset vww \
    --epochs 100
```

Expected outcome: TFLite INT8 accuracy ≈ 79% (matches paper's best result of 79.4%).

Results written to `results/single_runs/enas_STM32H743ZI_size80_run1_<timestamp>/`.

---

## Reproducing Experiments

### Full paper reproduction (636 models, ~297 CPU-hours)

```bash
# VWW sweep: ENAS + NanoNAS
python scripts/run_all_experiments.py --method enas    --dataset vww
python scripts/run_all_experiments.py --method nanonas --dataset vww

# Melanoma sweep: ENAS + NanoNAS
python scripts/run_all_experiments.py --method enas    --dataset melanoma
python scripts/run_all_experiments.py --method nanonas --dataset melanoma
```

Each sweep covers 8 hardware × 9 input resolutions × 3 runs. The 19 infeasible cells are automatically skipped by the Phase-0 check.

Estimated wall-clock time on a 16-core CPU server: **5–7 days** total. We recommend running each (dataset, method) pair on a separate node.

### Selective reproduction

```bash
# Single hardware, all sizes:
python scripts/run_all_experiments.py --hardware STM32H743ZI --dataset vww

# All hardware, single size (e.g., to reproduce paper Table 4):
python scripts/run_all_experiments.py --size 50 --dataset vww
python scripts/run_all_experiments.py --size 64 --dataset vww

# Reproduce best result (STM32H743 @ 80×80):
python scripts/run_single_experiment.py --hardware STM32H743ZI --size 80
```

### Ablation experiments (Paper Table 7)

```bash
python scripts/run_ablation.py --config configs/experiments/ablation.yaml
```

### Parsing logs into CSV summaries

After experiments complete:

```bash
python scripts/parse_logs.py
python scripts/generate_summary_tables.py
```

This produces `results/parsed_csv/*.csv` and `results/summary_tables/*.csv`.

---

## Reproducing Figures and Tables

Regenerate every figure in the paper from parsed CSV results:

```bash
python figures/scripts/generate_all_figures.py
```

| Output File | Paper Reference | Generator Script |
|---|---|---|
| `pipeline_overview.pdf` | Figure 1 | `plot_pipeline.py` |
| `pareto_accuracy_vs_searchtime.pdf` | Figure 2 | `plot_pareto.py` |
| `heatmap_4panel.pdf` | Figure 3 | `plot_heatmaps.py` |
| `table_proxy_fidelity.tex` | Table 1 | `generate_summary_tables.py` |
| `table_hardware_platforms.tex` | Table 2 | `generate_summary_tables.py` |
| `table_aggregate_results.tex` | Table 3 | `generate_summary_tables.py` |
| `table_focused_50_64.tex` | Table 4 | `generate_summary_tables.py` |
| `table_per_resolution.tex` | Table 5 | `generate_summary_tables.py` |
| `table_per_hardware.tex` | Table 6 | `generate_summary_tables.py` |
| `table_ablation.tex` | Table 7 | `generate_summary_tables.py` |

Individual figure regeneration:

```bash
python figures/scripts/plot_heatmaps.py    # Figure 3 (4-panel)
python figures/scripts/plot_pareto.py      # Figure 2
```

### Heatmap colour scheme (Figure 3)

The 4-panel heatmap in Figure 3 uses a diverging green-red colour scale:
- **Green cells** — ENAS performs better/faster than NanoNAS
- **Red cells** — NanoNAS performs better/faster than ENAS
- **Grey hatched cells** — RAM-infeasible (rejected by Phase-0 analytical check)

---

## Repository Structure

```
enas-tinyml/
├── README.md
├── LICENSE                          (MIT)
├── CONTRIBUTING.md
├── .gitignore
├── requirements.txt
├── environment.yml
├── Dockerfile
├── setup.py
│
├── enas/                            # ENAS framework
│   ├── enas_v2_1.py                 #   Main NAS class
│   ├── search_space.py              #   Cell-based search space
│   ├── blocks.py                    #   Block builders
│   ├── estimators.py                #   Analytical RAM/Flash/MACC
│   ├── scoring.py                   #   Multi-objective scoring (Eq. 6)
│   ├── mutations.py                 #   Stage 3 mutation strategies
│   └── quantization.py              #   INT8 PTQ utilities
│
├── nanonas/                         # NanoNAS baseline
│   └── nanonas.py
│
├── enas_v1/                         # ENAS-strategy ablation (Paper §5.5)
│   └── enas_v1.py
│
├── datasets/
│   ├── vww_loader.py
│   ├── melanoma_loader.py
│   ├── generate_vww_dataset.py
│   └── README.md
│
├── configs/
│   ├── hardware/                    # YAML constraint files (8 MCUs + 2 SBC)
│   ├── experiments/                 # ENAS/NanoNAS hyperparameter presets
│   └── datasets/                    # VWW + Melanoma configs
│
├── scripts/                         # Top-level reproducibility scripts
│   ├── run_single_experiment.py
│   ├── run_all_experiments.py
│   ├── run_ablation.py
│   ├── parse_logs.py
│   ├── generate_summary_tables.py
│   └── run_smoke_test.py
│
├── results/                         # Outputs (mostly git-ignored)
│   ├── raw_logs/
│   ├── parsed_csv/
│   ├── summary_tables/
│   └── models/
│
├── figures/
│   ├── output/                      # Generated PDFs
│   └── scripts/                     # Plotting code
│       ├── generate_all_figures.py
│       ├── plot_pareto.py           # Figure 2
│       ├── plot_heatmaps.py         # Figure 3 (4-panel)
│       └── style.py
│
├── notebooks/                       # Analysis Jupyter notebooks
├── paper/                           # Paper artifacts (PDF omitted for anonymity)
├── tests/                           # Unit tests
└── docs/                            # Extended documentation
```

---

## Reproducibility Notes

### Random seeds

Documented per-run seeds: **11, 42, 137** for runs 1, 2, 3 respectively. All scripts accept `--seed` to override.

### Hardware-in-the-loop validation

The analytical RAM/Flash/MACC estimators (Eq. 3 and `enas/estimators.py`) have been validated against `stm32tflm` ground-truth measurements. Mean absolute error: <8% RAM, <5% Flash, <2% MACC. Validation:
```bash
python tests/validate_estimators.py --samples 50
```

### Determinism

Floating-point summation order in TensorFlow makes runs only approximately reproducible. To approximate determinism:
```bash
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
```

Inter-run variance of ±2–4 pp accuracy is expected (matches paper Table 3 reported std. of 2.58–3.33 pp).

### Cross-experiment cache

ENAS uses a persistent JSON cache at `results/raw_logs/<exp>/enas_v2_1_cache.json`. Force fresh evaluation:
```bash
python scripts/run_single_experiment.py --no-cache ...
```

### Compute requirements

| Setup | Time | Use case |
|---|---|---|
| 1 CPU × 1 hw × 1 size × 1 run | 30–90 min | Spot-check |
| 16-core CPU server, full sweep | 5–7 days | Full reproduction (~297 CPU-hours) |
| 4 nodes × 16 cores in parallel | ~36 hours | Distributed reproduction |

No GPU is required. All paper results were produced on CPU.

### Known external dependencies

- **TFLite Lite Converter** — bundled with TensorFlow 2.13
- **stm32tflm binary** — required *only* for ground-truth hardware validation. Download from STMicroelectronics X-CUBE-AI Linux package (free registration) and place at repository root. The main ENAS pipeline uses the analytical estimator and does not require this binary during search.

---

## Citation

```bibtex
@inproceedings{anonymous2026enas,
  title  = {ENAS: An Efficient Hardware-Aware Neural Architecture Search Framework
            for TinyML on Resource-Constrained Microcontrollers},
  author = {Anonymous},
  booktitle = {SuRE @ IJCAI},
  year   = {2026},
  note   = {Under double-blind review}
}
```

---

## License

Code: MIT (see [LICENSE](LICENSE))
Datasets: VWW (CC BY 4.0 via COCO 2014 derivative), Melanoma (ISIC archive terms)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are paused during the anonymous review period.
