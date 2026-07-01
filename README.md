# ENAS — Efficient Hardware-Aware Neural Architecture Search for TinyML

This repository accompanies the paper
**"ENAS: An Efficient Hardware-Aware Neural Architecture Search Framework for TinyML on Resource-Constrained Microcontrollers"**,
Mohd Moin Khan, Naman Srivastava, and Pandarasamy Arjunan
(Edge Intelligence Lab, Indian Institute of Science, Bengaluru),
SuRE Workshop @ IJCAI 2026.

---

## Overview

**ENAS** is a CPU-only hardware-aware Neural Architecture Search framework that combines:

1. **Static analytical feasibility check** — replaces NanoNAS's *measured* per-candidate TFLite/stm32tflm check with an *analytical* pre-flight screen (Eq. 3).
2. **Cell-based search space** — standard, depthwise-separable, and bottleneck blocks with optional skip connections, stride control, and `relu`/`relu6`.
3. **Three-stage hybrid search** — random → top-K → mutation, with persistent cross-run caching.
4. **CPU-only operation** — no GPU required.

### Headline results (from the paper)

- **2.41× / 1.70× mean search-time speedup** on Visual Wake Words / Melanoma Cancer (both highly significant, Wilcoxon *p* < 10⁻⁸).
- **Accuracy cost is small**: −0.79 pp (VWW, *not* statistically significant, *p* = 0.118) and −1.31 pp (Cancer).
- **Measured resource trade-off** (Table 3): ENAS-selected models use **~0.38× the peak activation RAM** of NanoNAS at matched accuracy — the binding MCU constraint — in exchange for higher Flash and MACC.
- **79.4% TFLite INT8 accuracy** on STM32H743ZI @ 80×80 (best result).
- **636 fully trained models**, **zero deployment-pipeline failures**, **~297 CPU-hours** total (no GPU).

> **Reproducibility at a glance.** The parsed result CSVs for all 636 runs are shipped in
> [`results/parsed_csv/`](results/parsed_csv). You can regenerate the paper's main tables
> (mean ± std and significance) in seconds, without re-running the sweep:
> ```bash
> python scripts/build_camera_ready_tables.py
> ```

---

## ENAS framework summary

### Cell-based search space

```
A = (k, [c1, c2, …, cn])
```

where **`k ∈ {1, …, 16}`** is the stem filter count and **`n ∈ {1, …, 6}`** is the number of cells.
During search, Stage-1 random sampling draws `k ≤ 12`; Stage-3 mutation samples `k` from the full
`{1, …, 16}` range, so realised architectures span `k ∈ {1, …, 16}` (selected values in our runs
range from 1 to 16). Each cell `ci = (b, k, s, g, a, e)`:

| Parameter        | Values                                    | Role                        |
| ---------------- | ----------------------------------------- | --------------------------- |
| `b` (block type) | standard, depthwise-separable, bottleneck | primitive                   |
| `k` (kernel)     | 3, 5                                      | spatial kernel              |
| `s` (stride)     | 1, 2                                      | replaces explicit pooling   |
| `g` (skip)       | true, false                               | residual shortcut           |
| `a` (activation) | relu, relu6                               | INT8-friendly choice        |
| `e` (expansion)  | 1, 2                                      | bottleneck width multiplier |

Per-cell channel counts are **not searched independently**; they are derived from the stem width `k`
via a fixed decaying-multiplier schedule. After fixing this schedule, the structural search space is
on the order of **~10⁴ configurations**, made tractable by the hybrid search.

### Scoring function (paper Eq. 6)

```
s = w_acc · v_acc
  + w_eff · (1 − cube_root(r/R_max · f/F_max · m/M_max))
  + w_hr  · h          # h = 1 iff all three ratios ≤ 0.8, else 0
```

Calibrated values: `(w_acc, w_eff, w_hr) = (0.80, 0.15, 0.05)`, `E_p = 3` proxy epochs on
**30% of training data** (`proxy_data_fraction = 0.30`). Raising `E_p` from 1 to 3 lifts the
proxy/final-accuracy Spearman correlation from ρ = 0.18 (n.s.) to ρ = 0.71 (*p* < 0.001).

### Important note on the analytical estimators

The analytical RAM/Flash/MACC estimators (`enas/estimators.py`, Eq. 3) are used **only for
pre-flight feasibility screening**, not as accurate footprint predictions. Validated against
`stm32tflm` ground truth, the **RAM estimate is conservative (over-predicts ~3×)** and the **Flash
estimate is a loose lower bound (under-predicts ~6.5×)**. This is safe because the binding MCU
constraint is peak RAM and the RAM estimate errs conservatively. **All resource numbers reported in
the paper (Table 3) are measured with `stm32tflm`, not analytical.**

---

## Supported hardware

| Platform             | RAM    | Flash  | MACC   | Tier              |
| -------------------- | ------ | ------ | ------ | ----------------- |
| STM32L010RBT6        | 20 KB  | 128 KB | 0.75 M | Ultra-constrained |
| NUCLEO-L010RB        | 20 KB  | 64 KB  | 0.75 M | Ultra-constrained |
| Arduino Nano 33 IoT  | 32 KB  | 256 KB | 1.20 M | Constrained       |
| NUCLEO-L412KB        | 64 KB  | 128 KB | 3.20 M | Moderate          |
| Raspberry Pi Pico    | 264 KB | 2 MB   | 3.00 M | Moderate          |
| Arduino Nano 33 BLE  | 256 KB | 1 MB   | 4.00 M | Capable           |
| Arduino Nicla Vision | 1 MB   | 2 MB   | 8.00 M | High-capability   |
| STM32H743ZI          | 1 MB   | 2 MB   | 15.0 M | High-capability   |

Of 8 × 9 = 72 (hardware, resolution) cells, **53 are feasible** and 19 are rejected by the Phase-0
analytical check. Config files: `configs/hardware/*.yaml`.

---

## Datasets

- **Visual Wake Words (VWW)** — binary person/no-person from MS-COCO 2014. Generate with
  `python dataset/generate_vww_dataset.py --coco-root /path/to/coco --output dataset/vww/`.
- **Melanoma Cancer** — binary benign/malignant dermoscopic images (ISIC archive). See
  `dataset/README.md`.

---

## Installation

```bash
git clone https://github.com/EdgeIntelligenceLab/ENAS.git
cd ENAS
# Option 1 — pip
pip install -r requirements.txt
# Option 2 — conda
conda env create -f environment.yml && conda activate enas-tinyml
# Option 3 — Docker
docker build -t enas-tinyml . && docker run -it -v "$PWD":/workspace enas-tinyml
```

Verify: `python scripts/run_smoke_test.py`

---

## Reproducing the paper

### Step 0 — set up datasets (only needed for from-scratch runs)

See [`dataset/README.md`](dataset/README.md) for full instructions: download COCO 2014 and run
`dataset/generate_vww_dataset.py` + `dataset/create_test_dataset.py` for VWW, and download the
Melanoma dataset from Kaggle. Path A below needs no datasets.

### A) Regenerate tables from shipped results (seconds, recommended)

The parsed CSVs for all 636 runs are in `results/parsed_csv/`. Regenerate the main tables:

```bash
python scripts/build_camera_ready_tables.py        # Tables 2, 3, 5 (+ significance)
```

### B) Re-run the full sweep from scratch (~297 CPU-hours)

```bash
python scripts/run_all_experiments.py --method enas    --dataset vww
python scripts/run_all_experiments.py --method nanonas --dataset vww
python scripts/run_all_experiments.py --method enas    --dataset melanoma
python scripts/run_all_experiments.py --method nanonas --dataset melanoma
python scripts/parse_logs.py                       # raw logs -> results/parsed_csv/*.csv
```

### C) Measured resource footprints (Table 3)

`stm32tflm` measurements of the selected models are in
`results/parsed_csv/enas_measured_resources.csv`. To regenerate from saved `.tflite` files use the
batch measurement script and your local `stm32tflm` binary.

---

## Paper artifact mapping

| Paper item | Description | Source |
| --- | --- | --- |
| Figure 1 | ENAS pipeline + architecture template | TikZ in paper |
| Figure 2 | Analytical activation-memory feasibility boundary | `figures/scripts/plot_feasibility.py` |
| Figure 3 | Accuracy vs. search-time Pareto (212 cells) | `figures/scripts/plot_pareto.py` |
| Figure 4 | 4-panel per-cell Δ heatmaps (acc / search time) | `figures/scripts/plot_heatmaps.py` |
| Table 1 | Target MCU platforms | `docs/hardware_specs.md` (static) |
| Table 2 | Aggregate results + Wilcoxon significance | `scripts/build_camera_ready_tables.py` |
| **Table 3** | **Measured resource footprint of selected models** | `scripts/build_camera_ready_tables.py` |
| Table 4 | Focused 50×50 / 64×64 study (warm-seeded) | `scripts/build_camera_ready_tables.py` |
| Table 5 | Per-resolution sweep | `scripts/build_camera_ready_tables.py` |
| Table 6 | Per-hardware best (VWW) | `scripts/build_camera_ready_tables.py` |
| Table 7 | Ablation of design/calibration choices | `scripts/run_ablation.py` |

---

## Repository structure

```
enas/            ENAS v2.1 framework (search space, blocks, estimators, scoring, mutations, PTQ)
nanonas/         NanoNAS baseline
enas_v1/         ENAS-strategy ablation variant (2-D (k,c) + random search)  [see note below]
dataset/         Dataset generators and loaders
configs/         Hardware / dataset / experiment YAML configs
scripts/         Experiment runners, log parsing, table generation
figures/         Plotting code and generated figures
results/         parsed_csv/ ships the 636-run summaries; raw_logs/models are git-ignored
tests/           Unit tests + estimator validation
docs/            Extended documentation
```

> **Note on `enas_v1/`.** The ENAS-strategy ablation (paper §5.5, Table 7) uses the 2-D `(k, c)`
> space with parallel random search and the analytical pre-flight check. Ensure the runnable
> implementation (`enas_v1/enas_v1.py`) is present before invoking `scripts/run_ablation.py`.

---

## Reproducibility notes

- **Seeds.** Dataset shuffling uses seed 11 (full training) and 42 (proxy split); pass `--seed` to override per run.
- **Determinism.** TensorFlow float summation order makes runs only approximately reproducible; inter-run std of ±2–4 pp is expected and is reported as mean ± std in every results table.
- **Cache.** ENAS uses a persistent JSON cache; pass `--no-cache` to force fresh evaluation.
- **stm32tflm.** Needed only for measured RAM/Flash. Download from the STMicroelectronics X-CUBE-AI Linux package and place at the repo root.

---

## Citation

```bibtex
@inproceedings{khan2026enas,
  title     = {ENAS: An Efficient Hardware-Aware Neural Architecture Search Framework
               for TinyML on Resource-Constrained Microcontrollers},
  author    = {Khan, Mohd Moin and Srivastava, Naman and Arjunan, Pandarasamy},
  booktitle = {Proceedings of the SuRE Workshop at IJCAI 2026},
  year      = {2026},
  publisher = {CEUR-WS.org}
}
```

## Acknowledgements

ENAS builds on **NanoNAS** (Garavagno et al., MIT) — see [`NOTICE.md`](NOTICE.md). Please cite NanoNAS alongside ENAS when using the baseline.

## License

Code: MIT (see `LICENSE`). Datasets: VWW (CC BY 4.0 via COCO 2014 derivative); Melanoma (ISIC archive terms).
