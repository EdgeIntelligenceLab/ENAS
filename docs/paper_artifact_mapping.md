# Paper Artifact Mapping

This document maps each figure, table, and headline claim in the camera-ready
paper to the code/data that produces it.

## Figures

| Figure | Description | Generator | Output |
| --- | --- | --- | --- |
| Figure 1 | ENAS pipeline + architecture template | Hand-drawn TikZ in the paper | n/a |
| Figure 2 | Analytical activation-memory feasibility boundary | `figures/scripts/plot_feasibility.py` | `figures/output/feasibility_matrix.png` |
| Figure 3 | Accuracy vs. search-time Pareto (212 cells) | `figures/scripts/plot_pareto.py` | `figures/output/fig4_pareto.pdf` |
| Figure 4 | 4-panel per-cell Δ heatmaps (acc / search time) | `figures/scripts/plot_heatmaps.py` | `figures/output/fig_delta_combined.pdf` |

## Tables

| Table | Description | Source |
| --- | --- | --- |
| Table 1 | Target MCU platforms (8 hardware specs) | static — `docs/hardware_specs.md` |
| Table 2 | Aggregate results (mean±std acc, search, speedup, Wilcoxon p) | `scripts/build_camera_ready_tables.py` |
| Table 3 | Measured resource footprint of selected models (RAM/Flash via stm32tflm, MACC analytical) | `scripts/build_camera_ready_tables.py` |
| Table 4 | Focused 50×50 / 64×64 study on VWW (warm-seeded) | `scripts/build_camera_ready_tables.py` |
| Table 5 | Per-resolution accuracy + search-time sweep | `scripts/build_camera_ready_tables.py` |
| Table 6 | Per-hardware best TFLite accuracy on VWW | `scripts/build_camera_ready_tables.py` |
| Table 7 | Ablation of design and calibration choices | `scripts/run_ablation.py` |

## Headline claims

| Claim | Location | How to verify |
| --- | --- | --- |
| 2.41× / 1.70× mean speedup | Table 2, §5.1 | `python scripts/build_camera_ready_tables.py` |
| Speedup significant p<1e-8; VWW accuracy n.s. (p=0.118) | Table 2 | same (Wilcoxon printed) |
| ENAS uses ~0.38× peak RAM at matched accuracy | Table 3, §5.2 | same (Table 3 block) |
| 79.4% on STM32H743 @ 80×80 | Table 6, §5.5 | inspect `results/parsed_csv/enas_vww_summary.csv` (STM32H743ZI, size 80) |
| Search space k ∈ {1,…,16} | §3.2 | `enas/search_space.py` (`list(range(1,17))`) |

## Ablation variants (Table 7)

Run via `python scripts/run_ablation.py --config configs/experiments/ablation.yaml`.
NOTE: the ENAS-strategy row requires the runnable `enas_v1/enas_v1.py` implementation.
