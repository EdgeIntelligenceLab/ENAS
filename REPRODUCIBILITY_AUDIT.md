# Reproducibility Audit — ENAS repository

**Verdict on the original repository (`github.com/EdgeIntelligenceLab/ENAS`, 2 commits):**
**Not reproducible as shipped.** The framework code is real and well-structured, but the
artifact could not regenerate any paper result, several documented claims contradicted the
camera-ready paper, and one claim contradicted the paper's own measured data. This corrected
package fixes the documentation/consistency/integrity issues and ships the result data so the
main tables are regenerable in seconds. Two gaps still require the authors (listed at the end).

---

## What was already good

- `enas/` is a substantive modular implementation (not stubs): search space, blocks,
  estimators, scoring, mutations, PTQ.
- `enas/search_space.py` already uses the correct `k ∈ {1,…,16}` (`list(range(1,17))`).
- `enas/scoring.py` already uses the correct v2.1 weights `(0.80, 0.15, 0.05)` and `h ≤ 0.8`.
- `nanonas/nanonas.py` baseline is present and real.

## Blockers found (and how this package fixes them)

| # | Severity | Issue (original repo) | Fix in this package |
| - | -------- | --------------------- | ------------------- |
| 1 | **High** | **No results shipped.** `results/` and `figures/output/` contained only `.gitkeep`, so no table or figure could be regenerated without the full ~297 CPU-hour sweep. | Shipped all six parsed-result CSVs in `results/parsed_csv/` + `PROVENANCE.md`. |
| 2 | **High** | **No verified path from data → tables.** `generate_summary_tables.py` expected a CSV schema (`input_size`, `hardware`, `enas_vww_summary.csv` …) that the real logs do not use, and the inputs were absent anyway. | Added **`scripts/build_camera_ready_tables.py`**, tested here against the shipped CSVs; it reproduces Tables 2, 3, 5 (mean±std + Wilcoxon) exactly. |
| 3 | **High (integrity)** | **False estimator-accuracy claim.** `estimators.py` and `validate_estimators.py` and `docs/reproducibility.md` claimed “<8% RAM, <5% Flash” MAE. The paper's own measured analysis shows the analytical Flash *under-predicts ~6.5×* and RAM *over-predicts ~3×*. A reviewer running the validator would see the contradiction. | Corrected all three to state the estimator is a *conservative feasibility screen only*, and that Table 3 uses MEASURED stm32tflm values. |
| 4 | **High** | **`enas_v1/` (ENAS-strategy ablation) was a stub.** | **RESOLVED:** real `enas_v1/enas_v1.py` (790 lines) added and exported via `enas_v1/__init__.py`; Table 7 / §5.5 reproducible. |
| 5 | Med | **README/docs contradicted the camera-ready:** `k ∈ [1,12]` (reviewers flagged this exact error), search space “~5×10¹¹”, proxy data “20%”, and table/figure numbering that omitted the new resource table entirely. | README + `docs/paper_artifact_mapping.md` rewritten to match the camera-ready: `k∈{1,…,16}`, ~10⁴ structural configs, 30% proxy data, 4 figures, Table 1=hardware … Table 3=**resource** … Table 7=ablation. |
| 6 | Med | **Anonymity half-state:** hosted under the real lab name but README/citation/CONTRIBUTING still said “Anonymous”. | De-anonymized: real authors + affiliation, real BibTeX, `CITATION.cff`, anonymity section removed from CONTRIBUTING, `setup.py` URL fixed. |
| 7 | Low | **`Dockerfile` referenced but missing**; `dataset/` vs `datasets/` path mismatch in README; duplicate top-level vs `scripts/` entry points. | Added a working CPU `Dockerfile`; corrected paths; README now points at `scripts/`. |

## Verified in this package

```
$ python scripts/build_camera_ready_tables.py
Table 2 (VWW):    NanoNAS 72.68±2.58 / ENAS 71.89±3.33   speedup 2.41x   acc p=0.118   search p=4.7e-10
Table 2 (Cancer): NanoNAS 89.44±1.26 / ENAS 88.13±1.52   speedup 1.70x   acc p=3.4e-6  search p=5.9e-9
Table 3 (VWW):    ENAS/NanoNAS RAM 0.38x  Flash 2.85x  MACC 1.39x  (matched accuracy)
Table 3 (Cancer): ENAS/NanoNAS RAM 0.38x  Flash 5.73x  MACC 3.94x  (matched accuracy)
```
All match the camera-ready exactly.

## Status of earlier gaps

- **ENAS-strategy ablation** — RESOLVED (real `enas_v1/enas_v1.py` added).
- **Dataset pipeline** — RESOLVED: `dataset/README.md` documents COCO→VWW generation
  (`generate_vww_dataset.py`, `create_test_dataset.py`) and the Kaggle Melanoma download,
  and `configs/datasets/*.yaml` point at the resulting `dataset/` layout.
- **Author requirements** — RESOLVED: real `requirements.txt` (pip freeze) and
  `conda_requirements.txt` (conda list --export) shipped.
- **NanoNAS attribution / stm32tflm** — RESOLVED: `NOTICE.md` credits NanoNAS (MIT) and
  documents obtaining `stm32tflm`.

## Remaining (environment-dependent, cannot be done without the authors' machine)

1. **Figures** — `figures/output/` ships empty; run `figures/scripts/*.py` against
   `results/parsed_csv/` to populate, or commit the generated PDFs/PNGs.
2. **Full from-scratch sweep** (~297 CPU-hours) and **stm32tflm measurement** require the
   datasets, the STM binary, and compute; only the data→tables path is verified here.
3. **Legacy `generate_summary_tables.py` / `run_ablation.py`** still assume the older CSV
   schema; `scripts/build_camera_ready_tables.py` is the verified canonical path.
