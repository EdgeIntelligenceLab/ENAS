# Provenance of shipped result CSVs

These are the parsed summaries of all 636 trained models used to produce the
paper's tables. They let you regenerate Tables 2, 3, and 5 without re-running
the ~297 CPU-hour sweep:  `python scripts/build_camera_ready_tables.py`

| File | Contents |
| --- | --- |
| `nanonas_vww_summary.csv` / `nanonas_cancer_summary.csv` | NanoNAS baseline, all feasible cells, 3 runs each (RAM/Flash MEASURED via stm32tflm). |
| `enas_vww_summary.csv` / `enas_cancer_summary.csv` | ENAS v2.1 (no warm seeding) — backs Tables 2, 5, 6. |
| `enas_vww_focused_warmseed_summary.csv` | ENAS v2.1 WITH warm seeding at 50×50/64×64 — backs Table 4. |
| `enas_measured_resources.csv` | stm32tflm MEASURED RAM/Flash of the ENAS-selected `.tflite` models — backs Table 3. |

Notes:
- ENAS RAM/Flash columns inside the `enas_*_summary.csv` files are the *analytical*
  pre-flight estimates and are NOT used for any reported footprint; Table 3 uses the
  MEASURED values in `enas_measured_resources.csv`.
- The broad sweep is without warm seeding; the focused 50/64 study uses warm seeding
  (this is disclosed in the paper, Section 4.3).
