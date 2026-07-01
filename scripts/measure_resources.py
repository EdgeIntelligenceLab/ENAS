#!/usr/bin/env python3
"""
measure_enas_resources.py
=========================
Batch-measure Flash and peak RAM of the ENAS-selected models using the same
stm32tflm tool NanoNAS already uses, so the resource comparison is
measured-vs-measured (apples-to-apples) for reviewer weakness f.

Run this from your NanoNAS project root, i.e. the folder that contains
./stm32tflm and the all_results_* directories:

    cd ~/mtech_project/NAS/NanoNAS
    python measure_enas_resources.py

It does NOT retrain or re-quantize anything. It only reads the existing
resulting_architecture.tflite files, so it finishes in a couple of minutes.
Output: enas_v2_measured_resources.csv
"""

import os
import re
import csv
import sys
import subprocess
from pathlib import Path

# ─────────────────────────── CONFIG ────────────────────────────────────────
STM32TFLM = "./stm32tflm"          # path to the stm32tflm binary

# (label, directory). Add the warm-seeded focused dirs here too if you want a
# measured table that matches the paper's focused 50/64 study (Table 3).
RESULT_DIRS = [
    ("VWW",    "all_results_enas_v2_all_inputs"),
    ("Cancer", "all_results_enas_v2_cancer_all_inputs"),
    # ("VWW_seed50_64", "all_results_enas_v2_<your_warmseed_dir>"),
]

OUT_CSV  = "enas_v2_measured_resources.csv"
TIMEOUT  = 30                       # seconds per model
# ────────────────────────────────────────────────────────────────────────────

DIR_RE = re.compile(r'^(?P<hw>.+)_size(?P<size>\d+)_run(?P<run>\d+)$')


def measure(tflite_path):
    """Run stm32tflm on one .tflite. Returns (flash, ram, status)."""
    try:
        proc = subprocess.Popen(
            [STM32TFLM, str(tflite_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        return None, None, f"stm32tflm not found at '{STM32TFLM}'"

    try:
        outs, _ = proc.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None, None, "timeout"

    text = outs.decode(errors="ignore")
    nums = re.findall(r'\d+', text)
    if len(nums) < 2:
        return None, None, f"unparsed:{text.strip()[:80]!r}"

    # NanoNAS.py parses as: flash, ram = re.findall(...) -> flash first.
    flash, ram = int(nums[0]), int(nums[1])
    return flash, ram, "ok"


def main():
    rows = []
    measured = 0
    for dataset, base_name in RESULT_DIRS:
        base = Path(base_name)
        if not base.is_dir():
            print(f"[WARN] directory not found, skipping: {base}", file=sys.stderr)
            continue

        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            m = DIR_RE.match(d.name)
            if not m:
                continue

            tfl = d / "resulting_architecture.tflite"
            if not tfl.exists():
                # infeasible cell or missing model -> no tflite produced
                rows.append([dataset, m["hw"], int(m["size"]), int(m["run"]),
                             "", "", "no_tflite"])
                print(f"  [skip ] {dataset:7s} {d.name}: no .tflite")
                continue

            flash, ram, status = measure(tfl)
            rows.append([dataset, m["hw"], int(m["size"]), int(m["run"]),
                         flash if flash is not None else "",
                         ram if ram is not None else "", status])
            if status == "ok":
                measured += 1
            print(f"  [{status:5s}] {dataset:7s} {m['hw']:18s} "
                  f"{m['size']:>3}px run{m['run']}: "
                  f"flash={flash} ram={ram}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Hardware", "Input Size", "Run",
                    "Measured Flash (bytes)", "Measured RAM (bytes)", "Status"])
        w.writerows(rows)

    print(f"\nDone. Measured {measured} models. Wrote {len(rows)} rows -> {OUT_CSV}")
    print("Sanity check: for the first few rows, Flash should be larger than RAM. "
          "If they look swapped, flip the (flash, ram) unpack order in measure().")


if __name__ == "__main__":
    main()
