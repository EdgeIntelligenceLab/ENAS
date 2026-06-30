#!/usr/bin/env python
"""
run_all_experiments.py
======================
Full experiment matrix runner for paper reproduction.

Runs ENAS v2.1 (and optionally NanoNAS) across:
    - 8 hardware platforms
    - 9 input sizes  (32, 48, 50, 64, 72, 80, 96, 112, 128)
    - 3 runs each

Total: 216 experiments per method per dataset.

Usage:
  python scripts/run_all_experiments.py --dataset vww
  python scripts/run_all_experiments.py --dataset melanoma
  python scripts/run_all_experiments.py --method nanonas --dataset vww
  python scripts/run_all_experiments.py --hardware ArduinoNano33BLE
  python scripts/run_all_experiments.py --size 50
  python scripts/run_all_experiments.py --resume
"""

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from enas import ENAS

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_SIZES = [32, 48, 50, 64, 72, 80, 96, 112, 128]
DEFAULT_RUNS  = 3

HARDWARE_DIR = REPO_ROOT / "configs" / "hardware"
DATASET_DIR  = REPO_ROOT / "configs" / "datasets"
RESULTS_DIR  = REPO_ROOT / "results" / "raw_logs"


def load_all_hardware():
    """Load all hardware configs (excluding extended SBC/GPU platforms)."""
    hardware = {}
    main_hardware = {
        "STM32L010RBT6", "NUCLEO-L010RB", "ArduinoNano33IoT",
        "NUCLEO-L412KB",  "RaspberryPiPico", "ArduinoNano33BLE",
        "ArduinoNiclaVision", "STM32H743ZI"
    }
    for f in sorted(HARDWARE_DIR.glob("*.yaml")):
        if f.name == "all_hardware.yaml":
            continue
        with open(f) as fh:
            cfg = yaml.safe_load(fh)
        if cfg["name"] in main_hardware:
            hardware[cfg["name"]] = cfg
    return hardware


def is_feasible_input_size(input_shape, ram_constraint):
    W, H, C = input_shape
    return (W * H * 4 + W * H * C) <= ram_constraint


def already_done(log_path):
    """Check whether an experiment was completed in a previous run."""
    if not os.path.exists(log_path):
        return False
    with open(log_path) as f:
        return "Experiment finished at" in f.read()


def run_one(method, hardware_cfg, ds_cfg, size, run, output_root,
            epochs, lr, batch_size, seed):
    """Run one experiment (single hardware, size, run)."""
    hw_name      = hardware_cfg["name"]
    result_dir   = output_root / f"{hw_name}_size{size}_run{run}"
    log_file     = result_dir / f"log_input{size}.txt"

    if already_done(log_file):
        print(f"[SKIP] {hw_name} | {size}x{size} | Run {run}")
        return

    print(f"\n=== {hw_name} | {size}x{size} | Run {run} ===")

    input_shape = (size, size, 3)
    if not is_feasible_input_size(input_shape, hardware_cfg["ram_bytes"]):
        min_ram = size * size * 4 + size * size * 3
        os.makedirs(result_dir, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(f"Experiment start time: {datetime.now()}\n")
            f.write(f"Hardware: {hw_name}\nRun: {run}\n")
            f.write(f"Input shape: {input_shape}\n")
            f.write(f"[SKIPPED] Min possible RAM ({min_ram} bytes) "
                    f"exceeds constraint ({hardware_cfg['ram_bytes']} bytes).\n")
            f.write(f"No feasible solution found.\n")
            f.write(f"Experiment finished at: {datetime.now()}\n")
        return

    os.makedirs(result_dir, exist_ok=True)
    log_fh = open(log_file, "w")
    saved_stdout = sys.stdout
    sys.stdout = log_fh

    try:
        start = datetime.now()
        print(f"Experiment start time: {start}")
        print(f"Hardware: {hw_name}")
        print(f"Run: {run} | Seed: {seed}")
        print(f"Input shape: {input_shape}")
        print(f"RAM constraint: {hardware_cfg['ram_bytes']}")
        print(f"Flash constraint: {hardware_cfg['flash_bytes']}")
        print(f"MACC constraint: {hardware_cfg['macc_limit']}\n")

        if method == "enas":
            nas = ENAS(
                max_ram              = hardware_cfg["ram_bytes"],
                max_flash            = hardware_cfg["flash_bytes"],
                max_macc             = hardware_cfg["macc_limit"],
                path_to_training_set = ds_cfg["paths"]["train"],
                val_split            = ds_cfg["preprocessing"]["val_split"],
                input_shape          = input_shape,
                save_path            = str(result_dir),
                proxy_epochs         = 3,
                proxy_data_fraction  = 0.20,
                accuracy_weight      = 0.80,
                efficiency_weight    = 0.15,
                headroom_weight      = 0.05,
            )
        else:
            from nanonas import NanoNAS
            nas = NanoNAS(
                max_ram              = hardware_cfg["ram_bytes"],
                max_flash            = hardware_cfg["flash_bytes"],
                max_macc             = hardware_cfg["macc_limit"],
                path_to_training_set = ds_cfg["paths"]["train"],
                val_split            = ds_cfg["preprocessing"]["val_split"],
                input_shape          = input_shape,
                save_path            = str(result_dir),
            )

        nas.search()
        nas.train(epochs, lr, batch_size)
        nas.apply_uint8_post_training_quantization()
        nas.test_keras_model(ds_cfg["paths"]["test"])
        nas.test_tflite_model(ds_cfg["paths"]["test"])

        end = datetime.now()
        print(f"\nExperiment finished at: {end}")
        print(f"Total runtime: {end - start}")

    except Exception as e:
        print(f"\n[ERROR] Crashed: {e}")
        traceback.print_exc()
    finally:
        sys.stdout = saved_stdout
        log_fh.close()


def main():
    p = argparse.ArgumentParser(
        description="Run full experiment matrix for paper reproduction."
    )
    p.add_argument("--method",   default="enas",
                   choices=["enas", "nanonas"])
    p.add_argument("--dataset",  default="vww",
                   choices=["vww", "melanoma"])
    p.add_argument("--hardware", type=str, default=None,
                   help="Restrict to one hardware platform")
    p.add_argument("--size",     type=int, default=None,
                   help="Restrict to one input size")
    p.add_argument("--runs",     type=int, default=DEFAULT_RUNS)
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--lr",       type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--resume",   action="store_true",
                   help="Skip experiments already completed (default behaviour)")
    args = p.parse_args()

    hardware = load_all_hardware()
    ds_path  = DATASET_DIR / f"{args.dataset}.yaml"
    with open(ds_path) as f:
        ds_cfg = yaml.safe_load(f)

    sizes = [args.size] if args.size else DEFAULT_SIZES
    seeds = [11, 42, 137]   # documented per-run seeds

    output_root = RESULTS_DIR / f"{args.method}_{args.dataset}"
    os.makedirs(output_root, exist_ok=True)

    total    = 0
    for size in sizes:
        for hw_name, hw_cfg in hardware.items():
            if args.hardware and hw_name != args.hardware:
                continue
            for run in range(1, args.runs + 1):
                seed = seeds[(run - 1) % len(seeds)]
                run_one(args.method, hw_cfg, ds_cfg, size, run,
                        output_root, args.epochs, args.lr,
                        args.batch_size, seed)
                total += 1

    print(f"\n\n✔ Completed {total} experiment slots in {output_root}")


if __name__ == "__main__":
    main()
