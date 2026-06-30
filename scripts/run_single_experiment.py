#!/usr/bin/env python
"""
run_single_experiment.py
========================
Run ENAS v2.1 (or NanoNAS) on a specified hardware and input size.

Usage:
  python scripts/run_single_experiment.py --hardware ArduinoNiclaVision --size 64
  python scripts/run_single_experiment.py --hardware STM32H743ZI --size 64 --method nanonas
  python scripts/run_single_experiment.py --list-hardware
"""

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml

# Ensure repository root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from enas import ENAS

VALID_SIZES = [32, 48, 50, 64, 72, 80, 96, 112, 128]
HARDWARE_DIR = REPO_ROOT / "configs" / "hardware"
DATASET_DIR  = REPO_ROOT / "configs" / "datasets"


def load_hardware_config(name):
    path = HARDWARE_DIR / f"{name.lower().replace('-', '_')}.yaml"
    if not path.exists():
        path = HARDWARE_DIR / f"{name}.yaml"
    if not path.exists():
        for f in HARDWARE_DIR.glob("*.yaml"):
            with open(f) as fh:
                cfg = yaml.safe_load(fh)
            if cfg.get("name") == name:
                return cfg
        raise FileNotFoundError(f"Hardware config not found: {name}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset_config(name):
    path = DATASET_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def list_hardware():
    print("\nAvailable hardware platforms:")
    print("─" * 80)
    print(f"  {'Name':<25} {'RAM':>10} {'Flash':>10} {'MACC':>12}")
    print("─" * 80)
    for f in sorted(HARDWARE_DIR.glob("*.yaml")):
        if f.name == "all_hardware.yaml":
            continue
        with open(f) as fh:
            cfg = yaml.safe_load(fh)
        ram   = f"{cfg['ram_bytes']//1024} KB"
        flash = f"{cfg['flash_bytes']//1024} KB"
        macc  = f"{cfg['macc_limit']//1000} K"
        print(f"  {cfg['name']:<25} {ram:>10} {flash:>10} {macc:>12}")
    print("─" * 80)
    print(f"\nValid input sizes: {VALID_SIZES}\n")


def is_feasible(input_shape, ram_constraint):
    W, H, C = input_shape
    return (W * H * 4 + W * H * C) <= ram_constraint


def run(args):
    hw  = load_hardware_config(args.hardware)
    ds  = load_dataset_config(args.dataset)
    input_shape = (args.size, args.size, 3)

    # Output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(
            REPO_ROOT / "results" / "single_runs" /
            f"{args.method}_{hw['name']}_size{args.size}_"
            f"run{args.run}_{timestamp}"
        )
    os.makedirs(args.output, exist_ok=True)

    print("=" * 80)
    print(f"  ENAS Experiment — {args.method}")
    print("=" * 80)
    print(f"  Hardware       : {hw['name']}")
    print(f"  Tier           : {hw['tier']}")
    print(f"  RAM/Flash/MACC : {hw['ram_bytes']} / {hw['flash_bytes']} / {hw['macc_limit']}")
    print(f"  Dataset        : {args.dataset}")
    print(f"  Input size     : {args.size}x{args.size}x3")
    print(f"  Run ID         : {args.run}")
    print(f"  Method         : {args.method}")
    print(f"  Output dir     : {args.output}")
    print("=" * 80)

    if not is_feasible(input_shape, hw["ram_bytes"]):
        print(f"\n[INFEASIBLE] Minimum RAM exceeds device constraint.\n")
        return

    # Build NAS instance
    if args.method.lower() == "enas":
        nas = ENAS(
            max_ram              = hw["ram_bytes"],
            max_flash            = hw["flash_bytes"],
            max_macc             = hw["macc_limit"],
            path_to_training_set = ds["paths"]["train"],
            val_split            = ds["preprocessing"]["val_split"],
            input_shape          = input_shape,
            save_path            = args.output,
            proxy_epochs         = 3,
            proxy_data_fraction  = 0.20,
            accuracy_weight      = 0.80,
            efficiency_weight    = 0.15,
            headroom_weight      = 0.05,
        )
    else:
        from nanonas import NanoNAS
        nas = NanoNAS(
            max_ram              = hw["ram_bytes"],
            max_flash            = hw["flash_bytes"],
            max_macc             = hw["macc_limit"],
            path_to_training_set = ds["paths"]["train"],
            val_split            = ds["preprocessing"]["val_split"],
            input_shape          = input_shape,
            save_path            = args.output,
        )

    # ── Phase 1: search ───────────────────────────────────────────────────
    nas.search()

    # ── Phase 2: full training ────────────────────────────────────────────
    nas.train(
        training_epochs        = args.epochs,
        training_learning_rate = args.lr,
        training_batch_size    = args.batch_size,
    )

    # ── Phase 3: quantisation ─────────────────────────────────────────────
    nas.apply_uint8_post_training_quantization()

    # ── Phase 4: evaluation ───────────────────────────────────────────────
    nas.test_keras_model(ds["paths"]["test"])
    nas.test_tflite_model(ds["paths"]["test"])

    print(f"\nResults saved to: {args.output}")


def main():
    p = argparse.ArgumentParser(
        description="Run a single ENAS or NanoNAS experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hardware",   type=str, default=None)
    p.add_argument("--size",       type=int, default=None)
    p.add_argument("--dataset",    type=str, default="vww",
                   choices=["vww", "melanoma"])
    p.add_argument("--method",     type=str, default="enas",
                   choices=["enas", "nanonas"])
    p.add_argument("--run",        type=int, default=1)
    p.add_argument("--output",     type=str, default=None)
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--list-hardware", action="store_true")
    p.add_argument("--seed",       type=int, default=None,
                   help="Random seed (overrides default per-run seed)")
    args = p.parse_args()

    if args.list_hardware:
        list_hardware()
        return
    if args.hardware is None or args.size is None:
        p.error("--hardware and --size are required (or use --list-hardware)")

    try:
        run(args)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
