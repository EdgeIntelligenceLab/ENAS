#!/usr/bin/env python
"""
run_ablation.py
================
Run the ablation experiments from Paper Table 7.

Reproduces the following rows:
  - ENAS-strategy (2D search space, parallel random)
  - ENAS Ep=1, wacc=0.65 (baseline before calibration)
  - + Ep=3
  - + wacc=0.80
  - + two-phase Stage 2
  - + warm seed (full ENAS — same as default config)

Each ablation runs the full sweep on a subset of hardware and 50×50/64×64
input sizes (the two most commonly-targeted TinyML resolutions).

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --config configs/experiments/ablation.yaml
    python scripts/run_ablation.py --variants 'Ep=3' 'wacc=0.80'
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

# ── Ablation variants (Paper Table 7) ─────────────────────────────────────────
VARIANTS = {
    "Ep=1_wacc=0.65": dict(
        proxy_epochs        = 1,
        proxy_data_fraction = 0.20,
        accuracy_weight     = 0.65,
        efficiency_weight   = 0.25,
        headroom_weight     = 0.10,
        two_phase_stage2    = False,
        warm_seed           = False,
    ),
    "Ep=3": dict(
        proxy_epochs        = 3,
        proxy_data_fraction = 0.20,
        accuracy_weight     = 0.65,
        efficiency_weight   = 0.25,
        headroom_weight     = 0.10,
        two_phase_stage2    = False,
        warm_seed           = False,
    ),
    "wacc=0.80": dict(
        proxy_epochs        = 3,
        proxy_data_fraction = 0.20,
        accuracy_weight     = 0.80,
        efficiency_weight   = 0.15,
        headroom_weight     = 0.05,
        two_phase_stage2    = False,
        warm_seed           = False,
    ),
    "two_phase_stage2": dict(
        proxy_epochs        = 3,
        proxy_data_fraction = 0.20,
        accuracy_weight     = 0.80,
        efficiency_weight   = 0.15,
        headroom_weight     = 0.05,
        two_phase_stage2    = True,
        warm_seed           = False,
    ),
    "full_ENAS": dict(
        proxy_epochs        = 3,
        proxy_data_fraction = 0.20,
        accuracy_weight     = 0.80,
        efficiency_weight   = 0.15,
        headroom_weight     = 0.05,
        two_phase_stage2    = True,
        warm_seed           = True,
    ),
}

# ── Hardware subset for ablation (per-paper protocol) ─────────────────────────
ABLATION_HARDWARE = [
    "STM32L010RBT6", "ArduinoNano33IoT", "NUCLEO-L412KB",
    "RaspberryPiPico", "ArduinoNano33BLE",
    "ArduinoNiclaVision", "STM32H743ZI",
]
ABLATION_SIZES = [50, 64]
ABLATION_RUNS  = 3

HARDWARE_DIR = REPO_ROOT / "configs" / "hardware"
DATASET_DIR  = REPO_ROOT / "configs" / "datasets"


def is_feasible(input_shape, ram_constraint):
    W, H, C = input_shape
    return (W * H * 4 + W * H * C) <= ram_constraint


def load_hardware(name):
    fp = HARDWARE_DIR / f"{name.lower().replace('-', '_')}.yaml"
    if not fp.exists():
        for f in HARDWARE_DIR.glob("*.yaml"):
            with open(f) as fh:
                cfg = yaml.safe_load(fh)
            if cfg.get("name") == name:
                return cfg
    with open(fp) as f:
        return yaml.safe_load(f)


def run_variant(variant_name, params, hw_cfg, ds_cfg, size, run, output_root):
    hw_name    = hw_cfg["name"]
    result_dir = (output_root / variant_name /
                  f"{hw_name}_size{size}_run{run}")
    log_file   = result_dir / f"log_input{size}.txt"

    if log_file.exists():
        with open(log_file) as f:
            if "Experiment finished" in f.read():
                print(f"  [SKIP] {variant_name} | {hw_name} | {size}×{size} | Run {run}")
                return

    input_shape = (size, size, 3)
    if not is_feasible(input_shape, hw_cfg["ram_bytes"]):
        return

    os.makedirs(result_dir, exist_ok=True)
    print(f"\n=== Ablation: {variant_name} | {hw_name} | {size}×{size} | Run {run} ===")

    log_fh = open(log_file, "w")
    saved_stdout = sys.stdout
    sys.stdout = log_fh

    try:
        start = datetime.now()
        print(f"Variant: {variant_name}")
        print(f"Hyperparameters: {params}")
        print(f"Start time: {start}")
        print(f"Hardware: {hw_name}")
        print(f"Input shape: {input_shape}\n")

        nas = ENAS(
            max_ram              = hw_cfg["ram_bytes"],
            max_flash            = hw_cfg["flash_bytes"],
            max_macc             = hw_cfg["macc_limit"],
            path_to_training_set = ds_cfg["paths"]["train"],
            val_split            = ds_cfg["preprocessing"]["val_split"],
            input_shape          = input_shape,
            save_path            = str(result_dir),
            proxy_epochs         = params["proxy_epochs"],
            proxy_data_fraction  = params["proxy_data_fraction"],
            accuracy_weight      = params["accuracy_weight"],
            efficiency_weight    = params["efficiency_weight"],
            headroom_weight      = params["headroom_weight"],
        )

        nas.search()
        nas.train(100, 0.01, 128)
        nas.apply_uint8_post_training_quantization()
        nas.test_keras_model(ds_cfg["paths"]["test"])
        nas.test_tflite_model(ds_cfg["paths"]["test"])

        end = datetime.now()
        print(f"\nExperiment finished at: {end}")
        print(f"Total runtime: {end - start}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
    finally:
        sys.stdout = saved_stdout
        log_fh.close()


def main():
    p = argparse.ArgumentParser(description="Run ablation experiments.")
    p.add_argument("--dataset", default="vww", choices=["vww", "melanoma"])
    p.add_argument("--variants", nargs="+", default=None,
                   help="Subset of variant names to run (default: all)")
    p.add_argument("--config", default=None,
                   help="Optional YAML config (overrides defaults)")
    args = p.parse_args()

    ds_path = DATASET_DIR / f"{args.dataset}.yaml"
    with open(ds_path) as f:
        ds_cfg = yaml.safe_load(f)

    variants = args.variants if args.variants else list(VARIANTS.keys())
    for v in variants:
        if v not in VARIANTS:
            print(f"[ERROR] Unknown variant: {v}")
            print(f"Available: {list(VARIANTS.keys())}")
            sys.exit(1)

    output_root = REPO_ROOT / "results" / "raw_logs" / f"ablation_{args.dataset}"
    os.makedirs(output_root, exist_ok=True)

    print(f"\nAblation study: dataset={args.dataset}")
    print(f"Variants: {variants}")
    print(f"Hardware: {ABLATION_HARDWARE}")
    print(f"Sizes:    {ABLATION_SIZES}")
    print(f"Runs:     {ABLATION_RUNS}\n")

    for variant_name in variants:
        params = VARIANTS[variant_name]
        for hw_name in ABLATION_HARDWARE:
            hw_cfg = load_hardware(hw_name)
            for size in ABLATION_SIZES:
                for run in range(1, ABLATION_RUNS + 1):
                    run_variant(
                        variant_name, params, hw_cfg, ds_cfg, size, run, output_root
                    )

    print(f"\n✔ Ablation complete. Results in: {output_root}")


if __name__ == "__main__":
    main()
