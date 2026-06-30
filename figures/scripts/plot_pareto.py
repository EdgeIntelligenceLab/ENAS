#!/usr/bin/env python
"""
plot_pareto.py — Figure 2 of the ENAS paper.

Scatter plot of TFLite INT8 accuracy versus search time per configuration.
Shows all 212 cells (53 feasible × 2 methods × 2 datasets) on the
accuracy–search-time plane with log-scale X axis.

ENAS clusters at lower search times; NanoNAS spans a wider accuracy range
extending to higher search time. Cross-dataset stratification is visible:
Melanoma at the top (~89%), VWW at the bottom (~73%).

Usage:
    python figures/scripts/plot_pareto.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "figures" / "scripts"))

from style import apply_paper_style, COLOURS

DEFAULT_INPUT  = REPO_ROOT / "results" / "parsed_csv"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "output" / "pareto_accuracy_vs_searchtime.pdf"


def parse_time_to_minutes(t):
    if pd.isna(t) or not isinstance(t, str):
        return np.nan
    try:
        parts = t.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 60
        elif len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60
    except Exception:
        return np.nan
    return np.nan


def main():
    p = argparse.ArgumentParser(description="Generate Figure 2 (Pareto plot).")
    p.add_argument("--input",  default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    apply_paper_style()
    in_dir = Path(args.input)

    files = {
        ("enas",    "vww"):     "enas_vww_summary.csv",
        ("nanonas", "vww"):     "nanonas_vww_summary.csv",
        ("enas",    "melanoma"):"enas_melanoma_summary.csv",
        ("nanonas", "melanoma"):"nanonas_melanoma_summary.csv",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    style_map = {
        ("nanonas", "vww"):      dict(marker='o', s=42, c=COLOURS["nanonas"],
                                       label="NanoNAS — VWW",
                                       edgecolor='none'),
        ("enas",    "vww"):      dict(marker='s', s=42, c=COLOURS["enas"],
                                       label="ENAS — VWW",
                                       edgecolor='none'),
        ("nanonas", "melanoma"): dict(marker='o', s=42,
                                       c='none', edgecolor=COLOURS["nanonas"],
                                       linewidths=1.2,
                                       label="NanoNAS — Melanoma"),
        ("enas",    "melanoma"): dict(marker='s', s=42,
                                       c='none', edgecolor=COLOURS["enas"],
                                       linewidths=1.2,
                                       label="ENAS — Melanoma"),
    }

    total_points = 0
    for (method, dataset), fname in files.items():
        fp = in_dir / fname
        if not fp.exists():
            print(f"[WARN] {fp} not found; skipping.")
            continue
        df = pd.read_csv(fp)
        df = df[df["status"] == "COMPLETED"].copy()
        if df.empty:
            continue
        df["search_min"] = df["search_time"].apply(parse_time_to_minutes)
        df["acc_pct"]    = df["tflite_test_acc"] * 100

        valid = df.dropna(subset=["search_min", "acc_pct"])
        ax.scatter(
            valid["search_min"], valid["acc_pct"],
            **style_map[(method, dataset)],
            alpha=0.7, zorder=3,
        )
        total_points += len(valid)

    ax.set_xscale("log")
    ax.set_xlabel("Search time per configuration (min, log scale)", fontsize=10)
    ax.set_ylabel("TFLite INT8 test accuracy (%)", fontsize=10)
    ax.set_title(f"Accuracy vs. Search Time — All {total_points} Cells",
                 fontsize=11, fontweight='bold')
    ax.set_ylim(60, 95)
    ax.grid(True, which='both', linestyle='-', linewidth=0.5,
            color='#E8E8E8', zorder=1)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    # Annotate dataset clusters
    ax.annotate("Melanoma (~89%)",  xy=(20, 88), xytext=(50, 91),
                fontsize=8, color='gray', alpha=0.7,
                ha='center')
    ax.annotate("VWW (~73%)", xy=(20, 73), xytext=(50, 67),
                fontsize=8, color='gray', alpha=0.7,
                ha='center')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✔ Figure 2 written to: {out_path}")


if __name__ == "__main__":
    main()
