#!/usr/bin/env python
"""
generate_summary_tables.py
==========================
Generate publication-quality summary tables from parsed CSV results.

Outputs:
    - results/summary_tables/table_main_results.csv  (and .tex)
    - results/summary_tables/table_ablation.csv      (and .tex)
    - results/summary_tables/aggregated_means.csv    (per method/size/hw)

Usage:
    python scripts/generate_summary_tables.py
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT      = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = REPO_ROOT / "results" / "parsed_csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "summary_tables"


HARDWARE_ORDER = [
    "STM32L010RBT6", "NUCLEO-L010RB", "ArduinoNano33IoT",
    "NUCLEO-L412KB",  "RaspberryPiPico", "ArduinoNano33BLE",
    "ArduinoNiclaVision", "STM32H743ZI"
]

SIZES = [32, 48, 50, 64, 72, 80, 96, 112, 128]


def parse_time_to_minutes(time_str):
    """Convert 'H:MM:SS.uuuuuu' or 'M:SS' to minutes."""
    if not isinstance(time_str, str) or not time_str:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 60 + float(m) + float(s) / 60
        elif len(parts) == 2:
            m, s = parts
            return float(m) + float(s) / 60
    except Exception:
        return None
    return None


def aggregate_method(df, method_label):
    """Aggregate runs by (hardware, input_size) computing mean ± std."""
    df = df[df["status"] == "COMPLETED"].copy()
    df["search_min"] = df["search_time"].apply(parse_time_to_minutes)
    df["train_min"]  = df["train_time"].apply(parse_time_to_minutes)

    grouped = df.groupby(["hardware", "input_size"]).agg(
        tflite_mean = ("tflite_test_acc", "mean"),
        tflite_std  = ("tflite_test_acc", "std"),
        keras_mean  = ("keras_test_acc",  "mean"),
        nas_mean    = ("nas_val_acc",     "mean"),
        search_mean = ("search_min",      "mean"),
        train_mean  = ("train_min",       "mean"),
        n_runs      = ("run",             "count"),
    ).reset_index()
    grouped["method"] = method_label
    return grouped


def build_main_results_table(csv_dir, output_dir):
    """Build Table 1: TFLite accuracy comparison across all methods."""
    method_files = {
        "NanoNAS":   csv_dir / "nanonas_vww_summary.csv",
        "ENAS v1":   csv_dir / "enas_v1_vww_summary.csv",
        "ENAS v2":   csv_dir / "enas_v2_vww_summary.csv",
        "ENAS v2.1": csv_dir / "enas_vww_summary.csv",
    }

    rows = []
    for method, fpath in method_files.items():
        if not fpath.exists():
            print(f"  [WARN] {fpath} not found, skipping {method}")
            continue
        df = pd.read_csv(fpath)
        agg = aggregate_method(df, method)
        rows.append(agg)

    if not rows:
        print("  [ERROR] No summary CSVs found.")
        return

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_dir / "aggregated_means.csv", index=False)

    # Pivot: rows=size, cols=hardware, values=tflite_mean per method
    for method in combined["method"].unique():
        m_df = combined[combined["method"] == method]
        pivot = m_df.pivot(index="input_size", columns="hardware",
                           values="tflite_mean")
        pivot = pivot.reindex(columns=[h for h in HARDWARE_ORDER
                                       if h in pivot.columns])
        path = output_dir / f"main_results_{method.lower().replace(' ', '_').replace('.', '_')}.csv"
        pivot.to_csv(path)
        print(f"  ✔ Written {path.name}")

    # LaTeX table (compact 4-method comparison at common sizes 50 and 64)
    latex_table = generate_main_results_latex(combined)
    with open(output_dir / "table_main_results.tex", "w") as f:
        f.write(latex_table)
    print(f"  ✔ Written table_main_results.tex")


def generate_main_results_latex(combined_df):
    """Generate LaTeX source for Table 1 (main results)."""
    lines = [
        r"% Auto-generated main results table",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{TFLite test accuracy (\%) — four methods compared "
        r"across all hardware platforms at 50$\times$50 input resolution.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(HARDWARE_ORDER) + r"r}",
        r"\toprule",
        "Method & " + " & ".join(h.replace("_","\\_") for h in HARDWARE_ORDER)
            + r" & Mean \\",
        r"\midrule",
    ]
    for method in ["NanoNAS", "ENAS v1", "ENAS v2", "ENAS v2.1"]:
        m_df = combined_df[(combined_df["method"] == method) &
                           (combined_df["input_size"] == 50)]
        row = [method]
        accs = []
        for hw in HARDWARE_ORDER:
            v = m_df[m_df["hardware"] == hw]["tflite_mean"]
            if len(v) and not pd.isna(v.iloc[0]):
                acc = v.iloc[0] * 100
                row.append(f"{acc:.1f}")
                accs.append(acc)
            else:
                row.append("--")
        mean = sum(accs) / len(accs) if accs else 0
        row.append(f"{mean:.1f}")
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Generate summary tables.")
    p.add_argument("--input",  type=str, default=str(DEFAULT_INPUT))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    csv_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Generating main results table ──")
    build_main_results_table(csv_dir, out_dir)

    print(f"\n✔ Tables written to: {out_dir}")


if __name__ == "__main__":
    main()
