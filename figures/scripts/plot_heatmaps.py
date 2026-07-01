#!/usr/bin/env python
"""
plot_heatmaps.py — Figure 3 of the ENAS paper.

Generates a 4-panel heatmap of (ENAS − NanoNAS) deltas:
    (a) VWW    : ΔTFLite accuracy (pp)
    (b) Cancer : ΔTFLite accuracy (pp)
    (c) VWW    : Δsearch time (min)
    (d) Cancer : Δsearch time (min)

Colour scheme:
    - Green = ENAS better/faster
    - Red   = NanoNAS better/faster
    - Grey hatched = RAM-infeasible

Rows: hardware ordered by SRAM (low → high)
Cols: input resolutions {32, 48, 50, 64, 72, 80, 96, 112, 128}

Usage:
    python figures/scripts/plot_heatmaps.py
    python figures/scripts/plot_heatmaps.py --output figures/output/heatmap_4panel.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "figures" / "scripts"))

from style import (
    apply_paper_style, DIVERGING_CMAP,
    HARDWARE_ORDER, HARDWARE_LABELS, SIZES, COLOURS,
)

DEFAULT_INPUT  = REPO_ROOT / "results" / "parsed_csv"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "output" / "heatmap_4panel.pdf"


def parse_time_to_minutes(t):
    """Convert 'H:MM:SS.uuuuuu' to minutes."""
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


def build_delta_matrix(enas_df, nanonas_df, value_col, time_conversion=False):
    """
    Build a (n_hardware × n_sizes) matrix of (ENAS − NanoNAS) deltas.
    Cells with no data (infeasible) are returned as NaN.
    """
    n_hw    = len(HARDWARE_ORDER)
    n_sizes = len(SIZES)
    matrix  = np.full((n_hw, n_sizes), np.nan)

    # Aggregate by (hardware, input_size) — mean of completed runs only
    def agg(df, col):
        d = df[df["status"] == "COMPLETED"].copy()
        if time_conversion:
            d[col] = d[col].apply(parse_time_to_minutes)
        else:
            # Convert accuracy fraction to percentage if needed
            if d[col].max() <= 1.0:
                d[col] = d[col] * 100
        return d.groupby(["hardware", "input_size"])[col].mean()

    enas_agg    = agg(enas_df,    value_col)
    nanonas_agg = agg(nanonas_df, value_col)

    for i, hw in enumerate(HARDWARE_ORDER):
        for j, size in enumerate(SIZES):
            try:
                e = enas_agg.loc[(hw, size)]
                n = nanonas_agg.loc[(hw, size)]
                if not pd.isna(e) and not pd.isna(n):
                    matrix[i, j] = e - n
            except (KeyError, IndexError):
                pass

    return matrix


def draw_heatmap(ax, matrix, title, cbar_label, vmin, vmax,
                 fmt="{:+.1f}"):
    """Draw a single heatmap panel."""
    # Mask NaN cells (infeasible) and draw with hatch pattern
    mask = np.isnan(matrix)

    # Plot the heatmap
    im = ax.imshow(
        matrix, cmap=DIVERGING_CMAP, vmin=vmin, vmax=vmax,
        aspect="auto", origin="upper",
    )

    # Overlay grey hatched pattern on infeasible cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if mask[i, j]:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=COLOURS["infeas"],
                    hatch='///', edgecolor='white', linewidth=0.5,
                    alpha=0.6, zorder=2,
                ))
            else:
                val = matrix[i, j]
                txt_color = "white" if abs(val) > (vmax * 0.55) else "black"
                ax.text(j, i, fmt.format(val),
                        ha='center', va='center',
                        fontsize=7, color=txt_color, zorder=3)

    # Axes labels & ticks
    ax.set_xticks(np.arange(len(SIZES)))
    ax.set_xticklabels(SIZES, fontsize=8)
    ax.set_yticks(np.arange(len(HARDWARE_ORDER)))
    ax.set_yticklabels([HARDWARE_LABELS[h] for h in HARDWARE_ORDER],
                       fontsize=8)
    ax.set_xlabel("Input resolution (px)", fontsize=9)
    ax.set_ylabel("Hardware (SRAM ↑)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax.tick_params(axis='both', length=0)
    ax.grid(False)

    # Cell border
    for i in range(matrix.shape[0] + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.4, zorder=4)
    for j in range(matrix.shape[1] + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.4, zorder=4)

    # Colour bar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    return im


def main():
    p = argparse.ArgumentParser(description="Generate Figure 3 (4-panel heatmap).")
    p.add_argument("--input",  default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    apply_paper_style()

    in_dir = Path(args.input)

    # Load all four CSVs
    files = {
        "enas_vww":      in_dir / "enas_vww_summary.csv",
        "nanonas_vww":   in_dir / "nanonas_vww_summary.csv",
        "enas_cancer":   in_dir / "enas_melanoma_summary.csv",
        "nanonas_cancer":in_dir / "nanonas_melanoma_summary.csv",
    }
    dfs = {}
    for k, fp in files.items():
        if not fp.exists():
            print(f"[WARN] {fp} not found — creating placeholder.")
            dfs[k] = pd.DataFrame(columns=[
                "hardware", "input_size", "status",
                "tflite_test_acc", "search_time",
            ])
        else:
            dfs[k] = pd.read_csv(fp)

    # Build the four delta matrices
    delta_vww_acc = build_delta_matrix(
        dfs["enas_vww"], dfs["nanonas_vww"], "tflite_test_acc")
    delta_can_acc = build_delta_matrix(
        dfs["enas_cancer"], dfs["nanonas_cancer"], "tflite_test_acc")
    delta_vww_t = build_delta_matrix(
        dfs["enas_vww"], dfs["nanonas_vww"], "search_time",
        time_conversion=True)
    delta_can_t = build_delta_matrix(
        dfs["enas_cancer"], dfs["nanonas_cancer"], "search_time",
        time_conversion=True)

    # For search time, we want NanoNAS − ENAS so positive = ENAS faster (paper convention)
    # delta as ENAS − NanoNAS means negative numbers = ENAS faster (smaller time)
    # Paper's panels (c) and (d) show NanoNAS_time - ENAS_time so positive = ENAS faster
    delta_vww_t = -delta_vww_t
    delta_can_t = -delta_can_t

    # ── Figure layout: 2 rows × 2 cols ─────────────────────────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9),
        gridspec_kw={"hspace": 0.35, "wspace": 0.30}
    )

    # Symmetric ranges centred on zero
    acc_max = max(8, np.nanpercentile(np.abs(delta_vww_acc), 95))
    can_max = max(4, np.nanpercentile(np.abs(delta_can_acc), 95))
    vt_max  = max(100, np.nanpercentile(np.abs(delta_vww_t), 95))
    ct_max  = max(10, np.nanpercentile(np.abs(delta_can_t), 95))

    draw_heatmap(axes[0, 0], delta_vww_acc,
                 r"(a) VWW: $\Delta$ TFLite accuracy (pp)",
                 r"$\Delta$ accuracy (pp)", -acc_max, acc_max,
                 fmt="{:+.1f}")
    draw_heatmap(axes[0, 1], delta_can_acc,
                 r"(b) Melanoma: $\Delta$ TFLite accuracy (pp)",
                 r"$\Delta$ accuracy (pp)", -can_max, can_max,
                 fmt="{:+.1f}")
    draw_heatmap(axes[1, 0], delta_vww_t,
                 r"(c) VWW: $\Delta$ search time saved (min)",
                 r"$\Delta$ time (min)", -vt_max, vt_max,
                 fmt="{:+.0f}")
    draw_heatmap(axes[1, 1], delta_can_t,
                 r"(d) Melanoma: $\Delta$ search time saved (min)",
                 r"$\Delta$ time (min)", -ct_max, ct_max,
                 fmt="{:+.1f}")

    # Single legend at the bottom
    legend_elements = [
        mpatches.Patch(facecolor=COLOURS["good"],
                       edgecolor='black', label='Green: ENAS better / faster'),
        mpatches.Patch(facecolor=COLOURS["bad"],
                       edgecolor='black', label='Red: NanoNAS better / faster'),
        mpatches.Patch(facecolor=COLOURS["infeas"], hatch='///',
                       edgecolor='black', label='Grey hatched: infeasible'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center', ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False, fontsize=9,
    )

    fig.suptitle("Per-Cell Comparison: ENAS vs. NanoNAS",
                 fontsize=12, fontweight='bold', y=0.99)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✔ Figure 3 written to: {out_path}")

    # Also save individual panels
    panels = {
        "heatmap_vww_accuracy.pdf":    (delta_vww_acc, "(a) VWW: ΔTFLite accuracy (pp)", -acc_max, acc_max, "{:+.1f}"),
        "heatmap_melanoma_accuracy.pdf": (delta_can_acc, "(b) Melanoma: ΔTFLite accuracy (pp)", -can_max, can_max, "{:+.1f}"),
        "heatmap_vww_search_time.pdf": (delta_vww_t, "(c) VWW: Δsearch time saved (min)", -vt_max, vt_max, "{:+.0f}"),
        "heatmap_melanoma_search_time.pdf": (delta_can_t, "(d) Melanoma: Δsearch time saved (min)", -ct_max, ct_max, "{:+.1f}"),
    }
    for fname, (mat, title, vmin, vmax, fmt) in panels.items():
        fig_single, ax_single = plt.subplots(figsize=(8, 4.5))
        draw_heatmap(ax_single, mat, title, "Δ", vmin, vmax, fmt)
        fig_single.savefig(out_path.parent / fname, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"  ↳ {fname}")

    print("\n✔ All heatmaps generated successfully.")


if __name__ == "__main__":
    main()
