#!/usr/bin/env python
"""
plot_pipeline.py — Figure 1 of the ENAS paper.

Renders the four-phase ENAS pipeline diagram and the cell-based architecture
template, matching the paper's TikZ figure as closely as matplotlib allows.

Usage:
    python figures/scripts/plot_pipeline.py
    python figures/scripts/plot_pipeline.py --output figures/output/pipeline_diagram.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Paper colours ─────────────────────────────────────────────────────────────
COL_PRIMARY = "#1F6F75"        # teal — ENAS brand
COL_ACCENT  = "#E7913A"        # orange — for stages
COL_BG      = "#F8F8F4"        # cream — phase boxes
COL_TEXT    = "#2A2A2A"
COL_GRAY    = "#999999"


def draw_phase_box(ax, x, y, w, h, title, body, colour=COL_BG, edge=COL_PRIMARY):
    """Draw a single phase box with a title and body text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4, edgecolor=edge, facecolor=colour
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.06, title,
            ha="center", va="top", fontsize=10, fontweight="bold",
            color=edge)
    ax.text(x + w/2, y + h/2 - 0.08, body,
            ha="center", va="center", fontsize=8.5,
            color=COL_TEXT)


def draw_arrow(ax, x1, y1, x2, y2):
    """Connector arrow between phases."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=14,
        color=COL_PRIMARY, linewidth=1.5
    )
    ax.add_patch(arrow)


def draw_pipeline(ax):
    """Left panel: 4-phase ENAS pipeline."""
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("ENAS Pipeline", fontsize=11, fontweight="bold", pad=10)

    # Four phase boxes, vertical
    phases = [
        ("Phase 0\nFeasibility",      "Eq. 3 RAM lower-bound\ncheck"),
        ("Phase 1\n3-stage Search",   "70 candidates total:\n30 random + 40 mut."),
        ("Phase 2\nFull Training",    "100 ep, cosine LR\nbatch 128, ES(15)"),
        ("Phase 3\nINT8 PTQ + Eval",  "150 calibration samples\nTFLite test accuracy"),
    ]
    box_w, box_h, gap = 4.4, 0.7, 0.16
    y_top = 3.6
    for i, (title, body) in enumerate(phases):
        y = y_top - i * (box_h + gap)
        draw_phase_box(ax, 0.3, y, box_w, box_h, title, body)
        if i < len(phases) - 1:
            draw_arrow(ax, 0.3 + box_w/2, y,
                       0.3 + box_w/2, y - gap)

    # Phase 1 detail: 3 stages
    ax.text(2.5, 0.05,
            "Stage 1: Random 30  →  Stage 2: Top-K=8  →  Stage 3: Mutate 40",
            ha="center", va="bottom", fontsize=8, style="italic",
            color=COL_ACCENT)


def draw_architecture(ax):
    """Right panel: cell-based architecture template."""
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Cell-Based Architecture Template",
                 fontsize=11, fontweight="bold", pad=10)

    blocks = [
        ("Input", "(H × W × C)"),
        ("Stem", "Conv 3×3, k filters"),
        ("Cell 1", "(b, k, s, g, a, e)"),
        ("Cell 2", "(b, k, s, g, a, e)"),
        ("…",      "up to 5 cells"),
        ("Cell N", "(b, k, s, g, a, e)"),
        ("Head",   "GAP → Dropout → Dense"),
        ("Output", "softmax (2 classes)"),
    ]
    box_w, box_h, gap = 4.0, 0.36, 0.05
    y_top = 3.6
    for i, (name, body) in enumerate(blocks):
        y = y_top - i * (box_h + gap)
        col = COL_ACCENT if "Cell" in name else (
              COL_PRIMARY if name in ("Stem", "Head") else COL_GRAY)
        box = FancyBboxPatch(
            (0.5, y), box_w, box_h,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=1.0, edgecolor=col,
            facecolor="white" if name not in ("Stem", "Head") else COL_BG)
        ax.add_patch(box)
        ax.text(0.6, y + box_h/2, name, ha="left", va="center",
                fontsize=9, fontweight="bold", color=col)
        ax.text(4.4, y + box_h/2, body, ha="right", va="center",
                fontsize=8, color=COL_TEXT)
        if i < len(blocks) - 1:
            draw_arrow(ax, 2.5, y, 2.5, y - gap)

    # Cell legend at bottom
    ax.text(2.5, -0.1,
            "b: block type   k: kernel   s: stride   g: skip   a: activation   e: expansion",
            ha="center", va="top", fontsize=7.5, style="italic",
            color=COL_TEXT)


def main():
    parser = argparse.ArgumentParser(description="Render Figure 1.")
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "figures" / "output" /
                                    "pipeline_diagram.pdf"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw_pipeline(axes[0])
    draw_architecture(axes[1])
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  ✔ Saved {out}")
    print(f"  ✔ Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
