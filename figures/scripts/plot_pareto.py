#!/usr/bin/env python3
"""Regenerate the accuracy vs. search-time Pareto scatter (paper Fig. 3) from the
parsed result CSVs, with the legend inside the top-right corner."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- style to match the paper ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "cm",
})
C_NANO = "#2f4b52"   # dark teal  (NanoNAS)
C_ENAS = "#4bb3a4"   # teal       (ENAS)
C_GREY = "#9aa0a3"   # neutral for the VWW/Cancer style keys

def parse_time(s):
    s = str(s).strip()
    if s in ("", "nan"): return np.nan
    if ":" in s:
        h, m, rest = s.split(":"); return int(h)*60 + int(m) + float(rest)/60
    try: return float(s)
    except ValueError: return np.nan

def load(path):
    d = pd.read_csv(path)
    d = d[d["Status"].astype(str).str.upper() == "COMPLETED"].copy()
    d["acc"]   = pd.to_numeric(d["TFLite Test Acc"], errors="coerce") * 100
    d["stime"] = d["Search Time"].map(parse_time)
    return d.dropna(subset=["acc", "stime"])

def scatter(ax, d, color, marker, outlined):
    ax.scatter(d["stime"], d["acc"], s=34, marker=marker,
               facecolors=color,
               edgecolors=("#20343a" if outlined else "none"),
               linewidths=(0.7 if outlined else 0.0),
               alpha=(0.9 if outlined else 0.55), zorder=3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="results/parsed_csv")
    ap.add_argument("--out", default="fig4_pareto.pdf")
    a = ap.parse_args()
    d = Path(a.csv_dir)
    Nv, Ev = load(d/"nanonas_vww_summary.csv"),    load(d/"enas_vww_summary.csv")
    Nc, Ec = load(d/"nanonas_cancer_summary.csv"), load(d/"enas_cancer_summary.csv")

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.set_xscale("log")
    # VWW = solid (no edge); Cancer = outlined
    scatter(ax, Nv, C_NANO, "o", outlined=False)
    scatter(ax, Ev, C_ENAS, "s", outlined=False)
    scatter(ax, Nc, C_NANO, "o", outlined=True)
    scatter(ax, Ec, C_ENAS, "s", outlined=True)

    ax.set_xlabel("Search time per configuration (min, log)")
    ax.set_ylabel("TFLite INT8 accuracy (\\%)")
    ax.grid(True, which="both", ls=":", lw=0.6, color="0.8", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(1.8, 400)
    ax.set_ylim(64, 92)

    handles = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor=C_NANO,
               markeredgecolor="none", markersize=8, label="NanoNAS"),
        Line2D([0],[0], marker="s", color="none", markerfacecolor=C_ENAS,
               markeredgecolor="none", markersize=8, label="ENAS"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=C_GREY,
               markeredgecolor="none", markersize=8, label="VWW (solid)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=C_GREY,
               markeredgecolor="#20343a", markeredgewidth=0.9, markersize=8,
               label="Cancer (outlined)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="0.7", fancybox=False, borderpad=0.5, handletextpad=0.4,
              labelspacing=0.3, fontsize=9)

    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
