"""
Shared matplotlib styling for ENAS paper figures.

All figures use a publication-grade style with:
    - Sans-serif fonts at ~9pt
    - Clean grids and minimal chartjunk
    - Diverging green-red colourmap for delta plots
    - Consistent figure dimensions
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ── Colour palette (anonymous, no institute branding) ─────────────────────────
COLOURS = {
    "enas":     "#2E7D6F",     # teal
    "nanonas":  "#2C2C2C",     # near-black
    "enas_v1":  "#C5A24A",     # muted gold
    "enas_v2":  "#D87454",     # warm orange
    "accent":   "#7A4FA6",     # purple
    "good":     "#2E8B3D",     # green (ENAS better)
    "bad":      "#B53A3F",     # red  (NanoNAS better)
    "neutral":  "#F5F5F5",
    "infeas":   "#CCCCCC",
}

# Diverging green→white→red colourmap for delta plots
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "enas_diverging",
    [
        (0.00, COLOURS["bad"]),       # red at most negative
        (0.50, "#FFFFFF"),            # white at zero
        (1.00, COLOURS["good"]),      # green at most positive
    ],
    N=256,
)


def apply_paper_style():
    """Apply publication-grade matplotlib defaults."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.frameon":    False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#E8E8E8",
        "grid.linewidth":    0.5,
        "grid.linestyle":    "-",
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.format":    "pdf",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


# ── Hardware ordering used across all figures (low → high SRAM) ───────────────
HARDWARE_ORDER = [
    "NUCLEO-L010RB",       # 20 KB
    "STM32L010RBT6",       # 20 KB
    "ArduinoNano33IoT",    # 32 KB
    "NUCLEO-L412KB",       # 64 KB
    "RaspberryPiPico",     # 264 KB
    "ArduinoNano33BLE",    # 256 KB
    "ArduinoNiclaVision",  # 1 MB
    "STM32H743ZI",         # 1 MB
]

# Display labels for plots (shortened)
HARDWARE_LABELS = {
    "NUCLEO-L010RB":       "L010RB",
    "STM32L010RBT6":       "L010RBT6",
    "ArduinoNano33IoT":    "IoT",
    "NUCLEO-L412KB":       "L412KB",
    "RaspberryPiPico":     "Pico",
    "ArduinoNano33BLE":    "BLE",
    "ArduinoNiclaVision":  "Nicla",
    "STM32H743ZI":         "H743",
}

SIZES = [32, 48, 50, 64, 72, 80, 96, 112, 128]
