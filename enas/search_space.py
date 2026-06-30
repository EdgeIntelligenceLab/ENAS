"""
Cell-based search space definition for ENAS v2.1.

Each architecture is described by:
    - k         : base filter count in stem (global)
    - num_cells : number of cells stacked (global)
    - cells     : list of per-cell parameter dicts
"""

import random
from copy import deepcopy

# ── Master search space ───────────────────────────────────────────────────────
SEARCH_SPACE = {
    # Global parameters
    "k":               list(range(1, 17)),     # base filters 1–16
    "num_cells":       list(range(1, 7)),       # depth 1–6 cells

    # Per-cell parameters
    "block_type":      ["standard_conv",
                        "depthwise_separable",
                        "bottleneck"],
    "kernel_size":     [3, 5],
    "stride":          [1, 2],
    "skip_connection": [True, False],
    "activation":      ["relu", "relu6"],
    "expansion_ratio": [1, 2],
}


def sample_random_architecture(k_range=None, num_cells_range=None):
    """
    Sample a fully random architecture from the search space.

    Parameters
    ----------
    k_range : list[int] or None
        Allowed values of k. Defaults to SEARCH_SPACE["k"].
    num_cells_range : list[int] or None
        Allowed values of num_cells. Defaults to SEARCH_SPACE["num_cells"].

    Returns
    -------
    dict
        Architecture dict with keys "k" and "cells".
    """
    k         = random.choice(k_range or SEARCH_SPACE["k"])
    num_cells = random.choice(num_cells_range or SEARCH_SPACE["num_cells"])

    cells = []
    for _ in range(num_cells):
        cells.append({
            "block_type":      random.choice(SEARCH_SPACE["block_type"]),
            "kernel_size":     random.choice(SEARCH_SPACE["kernel_size"]),
            "stride":          random.choice(SEARCH_SPACE["stride"]),
            "skip_connection": random.choice(SEARCH_SPACE["skip_connection"]),
            "activation":      random.choice(SEARCH_SPACE["activation"]),
            "expansion_ratio": random.choice(SEARCH_SPACE["expansion_ratio"]),
        })

    return {"k": k, "cells": cells}


def architecture_to_key(arch):
    """Deterministic string key for cache lookup."""
    cell_str = "|".join(
        f"{c['block_type'][0]}{c['kernel_size']}"
        f"s{c['stride']}sk{int(c['skip_connection'])}"
        f"{c['activation'][4:]}{c['expansion_ratio']}"
        for c in arch["cells"]
    )
    return f"k{arch['k']}_nc{len(arch['cells'])}__{cell_str}"
