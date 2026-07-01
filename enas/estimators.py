"""
Analytical hardware estimators for ENAS v2.1.

These functions replace TFLite + stm32tflm during the search phase,
providing a large speedup during search. IMPORTANT: these analytical
estimates are used ONLY for pre-flight feasibility screening, not as
accurate footprint predictions. Validated against stm32tflm ground truth,
the RAM estimate is conservative (over-predicts peak RAM by ~3x) and the
Flash estimate is a loose lower bound (under-predicts measured Flash by
~6.5x). Feasibility decisions remain safe because the binding MCU
constraint is peak RAM and the RAM estimate errs on the conservative side.
All resource numbers reported in the paper (Table 3) are MEASURED with
stm32tflm, not these analytical estimates.
"""

import numpy as np


def estimate_macc(input_shape, architecture):
    """
    Analytically compute MACCs for an ENAS v2 cell-based architecture.

    Parameters
    ----------
    input_shape : tuple
        (W, H, C_in) of input image.
    architecture : dict
        {'k': int, 'cells': [cell_dict, ...]}

    Returns
    -------
    int
        Estimated MACC count.
    """
    W, H, C_in = input_shape
    k     = architecture["k"]
    cells = architecture["cells"]
    macc  = 0
    n     = k

    # Stem
    macc += C_in * 3 * 3 * W * H * n
    C_in  = n

    multiplier = 2.0
    for i, cell in enumerate(cells):
        if W <= 1 or H <= 1:
            break

        ks = cell["kernel_size"]
        s  = cell["stride"]
        bt = cell["block_type"]
        er = cell["expansion_ratio"]

        n_out      = int(np.ceil(n * multiplier))
        multiplier = multiplier - 2 ** -(i + 1)

        W_out = W // s if s == 2 else W
        H_out = H // s if s == 2 else H

        if bt == "standard_conv":
            macc += C_in * ks * ks * W_out * H_out * n_out
        elif bt == "depthwise_separable":
            macc += C_in * ks * ks * W_out * H_out
            macc += C_in * 1  * 1  * W_out * H_out * n_out
        elif bt == "bottleneck":
            mid = max(1, int(C_in * er))
            macc += C_in * 1  * 1  * W_out * H_out * mid
            macc += mid  * ks * ks * W_out * H_out * mid
            macc += mid  * 1  * 1  * W_out * H_out * n_out

        W, H, C_in = W_out, H_out, n_out
        n = n_out

    # GAP → Dense (2 classes default for binary)
    macc += n * 2
    return int(macc)


def estimate_flash(input_shape, architecture, num_classes=2):
    """
    Estimate INT8 Flash usage (model weights).

    Parameters
    ----------
    input_shape : tuple
    architecture : dict
    num_classes : int

    Returns
    -------
    int
        Estimated Flash bytes (INT8 + 3 KB TFLite metadata overhead).
    """
    W, H, C_in = input_shape
    k     = architecture["k"]
    cells = architecture["cells"]
    params = 0
    n      = k

    # Stem
    params += C_in * 3 * 3 * n + n
    params += 4 * n
    C_in    = n

    multiplier = 2.0
    for i, cell in enumerate(cells):
        if W <= 1 or H <= 1:
            break

        ks = cell["kernel_size"]
        s  = cell["stride"]
        bt = cell["block_type"]
        er = cell["expansion_ratio"]

        n_out      = int(np.ceil(n * multiplier))
        multiplier = multiplier - 2 ** -(i + 1)
        W = W // s if s == 2 else W
        H = H // s if s == 2 else H

        if bt == "standard_conv":
            params += C_in * ks * ks * n_out + n_out
            params += 4 * n_out
        elif bt == "depthwise_separable":
            params += C_in * ks * ks + C_in
            params += C_in * n_out  + n_out
            params += 4 * n_out
        elif bt == "bottleneck":
            mid = max(1, int(C_in * er))
            params += C_in * mid  + mid
            params += mid  * ks * ks * mid + mid
            params += mid  * n_out + n_out
            params += 4 * n_out

        C_in = n_out
        n    = n_out

    params += n * num_classes + num_classes
    return int(params * 1 + 3072)


def estimate_ram(input_shape, architecture):
    """
    Estimate peak inference RAM (INT8).

    Peak = input buffer + largest activation tensor + scratch.
    """
    W, H, C_in = input_shape
    k     = architecture["k"]
    cells = architecture["cells"]
    n     = k

    input_buf = W * H * C_in
    peak_act  = W * H * n

    multiplier = 2.0
    for i, cell in enumerate(cells):
        if W <= 1 or H <= 1:
            break
        s     = cell["stride"]
        n_out = int(np.ceil(n * multiplier))
        multiplier = multiplier - 2 ** -(i + 1)
        W = W // s if s == 2 else W
        H = H // s if s == 2 else H
        peak_act = max(peak_act, W * H * n_out)
        n = n_out

    scratch = 512
    return int(input_buf + peak_act + scratch)
