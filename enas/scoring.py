"""
Multi-objective scoring function for ENAS v2.1.

score = w_acc * accuracy
      + w_eff * (1 - normalised_hardware_cost)
      + w_hr  * headroom_bonus

Default weights are set to v2.1 finalised values:
    w_acc = 0.80, w_eff = 0.15, w_hr = 0.05
"""


def compute_score(val_acc, ram, flash, macc,
                  max_ram, max_flash, max_macc,
                  accuracy_weight=0.80,
                  efficiency_weight=0.15,
                  headroom_weight=0.05):
    """
    Compute the multi-objective score for an architecture.

    Parameters
    ----------
    val_acc : float
        Proxy validation accuracy in [0, 1].
    ram, flash, macc : int
        Estimated hardware usage.
    max_ram, max_flash, max_macc : int
        Hardware constraints.
    accuracy_weight, efficiency_weight, headroom_weight : float
        Should sum to 1.0.

    Returns
    -------
    float
        Score in [0, 1]. Higher is better.
    """
    ram_ratio   = min(ram   / max_ram,   1.0)
    flash_ratio = min(flash / max_flash, 1.0)
    macc_ratio  = min(macc  / max_macc,  1.0)

    hw_cost = (ram_ratio * flash_ratio * macc_ratio) ** (1 / 3)

    headroom = 1.0 if (ram_ratio   <= 0.8 and
                       flash_ratio <= 0.8 and
                       macc_ratio  <= 0.8) else 0.0

    score = (accuracy_weight   * val_acc +
             efficiency_weight * (1.0 - hw_cost) +
             headroom_weight   * headroom)

    return round(min(score, 1.0), 6)
