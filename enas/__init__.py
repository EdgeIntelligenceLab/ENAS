"""
ENAS — Edge Neural Architecture Search for TinyML.

Public API:
    ENAS         : Main NAS class (v2.1)
    SEARCH_SPACE : Cell-based search space definition
    estimate_ram, estimate_flash, estimate_macc : Analytical hardware estimators
    compute_score : Multi-objective scoring function
"""

from enas.enas_v2_1 import ENAS
from enas.search_space import SEARCH_SPACE
from enas.estimators import estimate_ram, estimate_flash, estimate_macc
from enas.scoring import compute_score

__version__ = "2.1.0"
__all__ = [
    "ENAS",
    "SEARCH_SPACE",
    "estimate_ram",
    "estimate_flash",
    "estimate_macc",
    "compute_score",
]
