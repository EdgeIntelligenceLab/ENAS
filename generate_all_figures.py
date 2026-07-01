#!/usr/bin/env python
"""Top-level convenience wrapper.

Runs all figure-generation scripts. Equivalent to:
    python figures/scripts/generate_all_figures.py
"""
import sys, subprocess
from pathlib import Path
sys.exit(subprocess.call(
    [sys.executable, str(Path(__file__).parent / "figures" / "scripts" / "generate_all_figures.py")] + sys.argv[1:]
))
