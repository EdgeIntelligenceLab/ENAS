#!/usr/bin/env python
"""Top-level convenience wrapper.

For convenience, this script runs `scripts/run_all_experiments.py`. It supports
the same arguments. See:
    python run_all_experiments.py --help
"""
import sys, subprocess
from pathlib import Path
sys.exit(subprocess.call(
    [sys.executable, str(Path(__file__).parent / "scripts" / "run_all_experiments.py")] + sys.argv[1:]
))
