#!/usr/bin/env python
"""
generate_all_figures.py
=======================
One-stop script to regenerate every paper figure from parsed CSVs.

Usage:
    python figures/scripts/generate_all_figures.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "figures" / "scripts"


def run_script(script_name):
    """Run a figure script in a subprocess."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"[SKIP] {script_name} not found.")
        return False

    print(f"\n── Running {script_name} ──")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, check=False,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"[ERROR] {script_name} failed:\n{result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Could not run {script_name}: {e}")
        return False


def main():
    print("=" * 70)
    print("  Generating all paper figures from parsed CSVs")
    print("=" * 70)

    scripts = [
        "plot_pipeline.py",       # Figure 1 (pipeline + architecture)
        "plot_pareto.py",         # Figure 2 (accuracy vs search time)
        "plot_heatmaps.py",       # Figure 3 (4-panel deltas)
    ]

    success_count = 0
    for s in scripts:
        if run_script(s):
            success_count += 1

    print(f"\n{'=' * 70}")
    print(f"  ✔ {success_count}/{len(scripts)} figure scripts completed successfully")
    print(f"  Output directory: {REPO_ROOT / 'figures' / 'output'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
