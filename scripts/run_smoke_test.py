#!/usr/bin/env python
"""
run_smoke_test.py
=================
Quick verification script that exercises the ENAS framework end-to-end
without requiring a real dataset. Useful for reviewers to verify their
installation works.

Builds a tiny dummy architecture, runs analytical estimation, and verifies
the multi-objective scoring function. Should complete in <10 seconds.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_imports():
    print("[1/5] Testing imports...", end=" ")
    from enas import ENAS, SEARCH_SPACE
    from enas import estimate_ram, estimate_flash, estimate_macc, compute_score
    from enas.blocks import build_model
    from enas.search_space import sample_random_architecture
    from enas.mutations import mutate_architecture
    print("✓")


def test_estimators():
    print("[2/5] Testing analytical estimators...", end=" ")
    from enas import estimate_ram, estimate_flash, estimate_macc

    # Simple test architecture
    arch = {
        "k": 4,
        "cells": [
            {"block_type": "depthwise_separable", "kernel_size": 3,
             "stride": 2, "skip_connection": False,
             "activation": "relu", "expansion_ratio": 1},
            {"block_type": "bottleneck", "kernel_size": 3,
             "stride": 2, "skip_connection": False,
             "activation": "relu6", "expansion_ratio": 2},
        ]
    }
    input_shape = (50, 50, 3)

    ram   = estimate_ram(input_shape, arch)
    flash = estimate_flash(input_shape, arch)
    macc  = estimate_macc(input_shape, arch)

    assert 1000 < ram < 100000, f"RAM out of range: {ram}"
    assert 100 < flash < 100000, f"Flash out of range: {flash}"
    assert 1000 < macc < 10000000, f"MACC out of range: {macc}"
    print(f"✓  (RAM={ram}, Flash={flash}, MACC={macc})")


def test_scoring():
    print("[3/5] Testing scoring function...", end=" ")
    from enas import compute_score

    # High accuracy, low hardware usage → high score
    s_high = compute_score(0.85, 10000, 20000, 500000,
                           100000, 200000, 5000000)
    # Low accuracy, high hardware usage → low score
    s_low = compute_score(0.55, 95000, 195000, 4900000,
                          100000, 200000, 5000000)
    assert s_high > s_low, f"Scoring inconsistent: {s_high} vs {s_low}"
    assert 0 <= s_high <= 1, f"Score out of [0,1]: {s_high}"
    print(f"✓  (s_high={s_high:.3f}, s_low={s_low:.3f})")


def test_search_space():
    print("[4/5] Testing search space sampling...", end=" ")
    from enas.search_space import sample_random_architecture, architecture_to_key
    from enas.mutations import mutate_architecture

    arch1 = sample_random_architecture()
    arch2 = sample_random_architecture()
    key1  = architecture_to_key(arch1)
    key2  = architecture_to_key(arch2)
    mutated = mutate_architecture(arch1, n_mutations=1)

    assert 1 <= arch1["k"] <= 16
    assert 1 <= len(arch1["cells"]) <= 6
    assert isinstance(key1, str)
    assert mutated["k"] != arch1["k"] or mutated["cells"] != arch1["cells"]
    print("✓")


def test_model_building():
    print("[5/5] Testing TensorFlow model construction...", end=" ")
    try:
        from enas.blocks import build_model
        arch = {
            "k": 4,
            "cells": [
                {"block_type": "standard_conv", "kernel_size": 3,
                 "stride": 1, "skip_connection": True,
                 "activation": "relu", "expansion_ratio": 1},
            ]
        }
        model, macc, cell_limited = build_model((50, 50, 3), 2, arch)
        assert not cell_limited
        assert macc > 0
        param_count = model.count_params()
        print(f"✓  (params={param_count}, macc={macc})")
    except ImportError as e:
        print(f"⚠ Skipped (TensorFlow not available): {e}")


def main():
    print("=" * 60)
    print("  ENAS Smoke Test")
    print("=" * 60)
    try:
        test_imports()
        test_estimators()
        test_scoring()
        test_search_space()
        test_model_building()
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✓ All smoke tests passed — installation is working.")
    print("=" * 60)
    print("\nNext steps:")
    print("  python scripts/run_single_experiment.py --list-hardware")
    print("  python scripts/run_single_experiment.py --hardware STM32H743ZI --size 50")


if __name__ == "__main__":
    main()
