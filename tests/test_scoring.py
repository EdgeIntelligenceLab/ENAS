"""Unit tests for multi-objective scoring."""
import pytest
from enas.scoring import compute_score


def test_high_accuracy_high_score():
    high = compute_score(0.85, 10000, 20000, 500000,
                          100000, 200000, 5000000)
    low  = compute_score(0.55, 10000, 20000, 500000,
                          100000, 200000, 5000000)
    assert high > low


def test_score_in_unit_interval():
    s = compute_score(0.75, 50000, 100000, 1000000,
                       100000, 200000, 5000000)
    assert 0 <= s <= 1


def test_headroom_bonus_applies():
    """Architecture using ≤80% of all budgets gets the headroom bonus."""
    s_with_headroom = compute_score(
        0.70, 50000, 100000, 1000000,        # 50%, 50%, 20%
        100000, 200000, 5000000,
        accuracy_weight=0.80, efficiency_weight=0.15, headroom_weight=0.05
    )
    s_no_headroom = compute_score(
        0.70, 95000, 195000, 4900000,        # 95%, ~98%, 98%
        100000, 200000, 5000000,
        accuracy_weight=0.80, efficiency_weight=0.15, headroom_weight=0.05
    )
    assert s_with_headroom > s_no_headroom
