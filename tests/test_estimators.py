"""Unit tests for analytical hardware estimators."""
import pytest
from enas.estimators import estimate_ram, estimate_flash, estimate_macc


@pytest.fixture
def simple_arch():
    return {
        "k": 4,
        "cells": [
            {"block_type": "depthwise_separable", "kernel_size": 3,
             "stride": 2, "skip_connection": False,
             "activation": "relu", "expansion_ratio": 1},
            {"block_type": "standard_conv", "kernel_size": 3,
             "stride": 2, "skip_connection": False,
             "activation": "relu", "expansion_ratio": 1},
        ]
    }


def test_ram_estimator(simple_arch):
    ram = estimate_ram((50, 50, 3), simple_arch)
    assert 1000 < ram < 100000
    # Larger input should give larger RAM
    assert estimate_ram((128, 128, 3), simple_arch) > ram


def test_flash_estimator(simple_arch):
    flash = estimate_flash((50, 50, 3), simple_arch)
    assert 500 < flash < 200000


def test_macc_estimator(simple_arch):
    macc = estimate_macc((50, 50, 3), simple_arch)
    assert 1000 < macc < 10_000_000


def test_macc_scales_with_resolution(simple_arch):
    m_small = estimate_macc((32, 32, 3),  simple_arch)
    m_large = estimate_macc((128, 128, 3), simple_arch)
    assert m_large > m_small * 5  # large 4× = 16× more pixels


def test_depthwise_cheaper_than_standard():
    """Depthwise separable should use less MACC than standard at same depth."""
    ds_arch = {
        "k": 8,
        "cells": [{"block_type": "depthwise_separable", "kernel_size": 3,
                   "stride": 1, "skip_connection": False,
                   "activation": "relu", "expansion_ratio": 1}]
    }
    std_arch = {
        "k": 8,
        "cells": [{"block_type": "standard_conv", "kernel_size": 3,
                   "stride": 1, "skip_connection": False,
                   "activation": "relu", "expansion_ratio": 1}]
    }
    assert estimate_macc((50, 50, 3), ds_arch) < estimate_macc((50, 50, 3), std_arch)
