"""Unit tests for search space sampling and mutation."""
import pytest
from enas.search_space import (
    SEARCH_SPACE, sample_random_architecture, architecture_to_key
)
from enas.mutations import mutate_architecture


def test_random_architecture_in_bounds():
    for _ in range(50):
        arch = sample_random_architecture()
        assert 1 <= arch["k"] <= 16
        assert 1 <= len(arch["cells"]) <= 6
        for cell in arch["cells"]:
            assert cell["block_type"] in SEARCH_SPACE["block_type"]
            assert cell["kernel_size"] in SEARCH_SPACE["kernel_size"]
            assert cell["stride"] in SEARCH_SPACE["stride"]


def test_architecture_key_deterministic():
    arch1 = sample_random_architecture()
    k1 = architecture_to_key(arch1)
    k2 = architecture_to_key(arch1)
    assert k1 == k2


def test_mutation_changes_something():
    arch = sample_random_architecture()
    mutated = mutate_architecture(arch, n_mutations=1)
    # Most of the time mutation should change at least one parameter
    # (occasionally a random mutation may produce an identical config)
    key_orig = architecture_to_key(arch)
    keys_mut = {architecture_to_key(mutate_architecture(arch, 1))
                for _ in range(20)}
    assert len(keys_mut - {key_orig}) > 0
