"""
Architecture mutation strategies for Stage 3 of the ENAS v2.1 search.

Currently implements:
    - mutate_architecture : single-parameter random mutation
    - guided_mutate       : block-type-aware mutation (experimental)
"""

import random
from copy import deepcopy
from enas.search_space import SEARCH_SPACE


def mutate_architecture(arch, n_mutations=1):
    """
    Single-parameter random mutation.

    Each mutation changes exactly one of:
        - k (global)
        - one parameter of one cell

    Parameters
    ----------
    arch : dict
        Architecture to mutate.
    n_mutations : int
        Number of independent mutations to apply.

    Returns
    -------
    dict
        Mutated architecture (deep copy).
    """
    mutated = deepcopy(arch)
    mutation_targets = ["k", "cell_param"]

    for _ in range(n_mutations):
        target = random.choice(mutation_targets)

        if target == "k":
            mutated["k"] = random.choice(SEARCH_SPACE["k"])

        elif target == "cell_param" and mutated["cells"]:
            cell_idx  = random.randint(0, len(mutated["cells"]) - 1)
            param_key = random.choice([
                "block_type", "kernel_size", "stride",
                "skip_connection", "activation", "expansion_ratio"
            ])
            mutated["cells"][cell_idx][param_key] = random.choice(
                SEARCH_SPACE[param_key]
            )

    return mutated


def guided_mutate(arch, top_k_block_distribution=None, n_mutations=1):
    """
    Guided mutation that prefers under-represented block types.

    When top_k_block_distribution is provided, mutations bias toward
    block types that appear less frequently in top-K survivors,
    improving exploration diversity.

    Parameters
    ----------
    arch : dict
    top_k_block_distribution : dict[str, int] or None
        Counter of block types in top-K survivors.
    n_mutations : int

    Returns
    -------
    dict
        Mutated architecture.
    """
    if top_k_block_distribution is None:
        return mutate_architecture(arch, n_mutations)

    mutated = deepcopy(arch)

    for _ in range(n_mutations):
        # 30% guided diversity, 70% standard mutation
        if random.random() < 0.3 and mutated["cells"]:
            cell_idx = random.randint(0, len(mutated["cells"]) - 1)
            least_used = min(
                SEARCH_SPACE["block_type"],
                key=lambda b: top_k_block_distribution.get(b, 0)
            )
            mutated["cells"][cell_idx]["block_type"] = least_used
        else:
            mutated = mutate_architecture(mutated, n_mutations=1)

    return mutated
