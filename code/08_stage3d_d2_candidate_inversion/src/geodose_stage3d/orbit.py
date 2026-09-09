from __future__ import annotations

from collections import Counter
from typing import Hashable, Sequence

import numpy as np

from .io import Stage3DError, require


def distinct_index_permutations(keys: Sequence[Hashable]) -> np.ndarray:
    """Enumerate one source-index assignment for each distinct multiset state.

    The returned row stores the source payload index assigned to each fixed slot.
    Numerically duplicate movable payloads are quotient-identified regardless of
    their source origin. Candidate replacement must occur before keys are supplied.
    """
    if len(keys) == 0:
        raise Stage3DError("Orbit cannot be empty")
    groups: dict[Hashable, list[int]] = {}
    key_order: list[Hashable] = []
    for index, key in enumerate(keys):
        if key not in groups:
            groups[key] = []
            key_order.append(key)
        groups[key].append(index)

    remaining = Counter(keys)
    current: list[Hashable] = []
    states: list[tuple[int, ...]] = []

    def recurse() -> None:
        if len(current) == len(keys):
            used = {key: 0 for key in key_order}
            assignment: list[int] = []
            for key in current:
                assignment.append(groups[key][used[key]])
                used[key] += 1
            states.append(tuple(assignment))
            return
        for key in key_order:
            if remaining[key] <= 0:
                continue
            remaining[key] -= 1
            current.append(key)
            recurse()
            current.pop()
            remaining[key] += 1

    recurse()
    identity = tuple(range(len(keys)))
    require(identity in states, "D1_ENUMERATION_COUNT_MISMATCH: identity state absent")
    states.insert(0, states.pop(states.index(identity)))
    array = np.asarray(states, dtype=np.int32)
    require(array.ndim == 2 and array.shape[1] == len(keys), "Permutation array shape mismatch")
    require(len({tuple(row) for row in array.tolist()}) == len(array), "Duplicate quotient-orbit state generated")
    return array


def assignment_string(assignment: np.ndarray) -> str:
    return ",".join(str(int(value)) for value in np.asarray(assignment).tolist())


def validate_exact_block_size(block_size: int, maximum: int = 8) -> None:
    if block_size <= 0:
        raise Stage3DError("D1_BLOCK_TOO_LARGE: block size must be positive")
    if block_size > maximum:
        raise Stage3DError(f"D1_BLOCK_TOO_LARGE: {block_size} exceeds {maximum}")
