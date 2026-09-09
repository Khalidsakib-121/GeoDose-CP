from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class ResidualClass:
    value: float
    source_indices: tuple[int, ...]


def canonical_float(value: float) -> float:
    x = float(value)
    if not np.isfinite(x):
        raise ValueError("Orbit payload must be finite")
    # IEEE signed zeros are numerically identical and must belong to one exact duplicate class.
    if x == 0.0:
        x = 0.0
    return x


def numeric_key(value: float) -> str:
    # Exact round-trippable numeric identity after the signed-zero canonicalization above.
    return format(canonical_float(value), ".17g")


def residual_classes(values: np.ndarray) -> list[ResidualClass]:
    arr = np.asarray(values, dtype=float)
    buckets: dict[str, list[int]] = {}
    repr_value: dict[str, float] = {}
    for idx, value in enumerate(arr):
        val = canonical_float(float(value))
        key = numeric_key(val)
        buckets.setdefault(key, []).append(int(idx))
        repr_value.setdefault(key, val)
    return [ResidualClass(repr_value[k], tuple(v)) for k, v in buckets.items()]


def distinct_value_permutations(values: np.ndarray) -> Iterator[tuple[float, ...]]:
    arr = np.asarray(values, dtype=float)
    keyed = [numeric_key(x) for x in arr]
    counter = Counter(keyed)
    representative = {numeric_key(x): canonical_float(float(x)) for x in arr}
    keys = sorted(counter.keys(), key=lambda k: float(representative[k]))
    n = len(arr)
    out: list[float] = [0.0] * n

    def rec(pos: int) -> Iterator[tuple[float, ...]]:
        if pos == n:
            yield tuple(out)
            return
        for key in keys:
            if counter[key] <= 0:
                continue
            counter[key] -= 1
            out[pos] = representative[key]
            yield from rec(pos + 1)
            counter[key] += 1

    yield from rec(0)


def distinct_permutation_count(values: np.ndarray) -> int:
    from math import factorial
    arr = np.asarray(values, dtype=float)
    counts = Counter(numeric_key(x) for x in arr)
    total = factorial(len(arr))
    for c in counts.values():
        total //= factorial(c)
    return int(total)
