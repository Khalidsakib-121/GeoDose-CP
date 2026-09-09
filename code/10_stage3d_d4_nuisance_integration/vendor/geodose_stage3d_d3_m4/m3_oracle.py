from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .graph_law import LocalGraphLaw, local_log_factor
from .io import D3M4Error
from .orbit import distinct_permutation_count, distinct_value_permutations, numeric_key, residual_classes


@dataclass(frozen=True)
class M3CandidateResult:
    pvalue: float
    state_count: int
    singular_state_count: int
    identity_score: float
    probability_sum: float
    source_marginals: np.ndarray
    state_records: tuple[dict[str, Any], ...]


def normalize_logweights(logw: np.ndarray) -> np.ndarray:
    logw = np.asarray(logw, dtype=float)
    if np.any(np.isnan(logw)):
        raise D3M4Error("NaN log weight")
    if np.isposinf(logw).any():
        raise D3M4Error("Positive-infinite log weight")
    finite = np.isfinite(logw)
    if not finite.any():
        raise D3M4Error("All orbit weights are non-finite")
    m = float(np.max(logw[finite]))
    w = np.zeros(len(logw), dtype=float)
    w[finite] = np.exp(logw[finite] - m)
    total = float(w.sum())
    if not (np.isfinite(total) and total > 0):
        raise D3M4Error("Orbit normalization failed")
    return w / total


def candidate_residuals(
    calibration_residual: np.ndarray,
    candidate_y: float,
    target_mean: float,
    target_scale: float = 1.0,
) -> np.ndarray:
    cal = np.asarray(calibration_residual, dtype=float)
    if not np.isfinite(cal).all() or not np.isfinite(candidate_y) or not np.isfinite(target_mean):
        raise D3M4Error("Non-finite M3 residual input")
    if not (np.isfinite(target_scale) and float(target_scale) > 0.0):
        raise D3M4Error("M3 target outcome scale must be finite and positive")
    e_target = (float(candidate_y) - float(target_mean)) / float(target_scale)
    return np.r_[cal, e_target]


def evaluate_candidate(
    calibration_residual: np.ndarray,
    candidate_y: float,
    target_mean: float,
    law: LocalGraphLaw,
    target_slot_position: int,
    target_scale: float = 1.0,
    tie_tolerance: float = 1e-12,
    keep_states: bool = True,
) -> M3CandidateResult:
    residual_payload = candidate_residuals(calibration_residual, candidate_y, target_mean, target_scale)
    if len(residual_payload) != len(law.block_nodes):
        raise D3M4Error("Residual payload count does not match block size")
    target_slot_position = int(target_slot_position)
    if not 0 <= target_slot_position < len(residual_payload):
        raise D3M4Error("Invalid M3 target slot position")
    if target_slot_position != len(residual_payload) - 1:
        raise D3M4Error("D3 exact comparator convention requires the target slot/source to be the final block position")
    target_candidate_residual = float(residual_payload[-1])
    identity_score = abs(target_candidate_residual)
    classes = residual_classes(residual_payload)
    class_lookup = {numeric_key(c.value): c.source_indices for c in classes}

    states: list[tuple[tuple[float, ...], float, bool]] = []
    for values in distinct_value_permutations(residual_payload):
        block = np.asarray(values, dtype=float)
        logw, singular = local_log_factor(block, law)
        states.append((values, logw, singular))
    expected_count = distinct_permutation_count(residual_payload)
    if len(states) != expected_count:
        raise D3M4Error(f"M3 quotient count mismatch: {len(states)} != {expected_count}")
    probs = normalize_logweights(np.asarray([s[1] for s in states], dtype=float))

    source_marginals = np.zeros(len(residual_payload), dtype=float)
    tail_mass = 0.0
    records: list[dict[str, Any]] = []
    singular_count = 0
    for idx, ((values, lw, singular), prob) in enumerate(zip(states, probs)):
        if singular:
            singular_count += 1
        target_value = float(values[target_slot_position])
        score = abs(target_value)
        if score + tie_tolerance >= identity_score:
            tail_mass += float(prob)
        members = class_lookup[numeric_key(target_value)]
        share = float(prob) / float(len(members))
        for j in members:
            source_marginals[int(j)] += share
        if keep_states:
            records.append({
                "state_index": idx,
                "assignment_residuals": ",".join(numeric_key(float(x)) for x in values),
                "target_residual": target_value,
                "target_score": score,
                "identity_score": identity_score,
                "log_spatial_factor": float(lw),
                "probability": float(prob),
                "singular_state": bool(singular),
            })
    if abs(float(source_marginals.sum()) - 1.0) > 1e-10:
        raise D3M4Error("M3 source marginal normalization failed")
    return M3CandidateResult(
        pvalue=float(min(max(tail_mass, 0.0), 1.0)),
        state_count=len(states),
        singular_state_count=singular_count,
        identity_score=identity_score,
        probability_sum=float(probs.sum()),
        source_marginals=source_marginals,
        state_records=tuple(records),
    )
