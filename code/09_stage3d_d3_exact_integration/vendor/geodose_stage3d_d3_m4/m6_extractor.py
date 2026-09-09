from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import D3M4Error


@dataclass(frozen=True)
class M6ExtractedCandidate:
    q6: np.ndarray
    source_scores: np.ndarray
    state_probability_sum: float
    archived_conservative_p: float
    replay_conservative_p: float
    state_count: int
    source_state_counts: np.ndarray
    observed_score: float
    candidate_hex: str


def parse_assignment(text: str) -> tuple[int, ...]:
    try:
        vals = tuple(int(x) for x in str(text).split(","))
    except Exception as exc:
        raise D3M4Error(f"Invalid D2 assignment string: {text}") from exc
    if not vals or sorted(vals) != list(range(len(vals))):
        raise D3M4Error(f"D2 assignment is not a permutation: {text}")
    return vals


def extract_m6_target_source_marginal(
    candidate_states: pd.DataFrame,
    candidate_trace: pd.DataFrame,
    fixture_id: str,
    candidate_hex: str,
    target_slot_position: int,
    score_tolerance: float = 1e-12,
) -> M6ExtractedCandidate:
    g = candidate_states[
        (candidate_states["fixture_id"].astype(str) == str(fixture_id))
        & (candidate_states["candidate_hex"].astype(str) == str(candidate_hex))
    ].copy()
    if g.empty:
        raise D3M4Error(f"No D2 state audit for {fixture_id}/{candidate_hex}")
    n = len(parse_assignment(g.iloc[0]["assignment"]))
    tpos = int(target_slot_position)
    if not 0 <= tpos < n:
        raise D3M4Error("Invalid D2 target-slot position")
    q6 = np.zeros(n, dtype=float)
    score_values: list[list[float]] = [[] for _ in range(n)]
    counts = np.zeros(n, dtype=int)
    replay_p = 0.0
    for row in g.itertuples(index=False):
        ass = parse_assignment(row.assignment)
        if len(ass) != n:
            raise D3M4Error("Inconsistent D2 assignment lengths")
        src = int(ass[tpos])
        prob = float(row.state_probability)
        if not (np.isfinite(prob) and prob >= 0.0):
            raise D3M4Error("Invalid D2 state probability")
        q6[src] += prob
        counts[src] += 1
        score_values[src].append(float(row.state_score))
        if bool(row.certified_strict_greater) or bool(row.certified_structural_tie) or bool(row.ambiguous_included):
            replay_p += prob
    source_scores = np.full(n, np.nan, dtype=float)
    for j, vals in enumerate(score_values):
        if not vals:
            raise D3M4Error(f"D2 target-source index {j} has no states")
        arr = np.asarray(vals, dtype=float)
        if not np.isfinite(arr).all() or float(arr.max() - arr.min()) > float(score_tolerance):
            raise D3M4Error(f"D2 source score is not target-index determined for source {j}")
        source_scores[j] = float(arr[0])
    observed_scores = g["observed_score"].to_numpy(float)
    if not np.isfinite(observed_scores).all() or float(observed_scores.max() - observed_scores.min()) > score_tolerance:
        raise D3M4Error("D2 observed score is not constant within candidate state audit")
    tr = candidate_trace[
        (candidate_trace["fixture_id"].astype(str) == str(fixture_id))
        & (candidate_trace["candidate_hex"].astype(str) == str(candidate_hex))
    ]
    if len(tr) != 1:
        raise D3M4Error(f"Expected one D2 candidate trace row for {fixture_id}/{candidate_hex}; found {len(tr)}")
    archived = float(tr.iloc[0]["conservative_p"])
    return M6ExtractedCandidate(
        q6=q6,
        source_scores=source_scores,
        state_probability_sum=float(g["state_probability"].sum()),
        archived_conservative_p=archived,
        replay_conservative_p=float(replay_p),
        state_count=int(len(g)),
        source_state_counts=counts,
        observed_score=float(observed_scores[0]),
        candidate_hex=str(candidate_hex),
    )
