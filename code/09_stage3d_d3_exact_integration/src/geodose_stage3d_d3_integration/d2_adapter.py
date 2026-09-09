from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .io import IntegrationError


@dataclass(frozen=True)
class D2Candidate:
    p_value: float
    included: bool
    candidate_y: float
    candidate_hex: str
    distinct_states: int
    probability_sum: float
    singular_inclusion: bool


class AcceptedD2ExactAdapter:
    """Read-only adapter over accepted D2 evaluated candidate states.

    This is intentionally not a new G1/G2/G3 implementation.  It exposes only
    candidates already evaluated by accepted D2 and refuses all others.
    """
    def __init__(self, trace: pd.DataFrame, fixture_id: str, alpha: float):
        self.fixture_id = str(fixture_id)
        self.alpha = float(alpha)
        self.trace = trace[trace["fixture_id"].astype(str) == self.fixture_id].copy()
        if self.trace.empty:
            raise IntegrationError(f"D2 fixture absent: {fixture_id}")
        if self.trace["candidate_hex"].duplicated().any():
            raise IntegrationError("Duplicate candidate_hex in accepted D2 trace")
        self._by_hex = {str(r.candidate_hex): r for r in self.trace.itertuples(index=False)}

    def evaluate_registered(self, candidate_y: float, candidate_hex: str) -> D2Candidate:
        h = str(candidate_hex)
        if h not in self._by_hex:
            raise IntegrationError("D3_M6_CANDIDATE_NOT_IN_ACCEPTED_D2_TRACE")
        row = self._by_hex[h]
        if float(candidate_y).hex() != h:
            raise IntegrationError("candidate_y/candidate_hex mismatch")
        p = float(row.conservative_p)
        singular = bool(row.singular_conservative_inclusion)
        included = bool(row.conservative_accept)
        expected = bool((p > self.alpha) or singular)
        if included != expected:
            raise IntegrationError("Accepted D2 inclusion rule mismatch")
        if not (0 <= p <= 1 and abs(float(row.probability_sum)-1.0) <= 1e-10):
            raise IntegrationError("Invalid accepted D2 probability record")
        return D2Candidate(p, included, float(row.candidate_y), h, int(row.distinct_states), float(row.probability_sum), singular)
