from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import D3M4Error
from .m3_oracle import normalize_logweights


@dataclass(frozen=True)
class M4CandidateResult:
    q4: np.ndarray
    log_product_weights: np.ndarray
    scores: np.ndarray
    observed_score: float
    pvalue: float
    probability_sum: float
    normalization_count: int


def evaluate_m4(
    spatial_source_marginal: np.ndarray,
    source_residuals: np.ndarray,
    log_dose_ratio: np.ndarray,
    target_source_index: int,
    tie_tolerance: float = 1e-12,
) -> M4CandidateResult:
    """Oracle M4: one normalized product of M2 dose ratio and M3 spatial source marginal.

    The N2/N3 target-design ratio is intentionally absent.  `spatial_source_marginal`
    must already be the M3 target-source marginal s_j(y); the full M3 orbit law is not accepted.
    """
    s = np.asarray(spatial_source_marginal, dtype=float)
    e = np.asarray(source_residuals, dtype=float)
    logd = np.asarray(log_dose_ratio, dtype=float)
    if s.ndim != 1 or e.shape != s.shape or logd.shape != s.shape:
        raise D3M4Error("M4 source arrays must have identical one-dimensional shape")
    if np.any(~np.isfinite(s)) or np.any(s < 0):
        raise D3M4Error("Invalid M3 spatial source marginal supplied to M4")
    if abs(float(s.sum()) - 1.0) > 1e-10:
        raise D3M4Error("M3 spatial source marginal does not sum to one")
    if np.any(np.isnan(logd)) or np.isposinf(logd).any():
        raise D3M4Error("M4 dose log-ratios may be finite or -inf, never NaN/+inf")
    if not np.isfinite(e).all():
        raise D3M4Error("M4 source residuals must be finite")
    t = int(target_source_index)
    if not 0 <= t < len(s):
        raise D3M4Error("Invalid M4 target source index")

    logs = np.full(len(s), -np.inf, dtype=float)
    pos = s > 0.0
    logs[pos] = np.log(s[pos])
    log_product = logs + logd
    q4 = normalize_logweights(log_product)
    scores = np.abs(e)
    observed_score = float(scores[t])
    p = float(q4[scores + float(tie_tolerance) >= observed_score].sum())
    return M4CandidateResult(
        q4=q4,
        log_product_weights=log_product,
        scores=scores,
        observed_score=observed_score,
        pvalue=float(min(max(p, 0.0), 1.0)),
        probability_sum=float(q4.sum()),
        normalization_count=1,
    )
