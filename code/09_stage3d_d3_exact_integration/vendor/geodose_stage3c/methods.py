from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .weights import weighted_conformal_quantile

# M3 and final M4 are deliberately not claimed in this gate.  H1 is the
# separately frozen geographic-distance heuristic (ABL7), and H4 is its naive
# product with M2.  Final M3/M4 require the graph-local residual-law machinery
# implemented alongside G2/G3 or N2/N3.
METHODS = ("M1", "M2", "H1", "H4", "M5")


@dataclass(frozen=True)
class MethodResult:
    lower: float
    upper: float
    center: float
    quantile: float
    interval_status: str
    refusal_code: str
    positive_weight_count: int
    ess: float
    max_normalized_weight: float
    target_normalized_weight: float
    max_all_normalized_weight: float
    log_weight_range: float


def build_interval(
    center: float,
    sorted_scores: np.ndarray,
    sorted_logw: np.ndarray,
    target_logw: float,
    alpha: float,
) -> MethodResult:
    result = weighted_conformal_quantile(sorted_scores, sorted_logw, target_logw, alpha)
    if result.status.startswith("refused"):
        return MethodResult(
            np.nan,
            np.nan,
            float(center),
            np.nan,
            "refused",
            "R03_NO_POSITIVE_OR_TARGET_WEIGHT",
            result.positive_weight_count,
            result.ess,
            result.max_normalized_weight,
            result.target_normalized_weight,
            result.max_all_normalized_weight,
            result.log_weight_range,
        )
    if np.isinf(result.q):
        return MethodResult(
            -np.inf,
            np.inf,
            float(center),
            np.inf,
            result.status,
            "",
            result.positive_weight_count,
            result.ess,
            result.max_normalized_weight,
            result.target_normalized_weight,
            result.max_all_normalized_weight,
            result.log_weight_range,
        )
    lower = float(center - result.q)
    upper = float(center + result.q)
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return MethodResult(
            np.nan,
            np.nan,
            float(center),
            float(result.q),
            "refused",
            "R10_NUMERICAL_NONFINITE",
            result.positive_weight_count,
            result.ess,
            result.max_normalized_weight,
            result.target_normalized_weight,
            result.max_all_normalized_weight,
            result.log_weight_range,
        )
    return MethodResult(
        lower,
        upper,
        float(center),
        float(result.q),
        result.status,
        "",
        result.positive_weight_count,
        result.ess,
        result.max_normalized_weight,
        result.target_normalized_weight,
        result.max_all_normalized_weight,
        result.log_weight_range,
    )
