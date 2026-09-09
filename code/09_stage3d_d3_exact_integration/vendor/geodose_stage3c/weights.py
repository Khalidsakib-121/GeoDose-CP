from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


def truncated_normal_density(a: np.ndarray, center: float, bandwidth: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a)
    interior = (a > 0.0) & (a < 1.0)
    if not interior.any():
        return out
    denominator = norm.cdf((1.0 - center) / bandwidth) - norm.cdf((0.0 - center) / bandwidth)
    z = (a[interior] - center) / bandwidth
    out[interior] = norm.pdf(z) / (bandwidth * denominator)
    return out


def intervention_density(
    a: np.ndarray,
    target_dose: float,
    bandwidth: float | None,
    endpoint_audited: bool,
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if target_dose == 0.0:
        return (a == 0.0).astype(float) if endpoint_audited else np.zeros(len(a))
    if target_dose == 1.0:
        return (a == 1.0).astype(float) if endpoint_audited else np.zeros(len(a))
    if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Interior target requires positive finite bandwidth")
    return truncated_normal_density(a, float(target_dose), float(bandwidth))


@dataclass(frozen=True)
class QuantileResult:
    q: float
    status: str
    positive_weight_count: int
    ess: float
    max_normalized_weight: float
    target_normalized_weight: float
    max_all_normalized_weight: float
    log_weight_range: float


def weighted_conformal_quantile(
    sorted_scores: np.ndarray,
    sorted_logw: np.ndarray,
    target_logw: float,
    alpha: float,
) -> QuantileResult:
    scores = np.asarray(sorted_scores, dtype=float)
    logw = np.asarray(sorted_logw, dtype=float)
    if len(scores) != len(logw):
        raise ValueError("score/weight length mismatch")
    finite = np.isfinite(logw) & np.isfinite(scores)
    positive_count = int(finite.sum())
    if positive_count == 0 or not np.isfinite(target_logw):
        return QuantileResult(np.nan, "refused_no_positive_or_target_weight", positive_count, 0.0, 1.0, 1.0, 1.0, np.inf)
    calibration_log = logw[finite]
    calibration_scores = scores[finite]
    maximum = max(float(np.max(calibration_log)), float(target_logw))
    calibration_weight = np.exp(calibration_log - maximum)
    target_weight = float(np.exp(target_logw - maximum))
    total = float(calibration_weight.sum() + target_weight)
    calibration_norm = calibration_weight / total
    target_norm = target_weight / total
    calibration_norm_sum = float(calibration_norm.sum())
    squared_sum = float(np.sum(calibration_weight * calibration_weight))
    ess = float((calibration_weight.sum() ** 2) / squared_sum) if squared_sum > 0 else 0.0
    max_calibration_norm = float(np.max(calibration_weight / calibration_weight.sum())) if calibration_weight.sum() > 0 else 1.0
    max_all_norm = max(float(np.max(calibration_norm)), target_norm)
    log_range = float(np.max(calibration_log) - np.min(calibration_log)) if len(calibration_log) else np.inf
    level = 1.0 - float(alpha)
    if calibration_norm_sum + 1e-15 < level:
        return QuantileResult(
            np.inf,
            "returned_infinite_target_pseudomass",
            positive_count,
            ess,
            max_calibration_norm,
            target_norm,
            max_all_norm,
            log_range,
        )
    cdf = np.cumsum(calibration_norm)
    index = int(np.searchsorted(cdf, level, side="left"))
    index = min(index, len(calibration_scores) - 1)
    return QuantileResult(
        float(calibration_scores[index]),
        "returned_finite",
        positive_count,
        ess,
        max_calibration_norm,
        target_norm,
        max_all_norm,
        log_range,
    )


def interval_score(lower: float, upper: float, y: float, alpha: float) -> float:
    if not (np.isfinite(lower) and np.isfinite(upper) and np.isfinite(y)):
        return np.inf
    score = upper - lower
    if y < lower:
        score += (2.0 / alpha) * (lower - y)
    elif y > upper:
        score += (2.0 / alpha) * (y - upper)
    return float(score)
