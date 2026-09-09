from __future__ import annotations

import math

import numpy as np
from scipy.special import betaln, ndtr

from .io import Stage3DError, require


def beta_log_pdf(a: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    require(np.all((a > 0.0) & (a < 1.0)), "Beta density evaluated outside the open unit interval")
    require(np.all(alpha > 0.0) and np.all(beta > 0.0), "Beta shape parameter is nonpositive")
    return (alpha - 1.0) * np.log(a) + (beta - 1.0) * np.log1p(-a) - betaln(alpha, beta)


def observational_mixed_density(
    a: np.ndarray,
    pi0: np.ndarray,
    pi1: np.ndarray,
    pii: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    a, pi0, pi1, pii, alpha, beta = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(pi0, dtype=float),
        np.asarray(pi1, dtype=float),
        np.asarray(pii, dtype=float),
        np.asarray(alpha, dtype=float),
        np.asarray(beta, dtype=float),
    )
    require(np.all(pi0 >= 0.0) and np.all(pi1 >= 0.0) and np.all(pii >= 0.0), "Negative mixed-treatment probability")
    require(np.allclose(pi0 + pi1 + pii, 1.0, atol=1e-12, rtol=0.0), "Mixed-treatment category probabilities do not sum to one")
    result = np.zeros(a.shape, dtype=float)
    at0 = a == 0.0
    at1 = a == 1.0
    interior = (a > 0.0) & (a < 1.0)
    invalid = ~(at0 | at1 | interior)
    require(not np.any(invalid), "Treatment value outside [0,1]")
    result[at0] = pi0[at0]
    result[at1] = pi1[at1]
    if np.any(interior):
        result[interior] = pii[interior] * np.exp(beta_log_pdf(a[interior], alpha[interior], beta[interior]))
    return result


def truncated_gaussian_density(a: np.ndarray, center: float, bandwidth: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    require(0.0 < center < 1.0, "Interior target center must lie in (0,1)")
    require(bandwidth > 0.0 and np.isfinite(bandwidth), "Invalid intervention bandwidth")
    normalizer = float(ndtr((1.0 - center) / bandwidth) - ndtr((0.0 - center) / bandwidth))
    require(normalizer > 0.0 and np.isfinite(normalizer), "Invalid truncated-Gaussian normalizer")
    result = np.zeros_like(a, dtype=float)
    interior = (a > 0.0) & (a < 1.0)
    z = (a[interior] - center) / bandwidth
    result[interior] = np.exp(-0.5 * z * z) / (bandwidth * math.sqrt(2.0 * math.pi) * normalizer)
    return result


def intervention_density(
    a: np.ndarray,
    target_dose: float,
    bandwidth: float | None,
    *,
    endpoint_audited: bool,
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if target_dose == 0.0 or target_dose == 1.0:
        if not endpoint_audited:
            raise Stage3DError("D1_UNSUPPORTED_ENDPOINT: endpoint is not audited")
        return (a == target_dose).astype(float)
    require(0.0 < target_dose < 1.0, "Target dose outside [0,1]")
    require(bandwidth is not None, "Interior intervention requires bandwidth")
    return truncated_gaussian_density(a, float(target_dose), float(bandwidth))


def representative_mixed_normalization_error(
    pi0: float,
    pi1: float,
    pii: float,
    alpha: float,
    beta: float,
    *,
    nodes: int = 512,
) -> float:
    x, w = np.polynomial.legendre.leggauss(nodes)
    a = 0.5 * (x + 1.0)
    weights = 0.5 * w
    interior = float(np.sum(weights * observational_mixed_density(
        a,
        np.full_like(a, pi0),
        np.full_like(a, pi1),
        np.full_like(a, pii),
        np.full_like(a, alpha),
        np.full_like(a, beta),
    )))
    return abs((pi0 + pi1 + interior) - 1.0)


def intervention_normalization_error(target_dose: float, bandwidth: float | None, endpoint_audited: bool) -> float:
    if target_dose in {0.0, 1.0}:
        mass = float(intervention_density(np.asarray([target_dose]), target_dose, bandwidth, endpoint_audited=endpoint_audited)[0])
        return abs(mass - 1.0)
    x, w = np.polynomial.legendre.leggauss(512)
    a = 0.5 * (x + 1.0)
    weights = 0.5 * w
    integral = float(np.sum(weights * intervention_density(a, target_dose, bandwidth, endpoint_audited=endpoint_audited)))
    return abs(integral - 1.0)
