from __future__ import annotations

import numpy as np

from .io import Stage3DError, require


def conditional_mean_matrix(outcome_form: str, baseline_by_slot: np.ndarray, x1_by_slot: np.ndarray, A_payload: np.ndarray) -> np.ndarray:
    baseline = np.asarray(baseline_by_slot, dtype=float)[:, None]
    x1 = np.asarray(x1_by_slot, dtype=float)[:, None]
    a = np.asarray(A_payload, dtype=float)[None, :]
    if outcome_form == "linear":
        return baseline + 0.50 * a
    if outcome_form == "nonlinear":
        return baseline + 0.55 * a - 0.35 * a * a + 0.25 * np.sin(2.0 * np.pi * a) + 0.20 * a * x1
    raise Stage3DError(f"Unknown outcome form: {outcome_form}")


def affine_residual_matrix(
    permutations: np.ndarray,
    Y_payload: np.ndarray,
    outcome_mean: np.ndarray,
    outcome_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    permutations = np.asarray(permutations, dtype=int)
    rows = np.arange(permutations.shape[1])[None, :]
    y = np.asarray(Y_payload, dtype=float)[permutations]
    mean = np.asarray(outcome_mean, dtype=float)[rows, permutations]
    scale = np.asarray(outcome_scale, dtype=float)[rows, permutations]
    require(np.all(scale > 0.0) and np.all(np.isfinite(scale)), "Invalid outcome scale")
    residual = (y - mean) / scale
    log_abs_dr_dy = -np.log(scale)
    return residual, log_abs_dr_dy


def affine_residual_scalar(
    assignment: np.ndarray,
    Y_payload: np.ndarray,
    outcome_mean: np.ndarray,
    outcome_scale: np.ndarray,
) -> tuple[np.ndarray, float]:
    assignment = np.asarray(assignment, dtype=int)
    residual = np.empty(len(assignment), dtype=float)
    log_j = 0.0
    for slot, payload in enumerate(assignment.tolist()):
        scale = float(outcome_scale[slot, payload])
        require(scale > 0.0 and np.isfinite(scale), "Invalid outcome scale")
        residual[slot] = (float(Y_payload[payload]) - float(outcome_mean[slot, payload])) / scale
        log_j += -float(np.log(scale))
    return residual, log_j


def inverse_power_transform(residual: np.ndarray, power: float, *, minimum_abs_residual: float) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(residual, dtype=float)
    require(power >= 1.0, "Transform power must be at least one")
    if abs(power - 1.0) < 1e-15:
        return residual.copy(), np.zeros_like(residual, dtype=float)
    absolute = np.abs(residual)
    if np.any(absolute < minimum_abs_residual):
        raise Stage3DError("D1_TRANSFORM_SINGULAR: transformed-GMRF residual is numerically zero")
    latent = np.sign(residual) * absolute ** (1.0 / power)
    log_abs_dz_dr = -np.log(power) + (1.0 / power - 1.0) * np.log(absolute)
    return latent, log_abs_dz_dr
