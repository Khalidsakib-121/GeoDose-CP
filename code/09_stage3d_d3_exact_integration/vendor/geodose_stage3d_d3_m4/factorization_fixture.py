from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .graph_law import LocalGraphLaw, local_log_factor
from .io import D3M4Error
from .m3_oracle import candidate_residuals, evaluate_candidate, normalize_logweights
from .m4_oracle import evaluate_m4
from .orbit import numeric_key


@dataclass(frozen=True)
class FactorizingCandidateResult:
    candidate_y: float
    q3: np.ndarray
    q4: np.ndarray
    q6_reference: np.ndarray
    p3: float
    p4: float
    p6_reference: float
    treatment_log_factor_range: float
    residual_log_factor_range: float
    q3_q4_max_abs: float
    q3_q6_max_abs: float
    p_max_abs: float
    accepted3: bool
    accepted4: bool
    accepted6: bool


def _homogeneous_g(a: np.ndarray) -> np.ndarray:
    # Positive, treatment-dependent but slot-homogeneous diagnostic density/mass.
    arr = np.asarray(a, dtype=float)
    val = 0.75 + 0.5 * arr
    if np.any(val <= 0) or not np.isfinite(val).all():
        raise D3M4Error("Factorizing fixture treatment density invalid")
    return val


def evaluate_factorizing_candidate(
    calibration_residual: np.ndarray,
    candidate_y: float,
    law: LocalGraphLaw,
    target_slot_position: int,
    synthetic_treatments: np.ndarray,
    alpha: float,
    tie_tolerance: float,
) -> FactorizingCandidateResult:
    # Permutation-commuting transform: m=0, s=1. Thus candidate y is residual E_t(y).
    m3 = evaluate_candidate(
        calibration_residual=np.asarray(calibration_residual, dtype=float),
        candidate_y=float(candidate_y),
        target_mean=0.0,
        law=law,
        target_slot_position=int(target_slot_position),
        target_scale=1.0,
        tie_tolerance=float(tie_tolerance),
        keep_states=False,
    )
    residual_payload = candidate_residuals(calibration_residual, candidate_y, 0.0, 1.0)
    if len(set(numeric_key(x) for x in residual_payload)) != len(residual_payload):
        raise D3M4Error("Factorizing reference fixture requires unique residual payloads")
    a = np.asarray(synthetic_treatments, dtype=float)
    if a.shape != residual_payload.shape:
        raise D3M4Error("Factorizing treatment vector shape mismatch")

    # q_h = g => d_j = 1. M4 is therefore exactly the M3 source marginal after one product normalization.
    m4 = evaluate_m4(
        m3.source_marginals,
        residual_payload,
        np.zeros(len(residual_payload), dtype=float),
        target_source_index=len(residual_payload) - 1,
        tie_tolerance=tie_tolerance,
    )

    # Independent labeled joint-payload enumeration. Treatments move jointly with residuals.
    # Slot-homogeneous g and q=g make the entire treatment product orbit-constant.
    logg = np.log(_homogeneous_g(a))
    state_sources: list[int] = []
    log_total: list[float] = []
    log_treat: list[float] = []
    log_residual: list[float] = []
    scores: list[float] = []
    tpos = int(target_slot_position)
    for perm in permutations(range(len(residual_payload))):
        assigned_e = residual_payload[np.asarray(perm, dtype=int)]
        residual_log, singular = local_log_factor(assigned_e, law)
        if singular:
            raise D3M4Error("Unexpected singular factorizing state")
        # Every source appears once and q=g at the target, so this sum is constant under permutation.
        treatment_log = float(logg[np.asarray(perm, dtype=int)].sum())
        jacobian_log = 0.0  # s_i(a)=1 in the synthetic reduction fixture.
        log_total.append(treatment_log + jacobian_log + residual_log)
        log_treat.append(treatment_log)
        log_residual.append(float(residual_log))
        state_sources.append(int(perm[tpos]))
        scores.append(abs(float(assigned_e[tpos])))
    probs = normalize_logweights(np.asarray(log_total, dtype=float))
    q6 = np.zeros(len(residual_payload), dtype=float)
    for src, prob in zip(state_sources, probs):
        q6[src] += float(prob)
    identity_score = abs(float(residual_payload[-1]))
    p6 = float(np.asarray(probs)[np.asarray(scores) + tie_tolerance >= identity_score].sum())
    q3_q4 = float(np.max(np.abs(m3.source_marginals - m4.q4)))
    q3_q6 = float(np.max(np.abs(m3.source_marginals - q6)))
    pmax = float(max(abs(m3.pvalue - m4.pvalue), abs(m3.pvalue - p6), abs(m4.pvalue - p6)))
    return FactorizingCandidateResult(
        candidate_y=float(candidate_y),
        q3=m3.source_marginals,
        q4=m4.q4,
        q6_reference=q6,
        p3=float(m3.pvalue),
        p4=float(m4.pvalue),
        p6_reference=float(p6),
        treatment_log_factor_range=float(np.max(log_treat) - np.min(log_treat)),
        residual_log_factor_range=float(np.max(log_residual) - np.min(log_residual)),
        q3_q4_max_abs=q3_q4,
        q3_q6_max_abs=q3_q6,
        p_max_abs=pmax,
        accepted3=bool(m3.pvalue > alpha),
        accepted4=bool(m4.pvalue > alpha),
        accepted6=bool(p6 > alpha),
    )
