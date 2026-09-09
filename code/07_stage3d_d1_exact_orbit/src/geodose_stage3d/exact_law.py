from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
from scipy.special import logsumexp

from .graph_precision import (
    block_is_connected,
    gaussian_full_log_factor,
    gaussian_local_log_factor,
    minimum_precision_eigenvalue,
    normalized_precision,
    precision_blocks,
)
from .io import Stage3DError, payload_key, require
from .orbit import distinct_index_permutations, validate_exact_block_size
from .mixed_measure import intervention_density, observational_mixed_density
from .residual_transform import (
    affine_residual_matrix,
    affine_residual_scalar,
    conditional_mean_matrix,
    inverse_power_transform,
)
from .types import ExactFixture, OrbitEvaluation, PreparedOrbit


def _log_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    require(np.all(np.isfinite(values)), "D1_TREATMENT_LIKELIHOOD_INVALID: nonfinite density")
    require(np.all(values >= 0.0), "D1_TREATMENT_LIKELIHOOD_INVALID: negative density")
    result = np.full(values.shape, -np.inf, dtype=float)
    positive = values > 0.0
    result[positive] = np.log(values[positive])
    return result


def prepare_orbit(fixture: ExactFixture, *, minimum_precision_eigenvalue_required: float) -> PreparedOrbit:
    validate_exact_block_size(len(fixture.slots), maximum=8)
    require(block_is_connected(fixture.block_nodes, fixture.case_edges), "D1_BLOCK_DISCONNECTED")
    unit_by_node = fixture.case_units.set_index("node_index", drop=False)
    slot_rows = unit_by_node.loc[fixture.block_nodes].reset_index(drop=True)
    require(len(slot_rows) == len(fixture.slots), "Slot-unit lookup mismatch")
    require(np.array_equal(slot_rows["node_index"].to_numpy(dtype=int), fixture.block_nodes), "Slot order changed")

    A_payload = np.asarray([payload.A for payload in fixture.payloads], dtype=float)
    Y_payload = np.asarray([payload.Y_reference for payload in fixture.payloads], dtype=float)
    target_indices = [i for i, payload in enumerate(fixture.payloads) if payload.is_target_payload]
    require(len(target_indices) == 1, "D1_TARGET_SLOT_INVALID: target payload count")
    target_payload_index = target_indices[0]

    baseline = slot_rows["outcome_baseline"].to_numpy(dtype=float)
    x1 = slot_rows["X_spatial_1"].to_numpy(dtype=float)
    outcome_mean = conditional_mean_matrix(str(fixture.case_registry_row["outcome_form"]), baseline, x1, A_payload)
    outcome_scale = np.ones_like(outcome_mean, dtype=float)

    b = len(fixture.slots)
    observational_g = np.empty((b, b), dtype=float)
    for slot in range(b):
        row = slot_rows.iloc[slot]
        observational_g[slot] = observational_mixed_density(
            A_payload,
            np.full(b, float(row["pi_atom_0"])),
            np.full(b, float(row["pi_atom_1"])),
            np.full(b, float(row["pi_interior"])),
            np.full(b, float(row["beta_alpha"])),
            np.full(b, float(row["beta_beta"])),
        )
    require(np.all(np.isfinite(observational_g)) and np.all(observational_g >= 0.0),
            "D1_TREATMENT_LIKELIHOOD_INVALID: observational density negative or nonfinite")

    intervention_q = intervention_density(
        A_payload,
        fixture.target_dose,
        fixture.bandwidth,
        endpoint_audited=fixture.endpoint_audited,
    )
    require(np.all(np.isfinite(intervention_q)) and np.all(intervention_q >= 0.0),
            "D1_TREATMENT_LIKELIHOOD_INVALID: intervention density negative or nonfinite")
    require(np.any(intervention_q > 0.0), "D1_ALL_ORBIT_WEIGHTS_ZERO: intervention has no payload support")
    target_observational_g = observational_g[fixture.target_slot_position]
    positivity_violation = (intervention_q > 0.0) & (target_observational_g <= 0.0)
    require(not np.any(positivity_violation),
            "D1_POSITIVITY_FAILURE: target intervention has mass where target-slot observational law is zero")

    # The finite-orbit check above is necessary for every realized payload.
    # The causal/intervention contract also requires positivity over the full
    # support of q, not merely at the observed orbit atoms.  For the frozen
    # Stage 3B mixed law this condition is analytic: a truncated-Gaussian
    # interior intervention is supported on (0,1), while endpoint interventions
    # are point masses at 0 or 1.
    target_row = slot_rows.iloc[fixture.target_slot_position]
    if fixture.target_dose == 0.0:
        require(float(target_row["pi_atom_0"]) > 0.0,
                "D1_POSITIVITY_FAILURE: target-slot law has no mass at endpoint 0")
    elif fixture.target_dose == 1.0:
        require(float(target_row["pi_atom_1"]) > 0.0,
                "D1_POSITIVITY_FAILURE: target-slot law has no mass at endpoint 1")
    else:
        require(float(target_row["pi_interior"]) > 0.0,
                "D1_POSITIVITY_FAILURE: target-slot law has no interior mass")
        require(float(target_row["beta_alpha"]) > 0.0 and float(target_row["beta_beta"]) > 0.0,
                "D1_POSITIVITY_FAILURE: target-slot interior density is not positive on (0,1)")

    n_nodes = len(fixture.case_units)
    Q = normalized_precision(n_nodes, fixture.case_edges, fixture.rho, fixture.residual_scale)
    min_eig = minimum_precision_eigenvalue(Q)
    require(min_eig > minimum_precision_eigenvalue_required, f"D1_PRECISION_NOT_PD: {min_eig}")
    Q_BB, Q_BD = precision_blocks(Q, fixture.block_nodes, fixture.boundary_nodes)

    outside_residual = fixture.case_units["shared_spatial_residual"].to_numpy(dtype=float)
    outside_latent = fixture.case_units["residual_latent_gaussian"].to_numpy(dtype=float)
    require(np.all(np.isfinite(outside_residual)) and np.all(np.isfinite(outside_latent)), "Boundary residual unavailable")
    boundary_residual = outside_residual[fixture.boundary_nodes]
    boundary_latent = outside_latent[fixture.boundary_nodes]

    return PreparedOrbit(
        fixture=fixture,
        slot_unit_rows=slot_rows,
        A_payload=A_payload,
        Y_payload_reference=Y_payload,
        target_payload_index=target_payload_index,
        outcome_mean=outcome_mean,
        outcome_scale=outcome_scale,
        observational_g=observational_g,
        intervention_q=intervention_q,
        Q=Q,
        Q_BB=Q_BB,
        Q_BD=Q_BD,
        outside_residual=outside_residual,
        outside_latent_z=outside_latent,
        boundary_residual=boundary_residual,
        boundary_latent_z=boundary_latent,
    )


def _candidate_payload_y(prepared: PreparedOrbit, candidate_y: float) -> np.ndarray:
    y = prepared.Y_payload_reference.copy()
    y[prepared.target_payload_index] = float(candidate_y)
    return y


def candidate_permutations(prepared: PreparedOrbit, candidate_y: float) -> np.ndarray:
    """Construct the distinct G1 quotient orbit after candidate replacement.

    The source index of the augmented target pair is metadata for replacement
    only.  It is not part of payload equality; equality is exactly the pair
    ``(A,Y_candidate)``.  Therefore the quotient can change at candidate values
    that collide with a calibration payload.
    """
    keys = [payload_key(payload, candidate_y=float(candidate_y)) for payload in prepared.fixture.payloads]
    return distinct_index_permutations(keys)


def target_support_positivity_audit(prepared: PreparedOrbit) -> dict[str, float | str | bool]:
    """Report analytic positivity over the complete intervention support."""
    fixture = prepared.fixture
    row = prepared.slot_unit_rows.iloc[fixture.target_slot_position]
    if fixture.target_dose == 0.0:
        support = "atom_0"
        observational_support_mass = float(row["pi_atom_0"])
        satisfied = observational_support_mass > 0.0
    elif fixture.target_dose == 1.0:
        support = "atom_1"
        observational_support_mass = float(row["pi_atom_1"])
        satisfied = observational_support_mass > 0.0
    else:
        support = "open_interval_0_1"
        observational_support_mass = float(row["pi_interior"])
        satisfied = (
            observational_support_mass > 0.0
            and float(row["beta_alpha"]) > 0.0
            and float(row["beta_beta"]) > 0.0
        )
    return {
        "intervention_support": support,
        "target_slot_pi_atom_0": float(row["pi_atom_0"]),
        "target_slot_pi_atom_1": float(row["pi_atom_1"]),
        "target_slot_pi_interior": float(row["pi_interior"]),
        "target_slot_beta_alpha": float(row["beta_alpha"]),
        "target_slot_beta_beta": float(row["beta_beta"]),
        "observational_mass_on_intervention_component": observational_support_mass,
        "complete_support_positivity_satisfied": bool(satisfied),
    }


def vectorized_orbit_evaluation(
    prepared: PreparedOrbit,
    candidate_y: float,
    *,
    non_gaussian_min_abs_residual: float,
) -> OrbitEvaluation:
    fixture = prepared.fixture
    permutations = candidate_permutations(prepared, candidate_y)
    b = permutations.shape[1]
    rows = np.arange(b)[None, :]
    Y_payload = _candidate_payload_y(prepared, candidate_y)

    residual, log_j_matrix = affine_residual_matrix(
        permutations,
        Y_payload,
        prepared.outcome_mean,
        prepared.outcome_scale,
    )
    log_outcome_jacobian = np.sum(log_j_matrix, axis=1)

    treatment_matrix = prepared.observational_g[rows, permutations].copy()
    treatment_matrix[:, fixture.target_slot_position] = prepared.intervention_q[permutations[:, fixture.target_slot_position]]
    log_treatment = np.sum(_log_nonnegative(treatment_matrix), axis=1)

    latent_matrix: np.ndarray | None = None
    if fixture.residual_law == "transformed_gmrf_power_1_5":
        latent_matrix, log_abs_dz_dr = inverse_power_transform(
            residual,
            fixture.transform_power,
            minimum_abs_residual=non_gaussian_min_abs_residual,
        )
        log_residual = gaussian_local_log_factor(
            latent_matrix,
            prepared.Q_BB,
            prepared.Q_BD,
            prepared.boundary_latent_z,
        ) + np.sum(log_abs_dz_dr, axis=1)
    elif fixture.residual_law == "gaussian_gmrf":
        log_residual = gaussian_local_log_factor(
            residual,
            prepared.Q_BB,
            prepared.Q_BD,
            prepared.boundary_residual,
        )
    else:
        raise Stage3DError(f"D1 residual law is unsupported: {fixture.residual_law}")

    log_total = log_treatment + log_outcome_jacobian + log_residual
    require(not np.any(np.isnan(log_total)), "D1_NONFINITE_ORBIT_WEIGHT: NaN log weight")
    finite = np.isfinite(log_total)
    require(np.any(finite), "D1_ALL_ORBIT_WEIGHTS_ZERO")
    normalizer = float(logsumexp(log_total[finite]))
    probability = np.zeros(len(log_total), dtype=float)
    probability[finite] = np.exp(log_total[finite] - normalizer)
    require(np.all(np.isfinite(probability)) and np.all(probability >= 0.0), "Invalid normalized probability")

    target_payload_positions = np.argmax(permutations == prepared.target_payload_index, axis=1).astype(int)
    target_slot_payload_indices = permutations[:, fixture.target_slot_position].astype(int)
    return OrbitEvaluation(
        evaluation_id=fixture.evaluation_id,
        candidate_y=float(candidate_y),
        permutations=permutations,
        log_treatment=log_treatment,
        log_outcome_jacobian=log_outcome_jacobian,
        log_residual=np.asarray(log_residual, dtype=float),
        log_total=log_total,
        probability=probability,
        residual_matrix=residual,
        latent_matrix=latent_matrix,
        target_payload_positions=target_payload_positions,
        target_slot_payload_indices=target_slot_payload_indices,
        diagnostics={
            "log_normalizer": normalizer,
            "positive_probability_states": int(np.count_nonzero(probability > 0.0)),
            "zero_probability_states": int(np.count_nonzero(probability == 0.0)),
            "probability_sum": float(np.sum(probability)),
            "minimum_precision_eigenvalue": minimum_precision_eigenvalue(prepared.Q),
        },
    )


def _scalar_gaussian_local_log_factor(
    residual_B: np.ndarray,
    Q_BB: np.ndarray,
    Q_BD: np.ndarray,
    residual_D: np.ndarray,
) -> float:
    """Loop-based reference calculation independent of the vectorized helper."""
    r = np.asarray(residual_B, dtype=float)
    value = 0.0
    for i in range(len(r)):
        for j in range(len(r)):
            value += -0.5 * float(r[i]) * float(Q_BB[i, j]) * float(r[j])
    for i in range(len(r)):
        for d in range(Q_BD.shape[1]):
            value += -float(r[i]) * float(Q_BD[i, d]) * float(residual_D[d])
    return float(value)


def _scalar_inverse_power_transform(
    residual: np.ndarray,
    power: float,
    *,
    minimum_abs_residual: float,
) -> tuple[np.ndarray, float]:
    """Scalar-loop inverse transform and log Jacobian for audit independence."""
    require(power >= 1.0, "Transform power must be at least one")
    latent = np.empty(len(residual), dtype=float)
    log_j = 0.0
    for index, raw_value in enumerate(np.asarray(residual, dtype=float).tolist()):
        value = float(raw_value)
        if abs(power - 1.0) < 1e-15:
            latent[index] = value
            continue
        absolute = abs(value)
        if absolute < minimum_abs_residual:
            raise Stage3DError("D1_TRANSFORM_SINGULAR: transformed-GMRF residual is numerically zero")
        latent[index] = math.copysign(absolute ** (1.0 / power), value)
        log_j += -math.log(power) + (1.0 / power - 1.0) * math.log(absolute)
    return latent, float(log_j)


def scalar_state_factors(
    prepared: PreparedOrbit,
    assignment: np.ndarray,
    candidate_y: float,
    *,
    non_gaussian_min_abs_residual: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray | None]:
    fixture = prepared.fixture
    Y_payload = _candidate_payload_y(prepared, candidate_y)
    log_treatment = 0.0
    for slot, payload in enumerate(np.asarray(assignment, dtype=int).tolist()):
        density = (
            float(prepared.intervention_q[payload])
            if slot == fixture.target_slot_position
            else float(prepared.observational_g[slot, payload])
        )
        if density == 0.0:
            log_treatment = -np.inf
            break
        require(density > 0.0 and np.isfinite(density), "D1_TREATMENT_LIKELIHOOD_INVALID")
        log_treatment += math.log(density)

    residual, log_outcome_jacobian = affine_residual_scalar(
        assignment,
        Y_payload,
        prepared.outcome_mean,
        prepared.outcome_scale,
    )
    latent: np.ndarray | None = None
    if fixture.residual_law == "transformed_gmrf_power_1_5":
        latent, log_abs_dz_dr_sum = _scalar_inverse_power_transform(
            residual,
            fixture.transform_power,
            minimum_abs_residual=non_gaussian_min_abs_residual,
        )
        log_residual = _scalar_gaussian_local_log_factor(
            latent, prepared.Q_BB, prepared.Q_BD, prepared.boundary_latent_z
        ) + log_abs_dz_dr_sum
    elif fixture.residual_law == "gaussian_gmrf":
        log_residual = _scalar_gaussian_local_log_factor(
            residual, prepared.Q_BB, prepared.Q_BD, prepared.boundary_residual
        )
    else:
        raise Stage3DError(f"Unsupported residual law: {fixture.residual_law}")
    return float(log_treatment), float(log_outcome_jacobian), log_residual, residual, latent


def scalar_reference_evaluation(
    prepared: PreparedOrbit,
    candidate_y: float,
    state_indices: np.ndarray,
    *,
    non_gaussian_min_abs_residual: float,
) -> dict[str, np.ndarray]:
    permutations = candidate_permutations(prepared, candidate_y)
    state_indices = np.asarray(state_indices, dtype=int)
    require(np.all((state_indices >= 0) & (state_indices < len(permutations))), "Scalar state index outside candidate orbit")
    log_treatment = np.empty(len(state_indices), dtype=float)
    log_j = np.empty(len(state_indices), dtype=float)
    log_r = np.empty(len(state_indices), dtype=float)
    for out_index, state_index in enumerate(state_indices.tolist()):
        values = scalar_state_factors(
            prepared,
            permutations[state_index],
            candidate_y,
            non_gaussian_min_abs_residual=non_gaussian_min_abs_residual,
        )
        log_treatment[out_index], log_j[out_index], log_r[out_index] = values[:3]
    return {
        "state_index": state_indices,
        "log_treatment": log_treatment,
        "log_outcome_jacobian": log_j,
        "log_residual": log_r,
        "log_total": log_treatment + log_j + log_r,
    }


def full_log_density_for_state(
    prepared: PreparedOrbit,
    assignment: np.ndarray,
    candidate_y: float,
    *,
    non_gaussian_min_abs_residual: float,
) -> float:
    _, _, _, residual_B, latent_B = scalar_state_factors(
        prepared,
        assignment,
        candidate_y,
        non_gaussian_min_abs_residual=non_gaussian_min_abs_residual,
    )
    fixture = prepared.fixture
    if fixture.residual_law == "gaussian_gmrf":
        full = prepared.outside_residual.copy()
        full[fixture.block_nodes] = residual_B
        return gaussian_full_log_factor(full, prepared.Q)
    if fixture.residual_law == "transformed_gmrf_power_1_5":
        require(latent_B is not None, "Missing transformed latent residual")
        full_residual = prepared.outside_residual.copy()
        full_residual[fixture.block_nodes] = residual_B
        full_latent = prepared.outside_latent_z.copy()
        full_latent[fixture.block_nodes] = latent_B
        _, log_abs_dz_dr = inverse_power_transform(
            full_residual,
            fixture.transform_power,
            minimum_abs_residual=non_gaussian_min_abs_residual,
        )
        return gaussian_full_log_factor(full_latent, prepared.Q) + float(np.sum(log_abs_dz_dr))
    raise Stage3DError(f"Unsupported residual law: {fixture.residual_law}")


def local_full_cancellation_audit(
    prepared: PreparedOrbit,
    evaluation: OrbitEvaluation,
    state_indices: np.ndarray,
    *,
    non_gaussian_min_abs_residual: float,
) -> dict[str, object]:
    state_indices = np.asarray(state_indices, dtype=int)
    full = np.asarray([
        full_log_density_for_state(
            prepared,
            evaluation.permutations[index],
            evaluation.candidate_y,
            non_gaussian_min_abs_residual=non_gaussian_min_abs_residual,
        )
        for index in state_indices.tolist()
    ])
    local = evaluation.log_residual[state_indices]
    difference = full - local
    return {
        "state_indices": state_indices,
        "full_log_residual": full,
        "local_log_residual": local,
        "full_minus_local": difference,
        "constant_range": float(np.max(difference) - np.min(difference)),
        "constant_mean": float(np.mean(difference)),
    }

def full_target_law_probability_audit(
    prepared: PreparedOrbit,
    evaluation: OrbitEvaluation,
    *,
    non_gaussian_min_abs_residual: float,
    maximum_states: int = 720,
) -> dict[str, float | int]:
    """Compare local G2 probabilities with direct full-joint target-law weights.

    This audit is intentionally restricted to small fixtures.  It evaluates the
    complete residual density on the full graph for every candidate-specific
    orbit state, combines it with the exact treatment and outcome-Jacobian
    factors, and normalizes independently.  Orbit-invariant far-field terms
    must cancel, so the probabilities must equal the graph-local result.
    """
    require(len(evaluation.permutations) <= int(maximum_states),
            "Full target-law audit exceeds frozen small-fixture limit")
    full_log_total = np.empty(len(evaluation.permutations), dtype=float)
    for index, assignment in enumerate(evaluation.permutations):
        # Recompute treatment and outcome Jacobian through the scalar reference
        # path as well as the full-graph residual density.  This prevents the
        # full-target audit from inheriting the vectorized factors it is meant
        # to verify.
        scalar_treatment, scalar_jacobian, _, _, _ = scalar_state_factors(
            prepared,
            assignment,
            evaluation.candidate_y,
            non_gaussian_min_abs_residual=non_gaussian_min_abs_residual,
        )
        full_residual = full_log_density_for_state(
            prepared,
            assignment,
            evaluation.candidate_y,
            non_gaussian_min_abs_residual=non_gaussian_min_abs_residual,
        )
        full_log_total[index] = float(scalar_treatment + scalar_jacobian + full_residual)
    finite = np.isfinite(full_log_total)
    require(np.any(finite), "D1_ALL_ORBIT_WEIGHTS_ZERO: full target-law audit")
    full_probability = np.zeros(len(full_log_total), dtype=float)
    full_probability[finite] = np.exp(full_log_total[finite] - float(logsumexp(full_log_total[finite])))
    local_finite = np.isfinite(evaluation.log_total)
    common = finite & local_finite
    difference = full_log_total[common] - evaluation.log_total[common]
    return {
        "state_count": int(len(full_log_total)),
        "maximum_absolute_probability_difference": float(np.max(np.abs(full_probability - evaluation.probability))),
        "full_minus_local_constant_range": float(np.max(difference) - np.min(difference)),
        "full_probability_sum": float(np.sum(full_probability)),
    }

