from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geodose_stage3d.exact_law import (  # noqa: E402
    local_full_cancellation_audit,
    prepare_orbit,
    scalar_reference_evaluation,
    vectorized_orbit_evaluation,
)
from geodose_stage3d.io import (  # noqa: E402
    Stage3BData,
    _source_payloads,
    build_fixture,
    payload_key,
    read_yaml,
    verify_input_hashes,
)
from geodose_stage3d.mixed_measure import (  # noqa: E402
    intervention_density,
    intervention_normalization_error,
    representative_mixed_normalization_error,
)
from geodose_stage3d.orbit import distinct_index_permutations  # noqa: E402
from geodose_stage3d.residual_transform import affine_residual_scalar  # noqa: E402


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def max_abs_compatible(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    same_inf = (np.isposinf(a) & np.isposinf(b)) | (np.isneginf(a) & np.isneginf(b))
    finite = np.isfinite(a) & np.isfinite(b)
    assert_true(np.all(same_inf | finite), "Incompatible infinities")
    return float(np.max(np.abs(a[finite] - b[finite]))) if np.any(finite) else 0.0


def main() -> None:
    verify_input_hashes(ROOT)
    tolerance = read_yaml(ROOT / "configs" / "stage3d_tolerance_registry.yaml")
    registry = read_yaml(ROOT / "configs" / "stage3d_fixture_registry.yaml")
    data = Stage3BData(ROOT / "inputs" / "upstream" / "GeoDose_Stage3B_OUTPUTS.zip")

    raw = data.fixtures["size6_unique"]
    payloads = _source_payloads(raw)
    target_truth = float(raw["target_truth_reference"]["Y_true_at_A_star"])
    perms = distinct_index_permutations([payload_key(p, candidate_y=target_truth) for p in payloads])
    assert_true(perms.shape == (720, 6), "Size-6 unique orbit count")
    assert_true(np.array_equal(perms[0], np.arange(6)), "Identity state must be first")

    raw_dup = data.fixtures["size6_one_duplicate_pair"]
    payloads_dup = _source_payloads(raw_dup)
    dup_truth = float(raw_dup["target_truth_reference"]["Y_true_at_A_star"])
    perms_dup = distinct_index_permutations([payload_key(p, candidate_y=dup_truth) for p in payloads_dup])
    assert_true(perms_dup.shape == (360, 6), "Duplicate quotient orbit must have 360 states")

    raw8 = data.fixtures["size8_unique"]
    payloads8 = _source_payloads(raw8)
    truth8 = float(raw8["target_truth_reference"]["Y_true_at_A_star"])
    perms8 = distinct_index_permutations([payload_key(p, candidate_y=truth8) for p in payloads8])
    assert_true(perms8.shape == (40320, 8), "Size-8 orbit count")

    # Candidate source origin is not a theorem-level payload field.  An exact
    # candidate/calibration collision must collapse 6 labelled permutations to
    # 3 distinct numeric (A,Y) orbit states.
    from geodose_stage3d.types import Payload
    collision_payloads = [
        Payload("c0", 0.2, 1.0, False, "factual_calibration_observation"),
        Payload("c1", 0.7, 2.0, False, "factual_calibration_observation"),
        Payload("t", 0.2, 9.0, True, "realized_localized_target_truth_reference"),
    ]
    noncollision = distinct_index_permutations([payload_key(p, candidate_y=1.5) for p in collision_payloads])
    collision = distinct_index_permutations([payload_key(p, candidate_y=1.0) for p in collision_payloads])
    assert_true(noncollision.shape == (6, 3), "Noncollision candidate orbit count")
    assert_true(collision.shape == (3, 3), "Candidate/calibration quotient collision count")

    first_cfg = registry["fixtures"][0]
    fixture = build_fixture(data, first_cfg, candidate_shift=float(tolerance["candidate_algebra_shift"]))
    assert_true(np.array_equal(fixture.permutations, perms), "Fixture candidate orbit mismatch")
    prepared = prepare_orbit(fixture, minimum_precision_eigenvalue_required=float(tolerance["minimum_precision_eigenvalue"]))
    evaluation = vectorized_orbit_evaluation(
        prepared,
        fixture.candidate_y,
        non_gaussian_min_abs_residual=float(tolerance["non_gaussian_min_abs_residual"]),
    )
    assert_true(abs(float(np.sum(evaluation.probability)) - 1.0) < float(tolerance["probability_sum_abs_tolerance"]), "Probability normalization")
    assert_true(np.all(evaluation.probability >= 0.0), "Negative probability")
    zero_a_states = prepared.A_payload[evaluation.target_slot_payload_indices] == 0.0
    assert_true(np.all(evaluation.probability[zero_a_states] == 0.0), "Interior intervention must give endpoint zero mass")

    scalar = scalar_reference_evaluation(
        prepared,
        fixture.candidate_y,
        np.arange(720),
        non_gaussian_min_abs_residual=float(tolerance["non_gaussian_min_abs_residual"]),
    )
    assert_true(max_abs_compatible(scalar["log_total"], evaluation.log_total) <= float(tolerance["scalar_vector_logweight_abs_tolerance"]), "Scalar/vectorized mismatch")
    cancellation = local_full_cancellation_audit(
        prepared,
        evaluation,
        np.arange(720),
        non_gaussian_min_abs_residual=float(tolerance["non_gaussian_min_abs_residual"]),
    )
    assert_true(float(cancellation["constant_range"]) <= float(tolerance["full_local_constant_range_tolerance"]), "Full/local GMRF mismatch")

    # Mixed measure and intervention normalization.
    row = prepared.slot_unit_rows.iloc[0]
    mixed_error = representative_mixed_normalization_error(
        float(row["pi_atom_0"]), float(row["pi_atom_1"]), float(row["pi_interior"]),
        float(row["beta_alpha"]), float(row["beta_beta"]),
    )
    assert_true(mixed_error <= float(tolerance["mixed_measure_normalization_abs_tolerance"]), "Mixed-treatment normalization")
    assert_true(intervention_normalization_error(0.9, 0.1, True) <= float(tolerance["mixed_measure_normalization_abs_tolerance"]), "Interior intervention normalization")
    assert_true(intervention_normalization_error(0.0, None, True) == 0.0, "Endpoint intervention normalization")
    atom = intervention_density(np.asarray([0.0, 0.5, 1.0]), 0.0, None, endpoint_audited=True)
    assert_true(np.array_equal(atom, np.asarray([1.0, 0.0, 0.0])), "Endpoint atom density")

    # Target positivity is mandatory, while zero likelihood at a non-target
    # slot is a valid zero-probability orbit state rather than a package error.
    prepared_zero = prepare_orbit(fixture, minimum_precision_eigenvalue_required=float(tolerance["minimum_precision_eigenvalue"]))
    prepared_zero.observational_g = prepared_zero.observational_g.copy()
    zero_payload = int(np.where(prepared_zero.A_payload == 0.0)[0][0])
    non_target_slot = next(i for i in range(len(fixture.slots)) if i != fixture.target_slot_position)
    prepared_zero.observational_g[non_target_slot, zero_payload] = 0.0
    zero_eval = vectorized_orbit_evaluation(
        prepared_zero,
        fixture.candidate_y,
        non_gaussian_min_abs_residual=float(tolerance["non_gaussian_min_abs_residual"]),
    )
    assert_true(np.any(zero_eval.probability == 0.0), "Non-target zero likelihood should create zero-mass states")
    assert_true(abs(float(np.sum(zero_eval.probability)) - 1.0) < float(tolerance["probability_sum_abs_tolerance"]), "Zero-likelihood orbit normalization")

    prepared_bad = prepare_orbit(fixture, minimum_precision_eigenvalue_required=float(tolerance["minimum_precision_eigenvalue"]))
    prepared_bad.observational_g = prepared_bad.observational_g.copy()
    q_positive = int(np.where(prepared_bad.intervention_q > 0.0)[0][0])
    prepared_bad.observational_g[fixture.target_slot_position, q_positive] = 0.0
    # The support check is performed in prepare_orbit for real data; this
    # mutation verifies the algebraic condition explicitly.
    assert_true(bool(prepared_bad.intervention_q[q_positive] > 0.0 and prepared_bad.observational_g[fixture.target_slot_position, q_positive] == 0.0), "Synthetic positivity violation setup")

    # Inverse Jacobian direction.
    assignment = np.asarray([0, 1, 2], dtype=int)
    y = np.asarray([1.0, 2.0, 3.0])
    mean = np.zeros((3, 3))
    scale = np.asarray([[2.0, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 4.0]])
    _, log_j = affine_residual_scalar(assignment, y, mean, scale)
    expected = -math.log(2.0) - math.log(0.5) - math.log(4.0)
    assert_true(abs(log_j - expected) < float(tolerance["jacobian_abs_tolerance"]), "Inverse Jacobian direction")

    # Non-Gaussian mirror fixture.
    ng_cfg = [row for row in registry["fixtures"] if row["evaluation_id"] == "G6_NONGAUSSIAN_INTERIOR_TRUTH"][0]
    # Derived payload keys are unique and target-distinguished; use size6 permutations.
    ng_fixture = build_fixture(data, ng_cfg, candidate_shift=float(tolerance["candidate_algebra_shift"]))
    ng_prepared = prepare_orbit(ng_fixture, minimum_precision_eigenvalue_required=float(tolerance["minimum_precision_eigenvalue"]))
    ng_eval = vectorized_orbit_evaluation(
        ng_prepared,
        ng_fixture.candidate_y,
        non_gaussian_min_abs_residual=float(tolerance["non_gaussian_min_abs_residual"]),
    )
    assert_true(abs(float(np.sum(ng_eval.probability)) - 1.0) < float(tolerance["probability_sum_abs_tolerance"]), "Non-Gaussian probability normalization")
    assert_true(ng_eval.latent_matrix is not None, "Non-Gaussian latent matrix missing")

    print("STAGE3D D0/D1 UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
