from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .audits import boundary_audit_frame, state_audit_frame, write_csv, write_csv_gz
from .exact_law import (
    full_target_law_probability_audit,
    local_full_cancellation_audit,
    prepare_orbit,
    scalar_reference_evaluation,
    target_support_positivity_audit,
    vectorized_orbit_evaluation,
)
from .io import (
    Stage3BData,
    build_fixture,
    read_yaml,
    require,
    verify_input_hashes,
)
from .mixed_measure import intervention_normalization_error, representative_mixed_normalization_error
from .provenance import environment_inventory, package_file_hashes, sha256_file, tree_hash, write_json
from .residual_transform import affine_residual_scalar


def _max_abs_with_infinities(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    require(left.shape == right.shape, "Array shape mismatch")
    same_pos_inf = np.isposinf(left) & np.isposinf(right)
    same_neg_inf = np.isneginf(left) & np.isneginf(right)
    finite = np.isfinite(left) & np.isfinite(right)
    compatible = finite | same_pos_inf | same_neg_inf
    require(np.all(compatible), "Scalar/vectorized infinity mismatch")
    return float(np.max(np.abs(left[finite] - right[finite]))) if np.any(finite) else 0.0


def _state_indices_for_reference(n_states: int, size8_count: int) -> np.ndarray:
    if n_states <= 720:
        return np.arange(n_states, dtype=int)
    count = min(int(size8_count), n_states)
    indices = np.linspace(0, n_states - 1, count, dtype=int)
    return np.unique(indices)


def _source_hashes(package_root: Path) -> dict[str, str]:
    include_roots = [
        package_root / "configs",
        package_root / "docs",
        package_root / "src",
        package_root / "tests",
    ]
    include_files = [
        package_root / "README.md",
        package_root / "CHANGELOG.md",
        package_root / "requirements_py310.txt",
        package_root / "RUN_STAGE3D_D1.bat",
        package_root / "RUN_STAGE3D_D1.ps1",
        package_root / "stage3d_d1_run.py",
        package_root / "verify_stage3d_d1.py",
        package_root / "package_stage3d_outputs.py",
    ]
    result: dict[str, str] = {}
    for root in include_roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix not in {".pyc", ".pyo"} and "__pycache__" not in path.parts:
                result[path.relative_to(package_root).as_posix()] = sha256_file(path)
    for path in include_files:
        require(path.is_file(), f"Missing execution-affecting file: {path}")
        result[path.relative_to(package_root).as_posix()] = sha256_file(path)
    return dict(sorted(result.items()))


def run_d1(package_root: str | Path, output_dir: str | Path, *, overwrite: bool) -> dict[str, Any]:
    package_root = Path(package_root).resolve()
    output_dir = Path(output_dir).resolve()
    require(package_root in output_dir.parents or output_dir.parent == package_root, "Output directory must be package-local")
    if output_dir.exists():
        if not overwrite:
            require(not any(output_dir.iterdir()), "Output directory is nonempty; use --overwrite")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    input_audit = verify_input_hashes(package_root)
    tolerances = read_yaml(package_root / "configs" / "stage3d_tolerance_registry.yaml")
    fixture_registry = read_yaml(package_root / "configs" / "stage3d_fixture_registry.yaml")
    math_contract = read_yaml(package_root / "configs" / "stage3d_math_contract.yaml")
    orbit_contract = read_yaml(package_root / "configs" / "stage3d_orbit_contract.yaml")
    treatment_contract = read_yaml(package_root / "configs" / "stage3d_treatment_measure_contract.yaml")
    residual_contract = read_yaml(package_root / "configs" / "stage3d_residual_law_contract.yaml")
    nuisance_contract = read_yaml(package_root / "configs" / "stage3d_nuisance_contract.yaml")

    require(math_contract["release_scope"] == "D0_D1_exact_orbit_algebra", "D1_CONTRACT_MISMATCH")
    require(math_contract["not_implemented"] and "G3_candidate_p_values" in math_contract["not_implemented"], "D1 scope boundary missing")
    require(int(orbit_contract["maximum_exact_block_size"]) == 8, "Exact block cap mismatch")
    require(treatment_contract["endpoint_jitter"] == "prohibited", "Endpoint jitter contract mismatch")
    require(residual_contract["gaussian_GMRF"]["residual_scale"] == 0.35, "Residual scale contract mismatch")
    require(nuisance_contract["D1_regime"] == "oracle_generator_truth_algebra_only", "D1 nuisance regime mismatch")

    stage3b_path = package_root / "inputs" / "upstream" / "GeoDose_Stage3B_OUTPUTS.zip"
    data = Stage3BData(stage3b_path)

    state_frames: list[pd.DataFrame] = []
    boundary_frames: list[pd.DataFrame] = []
    probability_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    cancellation_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    fixture_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    treatment_spot_rows: list[dict[str, Any]] = []
    target_tilt_rows: list[dict[str, Any]] = []
    global_positivity_rows: list[dict[str, Any]] = []
    full_target_law_rows: list[dict[str, Any]] = []
    full_local_detail_rows: list[dict[str, Any]] = []
    refusal_rows: list[dict[str, Any]] = []
    prepared_by_id = {}
    evaluation_by_id = {}

    for evaluation_cfg in fixture_registry["fixtures"]:
        eval_started = time.perf_counter()
        fixture = build_fixture(
            data,
            evaluation_cfg,
            candidate_shift=float(tolerances["candidate_algebra_shift"]),
        )
        require(len(fixture.slots) <= int(orbit_contract["maximum_exact_block_size"]), "D1_BLOCK_TOO_LARGE")
        prepared = prepare_orbit(
            fixture,
            minimum_precision_eigenvalue_required=float(tolerances["minimum_precision_eigenvalue"]),
        )
        support_audit = target_support_positivity_audit(prepared)
        require(bool(support_audit["complete_support_positivity_satisfied"]), "D1_POSITIVITY_FAILURE")
        global_positivity_rows.append({
            "evaluation_id": fixture.evaluation_id,
            "target_dose": fixture.target_dose,
            **support_audit,
        })
        result = vectorized_orbit_evaluation(
            prepared,
            fixture.candidate_y,
            non_gaussian_min_abs_residual=float(tolerances["non_gaussian_min_abs_residual"]),
        )
        permutations = result.permutations
        scalar_state_indices = _state_indices_for_reference(len(permutations), int(tolerances["size8_scalar_audit_states"]))
        full_local_state_indices = _state_indices_for_reference(len(permutations), int(tolerances["size8_full_local_audit_states"]))
        scalar = scalar_reference_evaluation(
            prepared,
            fixture.candidate_y,
            scalar_state_indices,
            non_gaussian_min_abs_residual=float(tolerances["non_gaussian_min_abs_residual"]),
        )
        max_treatment = _max_abs_with_infinities(scalar["log_treatment"], result.log_treatment[scalar_state_indices])
        max_jacobian = _max_abs_with_infinities(scalar["log_outcome_jacobian"], result.log_outcome_jacobian[scalar_state_indices])
        max_residual = _max_abs_with_infinities(scalar["log_residual"], result.log_residual[scalar_state_indices])
        max_total = _max_abs_with_infinities(scalar["log_total"], result.log_total[scalar_state_indices])
        require(max_total <= float(tolerances["scalar_vector_logweight_abs_tolerance"]), "D1_SCALAR_VECTORIZED_MISMATCH")

        cancellation = local_full_cancellation_audit(
            prepared,
            result,
            full_local_state_indices,
            non_gaussian_min_abs_residual=float(tolerances["non_gaussian_min_abs_residual"]),
        )
        require(float(cancellation["constant_range"]) <= float(tolerances["full_local_constant_range_tolerance"]), "D1_LOCAL_GLOBAL_MISMATCH")
        if len(permutations) <= int(tolerances["full_target_law_max_states"]):
            full_target = full_target_law_probability_audit(
                prepared,
                result,
                non_gaussian_min_abs_residual=float(tolerances["non_gaussian_min_abs_residual"]),
                maximum_states=int(tolerances["full_target_law_max_states"]),
            )
            require(
                float(full_target["maximum_absolute_probability_difference"])
                <= float(tolerances["full_target_probability_abs_tolerance"]),
                "D1_FULL_TARGET_LAW_MISMATCH",
            )
            full_target_law_rows.append({"evaluation_id": fixture.evaluation_id, **full_target})
        for audit_index, state_index in enumerate(cancellation["state_indices"].tolist()):
            full_local_detail_rows.append(
                {
                    "evaluation_id": fixture.evaluation_id,
                    "state_index": int(state_index),
                    "full_log_residual": float(cancellation["full_log_residual"][audit_index]),
                    "local_log_residual": float(cancellation["local_log_residual"][audit_index]),
                    "full_minus_local": float(cancellation["full_minus_local"][audit_index]),
                }
            )

        # Archive a small, transparent slotwise treatment-factor trace for
        # independent verification of target cancellation and non-target g.
        spot_state_indices = sorted(set([0, min(1, len(permutations) - 1), len(permutations) - 1]))
        for state_index in spot_state_indices:
            assignment = permutations[state_index]
            for slot_position, payload_index in enumerate(assignment.tolist()):
                if slot_position == fixture.target_slot_position:
                    factor_type = "target_intervention_q"
                    density = float(prepared.intervention_q[payload_index])
                else:
                    factor_type = "observational_g"
                    density = float(prepared.observational_g[slot_position, payload_index])
                treatment_spot_rows.append(
                    {
                        "evaluation_id": fixture.evaluation_id,
                        "state_index": int(state_index),
                        "slot_position": int(slot_position),
                        "payload_index": int(payload_index),
                        "payload_A": float(prepared.A_payload[payload_index]),
                        "factor_type": factor_type,
                        "density_or_mass": density,
                        "log_density_or_mass": -np.inf if density == 0.0 else float(np.log(density)),
                    }
                )
        target_g = prepared.observational_g[fixture.target_slot_position]
        for payload_index, (q_value, g_value) in enumerate(zip(prepared.intervention_q.tolist(), target_g.tolist())):
            if q_value > 0.0:
                require(g_value > 0.0, "D1_POSITIVITY_FAILURE: target tilt q/g undefined")
                ratio = float(q_value / g_value)
                reconstructed = float(ratio * g_value)
                error = abs(reconstructed - float(q_value))
            else:
                ratio = 0.0
                reconstructed = 0.0
                error = 0.0
            target_tilt_rows.append(
                {
                    "evaluation_id": fixture.evaluation_id,
                    "payload_index": int(payload_index),
                    "payload_A": float(prepared.A_payload[payload_index]),
                    "target_intervention_q": float(q_value),
                    "target_observational_g": float(g_value),
                    "target_tilt_ratio_q_over_g": ratio,
                    "reconstructed_q": reconstructed,
                    "absolute_error": error,
                    "positivity_required": bool(q_value > 0.0),
                    "positivity_satisfied": bool(q_value <= 0.0 or g_value > 0.0),
                }
            )
        require(abs(float(np.sum(result.probability)) - 1.0) <= float(tolerances["probability_sum_abs_tolerance"]), "Orbit probability does not sum to one")

        state_frames.append(state_audit_frame(fixture, result))
        boundary_frames.append(boundary_audit_frame(fixture))
        prepared_by_id[fixture.evaluation_id] = prepared
        evaluation_by_id[fixture.evaluation_id] = result

        positive = result.probability > 0.0
        entropy = float(-np.sum(result.probability[positive] * np.log(result.probability[positive])))
        probability_rows.append(
            {
                "evaluation_id": fixture.evaluation_id,
                "case_id": fixture.case_id,
                "residual_law": fixture.residual_law,
                "target_dose": fixture.target_dose,
                "candidate_y": fixture.candidate_y,
                "block_size": len(fixture.slots),
                "distinct_states": len(permutations),
                "positive_probability_states": int(np.count_nonzero(positive)),
                "zero_probability_states": int(np.count_nonzero(~positive)),
                "probability_sum": float(np.sum(result.probability)),
                "minimum_probability_positive": float(np.min(result.probability[positive])),
                "maximum_probability": float(np.max(result.probability)),
                "entropy": entropy,
                "minimum_precision_eigenvalue": float(result.diagnostics["minimum_precision_eigenvalue"]),
            }
        )
        comparison_rows.append(
            {
                "evaluation_id": fixture.evaluation_id,
                "audited_state_count": len(scalar_state_indices),
                "max_abs_log_treatment_difference": max_treatment,
                "max_abs_log_outcome_jacobian_difference": max_jacobian,
                "max_abs_log_residual_difference": max_residual,
                "max_abs_log_total_difference": max_total,
            }
        )
        cancellation_rows.append(
            {
                "evaluation_id": fixture.evaluation_id,
                "audited_state_count": len(full_local_state_indices),
                "full_minus_local_constant_mean": float(cancellation["constant_mean"]),
                "full_minus_local_constant_range": float(cancellation["constant_range"]),
            }
        )

        for slot, row in prepared.slot_unit_rows.iterrows():
            error = representative_mixed_normalization_error(
                float(row["pi_atom_0"]),
                float(row["pi_atom_1"]),
                float(row["pi_interior"]),
                float(row["beta_alpha"]),
                float(row["beta_beta"]),
            )
            normalization_rows.append(
                {
                    "evaluation_id": fixture.evaluation_id,
                    "normalization_object": "observational_mixed_treatment",
                    "slot_position": int(slot),
                    "absolute_error": error,
                }
            )
        normalization_rows.append(
            {
                "evaluation_id": fixture.evaluation_id,
                "normalization_object": "localized_intervention",
                "slot_position": fixture.target_slot_position,
                "absolute_error": intervention_normalization_error(fixture.target_dose, fixture.bandwidth, fixture.endpoint_audited),
            }
        )
        fixture_rows.append(
            {
                "evaluation_id": fixture.evaluation_id,
                "case_id": fixture.case_id,
                "scenario_id": fixture.scenario_id,
                "fixture_source": fixture.fixture_source,
                "target_dose": fixture.target_dose,
                "bandwidth": fixture.bandwidth,
                "candidate_y": fixture.candidate_y,
                "target_truth_reference": fixture.target_payload_truth,
                "block_size": len(fixture.slots),
                "target_slot_position": fixture.target_slot_position,
                "target_slot_node_index": fixture.slots[fixture.target_slot_position].node_index,
                "boundary_node_count": len(fixture.boundary_nodes),
                "internal_edge_count": int(sum(
                    1 for row in fixture.case_edges.itertuples(index=False)
                    if int(row.source_node) in set(fixture.block_nodes.tolist()) and int(row.target_node) in set(fixture.block_nodes.tolist())
                )),
                "boundary_edge_count": len(fixture.boundary_edges),
                "residual_law": fixture.residual_law,
                "rho": fixture.rho,
                "residual_scale": fixture.residual_scale,
                "transform_power": fixture.transform_power,
                "endpoint_audited": fixture.endpoint_audited,
                "distinct_states": len(permutations),
                "nuisance_regime": "oracle_generator_truth_algebra_only",
                "outcome_model_source": "stage3b_true_conditional_mean",
                "treatment_likelihood_source": "stage3b_oracle_mixed_law",
                "spatial_law_source": "stage3b_true_graph_and_registered_precision",
            }
        )
        runtime_rows.append(
            {
                "evaluation_id": fixture.evaluation_id,
                "seconds": time.perf_counter() - eval_started,
                "distinct_states": len(permutations),
                "scalar_audit_states": len(scalar_state_indices),
                "full_local_audit_states": len(full_local_state_indices),
            }
        )

    # Candidate replacement algebra check: same fixture, only target-payload Y differs.
    truth_eval = evaluation_by_id["G6_GAUSS_INTERIOR_TRUTH"]
    shifted_eval = evaluation_by_id["G6_GAUSS_INTERIOR_SHIFTED"]
    require(np.array_equal(truth_eval.permutations, shifted_eval.permutations), "Candidate algebra permutation mismatch")
    candidate_audit = {
        "truth_candidate_y": truth_eval.candidate_y,
        "shifted_candidate_y": shifted_eval.candidate_y,
        "candidate_difference": shifted_eval.candidate_y - truth_eval.candidate_y,
        "max_abs_log_treatment_change": _max_abs_with_infinities(truth_eval.log_treatment, shifted_eval.log_treatment),
        "max_abs_log_outcome_jacobian_change": _max_abs_with_infinities(truth_eval.log_outcome_jacobian, shifted_eval.log_outcome_jacobian),
        "max_abs_log_residual_change": float(np.max(np.abs(truth_eval.log_residual - shifted_eval.log_residual))),
        "target_payload_index": prepared_by_id["G6_GAUSS_INTERIOR_TRUTH"].target_payload_index,
        "rule": "candidate_y_replaces_only_the_augmented_source_response_before_candidate_specific_quotienting",
        "candidate_origin_part_of_payload_equivalence": False,
    }
    require(candidate_audit["max_abs_log_treatment_change"] == 0.0, "Candidate changed treatment factor")
    require(candidate_audit["max_abs_log_outcome_jacobian_change"] == 0.0, "Candidate changed frozen Jacobian factor")
    require(candidate_audit["max_abs_log_residual_change"] > 0.0, "Candidate did not change residual factor")

    # G1 candidate-specific quotient audit.  Target/calibration origin is not a
    # movable-payload field; an exact numerical collision must reduce the orbit.
    from .io import payload_key
    from .orbit import distinct_index_permutations
    from .types import Payload
    collision_payloads = [
        Payload("calib_a", 0.2, 1.0, False, "factual_calibration_observation"),
        Payload("calib_b", 0.7, 2.0, False, "factual_calibration_observation"),
        Payload("candidate", 0.2, 9.0, True, "realized_localized_target_truth_reference"),
    ]
    noncollision_count = len(distinct_index_permutations([
        payload_key(p, candidate_y=1.5) for p in collision_payloads
    ]))
    collision_count = len(distinct_index_permutations([
        payload_key(p, candidate_y=1.0) for p in collision_payloads
    ]))
    candidate_quotient_audit = {
        "payload_definition": ["A", "Y_candidate"],
        "source_origin_part_of_equivalence": False,
        "noncollision_distinct_states": noncollision_count,
        "target_calibration_collision_distinct_states": collision_count,
        "expected_noncollision_states": 6,
        "expected_collision_states": 3,
        "orbit_recomputed_after_candidate_replacement": True,
    }
    require(noncollision_count == 6 and collision_count == 3, "D1_CANDIDATE_QUOTIENT_MISMATCH")

    # Synthetic Jacobian-direction audit independent of Stage 3B's unit scale.
    assignment = np.asarray([0, 1, 2], dtype=int)
    y = np.asarray([1.2, -0.4, 2.5], dtype=float)
    mean = np.zeros((3, 3), dtype=float)
    scale = np.asarray([[2.0, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 4.0]], dtype=float)
    _, log_j = affine_residual_scalar(assignment, y, mean, scale)
    expected_log_j = -math.log(2.0) - math.log(0.5) - math.log(4.0)
    jacobian_audit = {
        "assignment": "0,1,2",
        "scales": [2.0, 0.5, 4.0],
        "observed_log_inverse_jacobian": log_j,
        "expected_log_inverse_jacobian": expected_log_j,
        "absolute_error": abs(log_j - expected_log_j),
        "direction": "inverse_outcome_jacobian_product_1_over_scale",
    }
    require(jacobian_audit["absolute_error"] <= float(tolerances["jacobian_abs_tolerance"]), "Jacobian direction audit failed")

    normalization_frame = pd.DataFrame(normalization_rows)
    require(float(normalization_frame["absolute_error"].max()) <= float(tolerances["mixed_measure_normalization_abs_tolerance"]), "Mixed-measure normalization audit failed")

    # Endpoint audit: exact atom interventions retain zero-probability off-atom states.
    endpoint_rows = []
    for evaluation_id in ["G6_GAUSS_ENDPOINT0", "G6_GAUSS_ENDPOINT1"]:
        prepared = prepared_by_id[evaluation_id]
        result = evaluation_by_id[evaluation_id]
        assigned_A = prepared.A_payload[result.target_slot_payload_indices]
        expected_positive = assigned_A == prepared.fixture.target_dose
        observed_positive = result.probability > 0.0
        require(np.array_equal(expected_positive, observed_positive), f"Endpoint atom state support mismatch: {evaluation_id}")
        endpoint_rows.append(
            {
                "evaluation_id": evaluation_id,
                "target_dose": prepared.fixture.target_dose,
                "states_with_matching_endpoint_at_target_slot": int(np.count_nonzero(expected_positive)),
                "positive_probability_states": int(np.count_nonzero(observed_positive)),
                "off_atom_zero_probability_states": int(np.count_nonzero(~observed_positive)),
            }
        )

    # Explicit fail-closed refusal-path audit.
    from .graph_precision import block_is_connected
    from .mixed_measure import intervention_density
    from .orbit import validate_exact_block_size
    from .residual_transform import inverse_power_transform

    refusal_tests = []
    try:
        intervention_density(np.asarray([0.0]), 0.0, None, endpoint_audited=False)
        refusal_tests.append(("D1_UNSUPPORTED_ENDPOINT", False, "no exception"))
    except Exception as exc:
        refusal_tests.append(("D1_UNSUPPORTED_ENDPOINT", "D1_UNSUPPORTED_ENDPOINT" in str(exc), str(exc)))
    try:
        validate_exact_block_size(9, maximum=8)
        refusal_tests.append(("D1_BLOCK_TOO_LARGE", False, "no exception"))
    except Exception as exc:
        refusal_tests.append(("D1_BLOCK_TOO_LARGE", "D1_BLOCK_TOO_LARGE" in str(exc), str(exc)))
    disconnected_edges = pd.DataFrame({"source_node": [0], "target_node": [1]})
    disconnected_ok = not block_is_connected(np.asarray([0, 1, 2]), disconnected_edges)
    refusal_tests.append(("D1_BLOCK_DISCONNECTED", disconnected_ok, "synthetic disconnected block rejected"))
    try:
        inverse_power_transform(np.asarray([0.0]), 1.5, minimum_abs_residual=float(tolerances["non_gaussian_min_abs_residual"]))
        refusal_tests.append(("D1_TRANSFORM_SINGULAR", False, "no exception"))
    except Exception as exc:
        refusal_tests.append(("D1_TRANSFORM_SINGULAR", "D1_TRANSFORM_SINGULAR" in str(exc), str(exc)))
    # Direct target-positivity contract: q-positive/g-zero is not a legal tilt.
    positivity_rejected = True
    q_test = np.asarray([1.0, 0.0])
    g_test = np.asarray([0.0, 0.5])
    try:
        require(not np.any((q_test > 0.0) & (g_test <= 0.0)), "D1_POSITIVITY_FAILURE")
        positivity_rejected = False
    except Exception as exc:
        refusal_tests.append(("D1_POSITIVITY_FAILURE", "D1_POSITIVITY_FAILURE" in str(exc), str(exc)))
    if not positivity_rejected:
        refusal_tests.append(("D1_POSITIVITY_FAILURE", False, "no exception"))
    for code, passed, details in refusal_tests:
        refusal_rows.append({"refusal_code": code, "test_passed": bool(passed), "details": details})
    require(all(row["test_passed"] for row in refusal_rows), "D1 refusal-path audit failed")

    # Duplicate quotient audit.
    duplicate = prepared_by_id["G6_GAUSS_DUPLICATE_TRUTH"]
    duplicate_rows = [{
        "evaluation_id": duplicate.fixture.evaluation_id,
        "block_size": len(duplicate.fixture.payloads),
        "distinct_states": len(duplicate.fixture.permutations),
        "full_factorial_states": math.factorial(len(duplicate.fixture.payloads)),
        "expected_quotient_states": 360,
        "duplicate_state_removed_count": math.factorial(len(duplicate.fixture.payloads)) - len(duplicate.fixture.permutations),
        "candidate_source_index_tracked": True,
        "candidate_origin_part_of_payload_equivalence": False,
    }]
    require(len(duplicate.fixture.permutations) == 360, "D1_ENUMERATION_COUNT_MISMATCH: duplicate quotient")

    state_frame = pd.concat(state_frames, ignore_index=True)
    boundary_frame = pd.concat(boundary_frames, ignore_index=True).drop_duplicates().sort_values(["evaluation_id", "source_node", "target_node"])
    probability_frame = pd.DataFrame(probability_rows).sort_values("evaluation_id")
    comparison_frame = pd.DataFrame(comparison_rows).sort_values("evaluation_id")
    cancellation_frame = pd.DataFrame(cancellation_rows).sort_values("evaluation_id")
    fixture_frame = pd.DataFrame(fixture_rows).sort_values("evaluation_id")
    runtime_frame = pd.DataFrame(runtime_rows).sort_values("evaluation_id")
    endpoint_frame = pd.DataFrame(endpoint_rows).sort_values("evaluation_id")
    duplicate_frame = pd.DataFrame(duplicate_rows)
    treatment_spot_frame = pd.DataFrame(treatment_spot_rows).sort_values(["evaluation_id", "state_index", "slot_position"])
    target_tilt_frame = pd.DataFrame(target_tilt_rows).sort_values(["evaluation_id", "payload_index"])
    global_positivity_frame = pd.DataFrame(global_positivity_rows).sort_values("evaluation_id")
    full_target_law_frame = pd.DataFrame(full_target_law_rows).sort_values("evaluation_id")
    full_local_detail_frame = pd.DataFrame(full_local_detail_rows).sort_values(["evaluation_id", "state_index"])
    refusal_frame = pd.DataFrame(refusal_rows).sort_values("refusal_code")

    write_csv_gz(output_dir / "stage3d_d1_orbit_state_audit.csv.gz", state_frame)
    write_csv(output_dir / "stage3d_d1_fixture_registry.csv", fixture_frame)
    write_csv(output_dir / "stage3d_d1_probability_audit.csv", probability_frame)
    write_csv(output_dir / "stage3d_d1_scalar_vectorized_audit.csv", comparison_frame)
    write_csv(output_dir / "stage3d_d1_full_local_cancellation_audit.csv", cancellation_frame)
    write_csv(output_dir / "stage3d_d1_mixed_measure_normalization_audit.csv", normalization_frame)
    write_csv(output_dir / "stage3d_d1_boundary_edge_audit.csv", boundary_frame)
    write_csv(output_dir / "stage3d_d1_endpoint_atom_audit.csv", endpoint_frame)
    write_csv(output_dir / "stage3d_d1_duplicate_orbit_audit.csv", duplicate_frame)
    write_csv(output_dir / "stage3d_d1_treatment_factor_spot_audit.csv", treatment_spot_frame)
    write_csv(output_dir / "stage3d_d1_target_tilt_audit.csv", target_tilt_frame)
    write_csv(output_dir / "stage3d_d1_global_positivity_audit.csv", global_positivity_frame)
    write_csv(output_dir / "stage3d_d1_full_target_law_audit.csv", full_target_law_frame)
    write_csv_gz(output_dir / "stage3d_d1_full_local_state_audit.csv.gz", full_local_detail_frame)
    write_csv(output_dir / "stage3d_d1_refusal_audit.csv", refusal_frame)
    write_csv(output_dir / "stage3d_d1_runtime.csv", runtime_frame)
    write_json(output_dir / "stage3d_d1_candidate_algebra_audit.json", candidate_audit)
    write_json(output_dir / "stage3d_d1_candidate_quotient_audit.json", candidate_quotient_audit)
    write_json(output_dir / "stage3d_d1_jacobian_audit.json", jacobian_audit)
    write_json(output_dir / "stage3d_d1_input_audit.json", input_audit)
    write_json(output_dir / "stage3d_d1_environment_inventory.json", environment_inventory())

    source_hashes = _source_hashes(package_root)
    source_record = {"files": source_hashes, "tree_sha256": tree_hash(source_hashes)}
    write_json(output_dir / "stage3d_d1_source_hashes.json", source_record)

    contract_summary = {
        "stage": "3D",
        "gate": "D0_D1",
        "script_version": __version__,
        "scope": "exact candidate-specific quotient-orbit and graph-local algebra only",
        "nuisance_regime": "oracle_generator_truth_algebra_only",
        "payload_equivalence": "exact numerical pair (A,Y_candidate); source origin excluded",
        "headline_theorem": "G2 non-Gaussian graph-local target-orbit law",
        "G3_implemented": False,
        "prediction_sets_generated": False,
        "coverage_evaluated": False,
        "M3_implemented": False,
        "M4_implemented": False,
        "M6_implemented": False,
        "dense_precision_inverse_formed": False,
        "evaluation_count": len(fixture_frame),
        "primary_block_size": int(orbit_contract["primary_block_size"]),
        "stress_block_size": int(orbit_contract["stress_block_size"]),
    }
    write_json(output_dir / "stage3d_d1_contract_summary.json", contract_summary)

    counts = {
        "evaluation_count": len(fixture_frame),
        "total_orbit_state_rows": len(state_frame),
        "maximum_distinct_states": int(fixture_frame["distinct_states"].max()),
        "size6_unique_states": 720,
        "size8_unique_states": 40320,
        "size6_duplicate_states": 360,
        "boundary_edge_audit_rows": len(boundary_frame),
        "normalization_audit_rows": len(normalization_frame),
        "treatment_spot_audit_rows": len(treatment_spot_frame),
        "full_local_state_audit_rows": len(full_local_detail_frame),
        "refusal_audit_rows": len(refusal_frame),
        "target_tilt_audit_rows": len(target_tilt_frame),
        "global_positivity_audit_rows": len(global_positivity_frame),
        "full_target_law_audit_rows": len(full_target_law_frame),
        "G3_implemented": False,
        "coverage_evaluated": False,
        "production_experiments_run": False,
    }
    write_json(output_dir / "stage3d_d1_counts.json", counts)

    # Manifest tracks every completed D1 artifact except itself and the later terminal verification record.
    tracked = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"stage3d_d1_manifest.json", "STAGE3D_D1_VERIFICATION.json"}:
            continue
        tracked[path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "stage": "3D_D0_D1",
        "script_version": __version__,
        "input_archive_sha256": {name: row["observed_sha256"] for name, row in input_audit.items()},
        "source_tree_sha256": source_record["tree_sha256"],
        "outputs": tracked,
        "production_experiments_run": False,
    }
    write_json(output_dir / "stage3d_d1_manifest.json", manifest)

    elapsed = time.perf_counter() - started
    print("STAGE 3D D0/D1 EXACT-ORBIT ALGEBRA GENERATION COMPLETE")
    print(f"Evaluations: {len(fixture_frame)}")
    print(f"Orbit state rows: {len(state_frame):,}")
    print(f"Maximum exact states: {int(fixture_frame['distinct_states'].max()):,}")
    print("G3 implemented: no")
    print("Prediction sets generated: no")
    print("Coverage evaluated: no")
    print(f"Elapsed seconds: {elapsed:.3f}")
    return {"counts": counts, "manifest": manifest, "elapsed_seconds": elapsed}
