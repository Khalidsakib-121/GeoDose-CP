from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_inversion import (
    CandidateEvaluator,
    acceptance,
    candidate_landmarks,
    initial_candidate_points,
    intersect_component_sets,
    outer_components_from_cache,
    point_in_components,
    refine_boundaries,
    summarize_components,
)
from .d2_io import (
    build_d2_fixture,
    derive_seed,
    frozen_orbit_seed,
    registered_candidate_domain,
    stage3b_data,
    verify_d2_inputs,
)
from .d2_version import D2_VERSION
from .exact_law import prepare_orbit
from .io import read_yaml, require
from .provenance import environment_inventory, package_file_hashes, sha256_file, tree_hash, write_json
from .topology_validation import run_topology_validation


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    import gzip

    raw = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    with path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
            compressed.write(raw)


def _source_hashes(root: Path) -> dict[str, str]:
    excluded = {".venv", "outputs_stage3d_d2", "__pycache__", ".pytest_cache"}
    hashes = package_file_hashes(root, excluded_parts=excluded)
    return {
        name: value
        for name, value in hashes.items()
        if not name.startswith("inputs/upstream/") and not name.startswith("inputs/math/")
    }


def _point_sources(value: float, source_map: dict[str, set[float]], post_sources: dict[str, set[float]]) -> str:
    labels = [label for label, values in source_map.items() if float(value) in values]
    labels.extend(label for label, values in post_sources.items() if float(value) in values)
    return "+".join(sorted(labels)) if labels else "adaptive_refinement"


def _boundary_contains(value: float, boundaries: list[Any], multiplier: float) -> bool:
    value = float(value)
    for item in boundaries:
        pad = float(multiplier) * float(item.width)
        if float(item.left_y) - pad <= value <= float(item.right_y) + pad:
            return True
    return False


def _independent_grid_audit(
    fixture_id: str,
    prepared,
    tie_uniform: float,
    alpha: float,
    lower: float,
    upper: float,
    components_by_mode: dict[str, list[dict[str, Any]]],
    boundaries_by_mode: dict[str, list[Any]],
    singular_landmarks: list[float],
    tolerances: dict[str, Any],
    *,
    grid_points: int,
    boundary_multiplier: float,
) -> tuple[list[dict[str, Any]], int]:
    # Offset grid is independent of the primary endpoint-inclusive certification
    # grid and therefore probes unsampled candidate locations.
    step = (upper - lower) / float(grid_points)
    points = lower + (np.arange(int(grid_points), dtype=float) + 0.5) * step
    evaluator = CandidateEvaluator(
        prepared,
        alpha=alpha,
        tie_uniform=tie_uniform,
        score_abs_tolerance=float(tolerances["score_tie_abs_tolerance"]),
        score_rel_tolerance=float(tolerances["score_tie_rel_tolerance"]),
        non_gaussian_min_abs_residual=float(tolerances["non_gaussian_min_abs_residual"]),
    )
    results = [evaluator.evaluate(float(value)) for value in points]
    rows: list[dict[str, Any]] = []
    for mode in ("randomized", "conservative"):
        false_negative = 0
        false_positive = 0
        unexplained_false_positive = 0
        for value, result in zip(points.tolist(), results):
            direct = acceptance(result, mode)
            represented = point_in_components(float(value), components_by_mode[mode])
            false_negative += int(direct and not represented)
            if represented and not direct:
                false_positive += 1
                explained = _boundary_contains(
                    float(value), boundaries_by_mode[mode], float(boundary_multiplier)
                ) or any(abs(float(value) - landmark) <= step for landmark in singular_landmarks)
                unexplained_false_positive += int(not explained)
        require(false_negative == 0, "D2_INDEPENDENT_GRID_FALSE_NEGATIVE")
        require(unexplained_false_positive == 0, "D2_OUTER_SET_EXCESS_UNEXPLAINED")
        rows.append(
            {
                "fixture_id": fixture_id,
                "tie_mode": mode,
                "independent_candidate_count": int(len(points)),
                "false_negative_count": int(false_negative),
                "outer_false_positive_count": int(false_positive),
                "unexplained_outer_false_positive_count": int(unexplained_false_positive),
                "grid_step": float(step),
                "audit_pass": True,
            }
        )
    return rows, len(evaluator.cached_results())


def _state_audit_rows(fixture_id: str, result, alpha: float) -> list[dict[str, Any]]:
    if result.state_probability is None:
        return []
    rows: list[dict[str, Any]] = []
    for index in range(len(result.permutations)):
        category = str(result.state_category[index])
        rows.append(
            {
                "fixture_id": fixture_id,
                "candidate_y": result.candidate_y,
                "candidate_hex": float(result.candidate_y).hex(),
                "state_index": index,
                "assignment": ",".join(str(int(v)) for v in result.permutations[index].tolist()),
                "state_probability": float(result.state_probability[index]),
                "state_score": float(result.state_score[index]),
                "observed_score": float(result.observed_score),
                "score_category": category,
                "certified_strict_greater": category == "certified_strict_greater",
                "certified_structural_tie": category == "certified_structural_tie",
                "ambiguous_included": category == "ambiguous_included",
                "certified_strict_less": category == "certified_strict_less",
                "alpha": alpha,
            }
        )
    return rows


def run_d2(package_root: str | Path, output_dir: str | Path, *, overwrite: bool) -> dict[str, Any]:
    root = Path(package_root).resolve()
    output = Path(output_dir).resolve()
    require(root in output.parents or output.parent == root, "D2 output must be package-local")
    if output.exists():
        if not overwrite:
            require(not any(output.iterdir()), "D2 output nonempty; use --overwrite")
        else:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    input_audit = verify_d2_inputs(root)
    math_contract = read_yaml(root / "configs" / "stage3d_d2_math_contract.yaml")
    candidate_contract = read_yaml(root / "configs" / "stage3d_d2_candidate_contract.yaml")
    fixture_registry = read_yaml(root / "configs" / "stage3d_d2_fixture_registry.yaml")
    tolerances = read_yaml(root / "configs" / "stage3d_d2_tolerances.yaml")
    refusal_contract = read_yaml(root / "configs" / "stage3d_d2_refusal_contract.yaml")
    d1_tolerances = read_yaml(root / "inputs" / "d1_contracts" / "stage3d_tolerance_registry.yaml")

    require(
        math_contract["release_scope"] == "D2_G3_candidate_inversion_reference_domain_truncated_outer",
        "D2_CONTRACT_MISMATCH",
    )
    require(math_contract["score"]["expression"] == "S_pi_y = abs(E_target_slot_pi_y)", "D2_CONTRACT_MISMATCH")
    require(candidate_contract["randomization"]["one_uniform_per_fixture"] is True, "D2_CONTRACT_MISMATCH")
    require(candidate_contract["candidate_domain"]["target_truth_allowed"] is False, "D2_TARGET_TRUTH_LEAKAGE")
    require(candidate_contract["candidate_domain"]["observed_outcomes_allowed"] is False, "D2_TARGET_TRUTH_LEAKAGE")
    require(candidate_contract["candidate_domain"]["unbounded_claim"] is False, "D2_CONTRACT_MISMATCH")
    require(int(tolerances["size8_maximum_states"]) == 40320, "D2_CONTRACT_MISMATCH")

    data = stage3b_data(root)
    base_orbit_seed = frozen_orbit_seed(root, "S4", 1)
    alpha = float(candidate_contract["alpha_primary"])

    grid_registry_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    independent_grid_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    refusal_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []

    for registry_row in fixture_registry["full_inversion_fixtures"]:
        fixture_started = time.perf_counter()
        fixture_id = str(registry_row["d2_fixture_id"])
        print(f"[D2] Starting full inversion fixture {fixture_id}", flush=True)
        fixture = build_d2_fixture(data, root, str(registry_row["d1_evaluation_id"]))
        require(len(fixture.slots) == 6, "D2_CONTRACT_MISMATCH: full inversion is size6 only")
        prepared = prepare_orbit(
            fixture,
            minimum_precision_eigenvalue_required=float(tolerances["minimum_precision_eigenvalue"]),
        )
        domain = registered_candidate_domain(candidate_contract, fixture.case_id)
        lower, upper = float(domain["lower"]), float(domain["upper"])
        landmarks = candidate_landmarks(prepared, lower, upper)
        initial_points, source_map = initial_candidate_points(
            lower,
            upper,
            landmarks,
            coarse_points=int(candidate_contract["initial_grid"]["coarse_points"]),
            staggered_points=int(candidate_contract["initial_grid"]["staggered_midpoints"]),
            certification_points=int(candidate_contract["initial_grid"]["certification_grid_points"]),
            flank_relative=float(candidate_contract["initial_grid"]["landmark_flank_relative_to_domain"]),
            flank_minimum=float(candidate_contract["initial_grid"]["landmark_flank_minimum"]),
        )

        tie_seed = derive_seed(base_orbit_seed, "D2_G3_TIE_UNIFORM", fixture_id)
        tie_uniform = float(np.random.default_rng(tie_seed).random())
        evaluator = CandidateEvaluator(
            prepared,
            alpha=alpha,
            tie_uniform=tie_uniform,
            score_abs_tolerance=float(tolerances["score_tie_abs_tolerance"]),
            score_rel_tolerance=float(tolerances["score_tie_rel_tolerance"]),
            non_gaussian_min_abs_residual=float(tolerances["non_gaussian_min_abs_residual"]),
        )
        for point in initial_points.tolist():
            evaluator.evaluate(point)

        boundaries_by_mode: dict[str, list[Any]] = {}
        for mode in ("randomized", "conservative"):
            boundaries = refine_boundaries(
                evaluator,
                initial_points,
                mode=mode,
                absolute_tolerance=float(tolerances["boundary_absolute_tolerance"]),
                relative_tolerance=float(tolerances["boundary_relative_tolerance"]),
                maximum_depth=int(tolerances["boundary_maximum_depth"]),
            )
            boundaries_by_mode[mode] = boundaries
            for index, item in enumerate(boundaries, start=1):
                boundary_rows.append(
                    {
                        "fixture_id": fixture_id,
                        "tie_mode": mode,
                        "boundary_id": index,
                        **item.__dict__,
                    }
                )

        certified_exact_points = set(float(value) for value in landmarks["transform_singularity"])
        components_by_mode = {
            mode: outer_components_from_cache(
                evaluator,
                lower,
                upper,
                mode=mode,
                certified_exact_points=certified_exact_points,
            )
            for mode in ("randomized", "conservative")
        }
        # Preserve the exact randomized-subset-conservative relationship after
        # numerical set enlargement. Intersecting two outer supersets still
        # contains the exact randomized set because that set is contained in
        # the exact conservative set and therefore in its outer superset.
        components_by_mode["randomized"] = intersect_component_sets(
            components_by_mode["randomized"],
            components_by_mode["conservative"],
            tie_mode="randomized",
        )

        for mode, components in components_by_mode.items():
            for component in components:
                component_rows.append({"fixture_id": fixture_id, **component})
            summary = summarize_components(components, lower, upper)
            summary_rows.append(
                {
                    "fixture_id": fixture_id,
                    "case_id": fixture.case_id,
                    "residual_law": fixture.residual_law,
                    "target_dose": fixture.target_dose,
                    "tie_mode": mode,
                    "alpha": alpha,
                    "domain_lower": lower,
                    "domain_upper": upper,
                    **summary,
                    "returned_set_representation": "coverage_preserving_outer_numerical_set_intersect_registered_domain",
                    "software_reference_only": True,
                    "coverage_claim": False,
                }
            )

        # Randomized certified set must remain a subset of the conservative set
        # at every evaluated candidate.
        for result in evaluator.cached_results():
            require(
                not result.randomized_accept or result.conservative_accept,
                "D2_RANDOMIZED_NOT_SUBSET_CONSERVATIVE",
            )

        audit_rows, audit_eval_count = _independent_grid_audit(
            fixture_id,
            prepared,
            tie_uniform,
            alpha,
            lower,
            upper,
            components_by_mode,
            boundaries_by_mode,
            landmarks["transform_singularity"],
            tolerances,
            grid_points=int(candidate_contract["independent_grid_audit"]["points"]),
            boundary_multiplier=float(tolerances["outer_false_positive_boundary_multiplier"]),
        )
        independent_grid_rows.extend(audit_rows)

        # Target truth is evaluated only after the domain, grids, boundaries,
        # and returned outer components are frozen.
        post_sources = {"post_inversion_truth_audit": set(), "post_inversion_shifted_audit": set()}
        truth = float(fixture.target_payload_truth)
        shifted = truth + float(d1_tolerances["candidate_algebra_shift"])
        require(lower <= truth <= upper, "D2_CANDIDATE_DOMAIN_INVALID: diagnostic truth outside registered domain")
        truth_result = evaluator.evaluate(truth)
        shifted_result = evaluator.evaluate(shifted)
        post_sources["post_inversion_truth_audit"].add(truth)
        post_sources["post_inversion_shifted_audit"].add(shifted)
        truth_rows.append(
            {
                "fixture_id": fixture_id,
                "target_truth": truth,
                "truth_randomized_p": truth_result.randomized_p,
                "truth_conservative_p": truth_result.conservative_p,
                "truth_in_randomized_outer_components": point_in_components(truth, components_by_mode["randomized"]),
                "truth_in_conservative_outer_components": point_in_components(truth, components_by_mode["conservative"]),
                "post_inversion_diagnostic_only": True,
                "coverage_evidence": False,
            }
        )
        tie_rows.append(
            {
                "fixture_id": fixture_id,
                "base_stage3a_orbit_seed": base_orbit_seed,
                "derived_tie_seed": tie_seed,
                "tie_uniform": tie_uniform,
                "tie_uniform_hex": float(tie_uniform).hex(),
                "uniform_held_fixed_across_all_candidates": True,
                "ambiguous_mass_included_in_full": True,
                "conservative_p_is_primary_certified": True,
            }
        )

        audit_candidates = sorted({float(domain["center"]), truth, shifted})
        for candidate in audit_candidates:
            result = evaluator.evaluate(candidate, keep_state=True)
            state_rows.extend(_state_audit_rows(fixture_id, result, alpha))

        singular_count = sum(result.singular_conservative_inclusion for result in evaluator.cached_results())
        if singular_count:
            warning_rows.append(
                {
                    "fixture_id": fixture_id,
                    "warning_code": "D2_TRANSFORM_SINGULAR_POINT_INCLUDED",
                    "candidate_count": int(singular_count),
                    "effect": "candidate included in both sets; no pointwise orbit-pmf claim",
                }
            )
        warning_rows.append(
            {
                "fixture_id": fixture_id,
                "warning_code": "D2_DOMAIN_TRUNCATED_SET",
                "candidate_count": int(len(evaluator.cached_results())),
                "effect": "returned set is restricted to fixed [-8,8] domain; no full-real-line or unbounded claim",
            }
        )

        for result in evaluator.cached_results():
            trace_rows.append(
                {
                    "fixture_id": fixture_id,
                    "case_id": fixture.case_id,
                    "residual_law": fixture.residual_law,
                    "candidate_y": result.candidate_y,
                    "candidate_hex": float(result.candidate_y).hex(),
                    "candidate_source": _point_sources(result.candidate_y, source_map, post_sources),
                    "distinct_states": result.distinct_states,
                    "observed_score": result.observed_score,
                    "certified_strict_mass": result.certified_strict_mass,
                    "structural_tie_mass": result.structural_tie_mass,
                    "numerically_ambiguous_mass": result.numerically_ambiguous_mass,
                    "certified_less_mass": result.certified_less_mass,
                    "conservative_p": result.conservative_p,
                    "randomized_p": result.randomized_p,
                    "tie_uniform": result.tie_uniform,
                    "conservative_accept": result.conservative_accept,
                    "randomized_accept": result.randomized_accept,
                    "randomized_exact_tie_claim": result.randomized_exact_tie_claim,
                    "singular_conservative_inclusion": result.singular_conservative_inclusion,
                    "singular_zero_state_count": result.singular_zero_state_count,
                    "probability_sum": result.probability_sum,
                    "positive_probability_states": result.positive_probability_states,
                    "zero_probability_states": result.zero_probability_states,
                    "quotient_changed_from_reference": result.quotient_changed_from_reference,
                }
            )

        grid_registry_rows.append(
            {
                "fixture_id": fixture_id,
                "case_id": fixture.case_id,
                "residual_law": fixture.residual_law,
                "target_dose": fixture.target_dose,
                **domain,
                "alpha": alpha,
                "coarse_points": int(candidate_contract["initial_grid"]["coarse_points"]),
                "staggered_points": int(candidate_contract["initial_grid"]["staggered_midpoints"]),
                "certification_grid_points": int(candidate_contract["initial_grid"]["certification_grid_points"]),
                "initial_unique_points": int(len(initial_points)),
                "score_tie_landmarks": len(landmarks["score_tie"]),
                "payload_collision_landmarks": len(landmarks["payload_collision"]),
                "transform_singularity_landmarks": len(landmarks["transform_singularity"]),
                "tie_seed": tie_seed,
                "tie_uniform": tie_uniform,
                "domain_frozen_before_truth_audit": True,
            }
        )
        print(f"[D2] Finished {fixture_id} in {time.perf_counter() - fixture_started:.3f} s", flush=True)
        runtime_rows.append(
            {
                "fixture_id": fixture_id,
                "task": "full_candidate_inversion",
                "seconds": time.perf_counter() - fixture_started,
                "candidate_evaluations": len(evaluator.cached_results()),
                "independent_grid_evaluations": audit_eval_count,
            }
        )

    # Size-8 p-value stress only; full set inversion remains deferred.
    for registry_row in fixture_registry["stress_fixtures"]:
        stress_started = time.perf_counter()
        fixture_id = str(registry_row["d2_fixture_id"])
        print(f"[D2] Starting size8 stress fixture {fixture_id}", flush=True)
        fixture = build_d2_fixture(data, root, str(registry_row["d1_evaluation_id"]))
        prepared = prepare_orbit(
            fixture,
            minimum_precision_eigenvalue_required=float(tolerances["minimum_precision_eigenvalue"]),
        )
        domain = registered_candidate_domain(candidate_contract, fixture.case_id)
        tie_seed = derive_seed(base_orbit_seed, "D2_G3_TIE_UNIFORM", fixture_id)
        tie_uniform = float(np.random.default_rng(tie_seed).random())
        evaluator = CandidateEvaluator(
            prepared,
            alpha=alpha,
            tie_uniform=tie_uniform,
            score_abs_tolerance=float(tolerances["score_tie_abs_tolerance"]),
            score_rel_tolerance=float(tolerances["score_tie_rel_tolerance"]),
            non_gaussian_min_abs_residual=float(tolerances["non_gaussian_min_abs_residual"]),
        )
        candidates = [
            ("domain_lower", float(domain["lower"])),
            ("domain_center", float(domain["center"])),
            ("post_inversion_truth_audit", float(fixture.target_payload_truth)),
            (
                "post_inversion_shifted_audit",
                float(fixture.target_payload_truth) + float(d1_tolerances["candidate_algebra_shift"]),
            ),
            ("domain_upper", float(domain["upper"])),
        ]
        for label, candidate in candidates:
            result = evaluator.evaluate(candidate)
            require(result.distinct_states <= int(tolerances["size8_maximum_states"]), "D2_SIZE8_LIMIT_EXCEEDED")
            stress_rows.append(
                {
                    "fixture_id": fixture_id,
                    "candidate_label": label,
                    "candidate_y": candidate,
                    "distinct_states": result.distinct_states,
                    "randomized_p": result.randomized_p,
                    "conservative_p": result.conservative_p,
                    "randomized_accept": result.randomized_accept,
                    "conservative_accept": result.conservative_accept,
                    "ambiguous_mass": result.numerically_ambiguous_mass,
                    "full_inversion_performed": False,
                }
            )
            if label == "post_inversion_truth_audit":
                detailed = evaluator.evaluate(candidate, keep_state=True)
                state_rows.extend(_state_audit_rows(fixture_id, detailed, alpha))
        runtime_rows.append(
            {
                "fixture_id": fixture_id,
                "task": "size8_candidate_pvalue_stress",
                "seconds": time.perf_counter() - stress_started,
                "candidate_evaluations": len(candidates),
                "independent_grid_evaluations": 0,
            }
        )

    topology_rows = run_topology_validation(
        absolute_tolerance=float(tolerances["boundary_absolute_tolerance"]),
        relative_tolerance=float(tolerances["boundary_relative_tolerance"]),
        maximum_depth=int(tolerances["boundary_maximum_depth"]),
    )

    print("[D2] Writing scientific outputs", flush=True)
    write_json(output / "stage3d_d2_input_audit.json", input_audit)
    write_json(
        output / "stage3d_d2_contract_summary.json",
        {
            "script_version": D2_VERSION,
            "math_contract": math_contract,
            "candidate_contract": candidate_contract,
            "refusal_contract": refusal_contract,
            "base_stage3a_orbit_seed": base_orbit_seed,
        },
    )
    _write_csv(output / "stage3d_d2_candidate_grid_registry.csv", pd.DataFrame(grid_registry_rows))
    _write_csv_gz(output / "stage3d_d2_candidate_pvalue_trace.csv.gz", pd.DataFrame(trace_rows))
    _write_csv(output / "stage3d_d2_boundary_refinement_audit.csv", pd.DataFrame(boundary_rows))
    _write_csv(output / "stage3d_d2_prediction_set_components.csv", pd.DataFrame(component_rows))
    _write_csv(output / "stage3d_d2_prediction_set_summary.csv", pd.DataFrame(summary_rows))
    _write_csv(output / "stage3d_d2_independent_grid_audit.csv", pd.DataFrame(independent_grid_rows))
    _write_csv(output / "stage3d_d2_topology_validation.csv", pd.DataFrame(topology_rows))
    _write_csv(output / "stage3d_d2_truth_membership_diagnostic.csv", pd.DataFrame(truth_rows))
    _write_csv(output / "stage3d_d2_tie_randomization_audit.csv", pd.DataFrame(tie_rows))
    _write_csv_gz(output / "stage3d_d2_candidate_state_audit.csv.gz", pd.DataFrame(state_rows))
    _write_csv(
        output / "stage3d_d2_warning_log.csv",
        pd.DataFrame(warning_rows, columns=["fixture_id", "warning_code", "candidate_count", "effect"]),
    )
    _write_csv(
        output / "stage3d_d2_refusal_log.csv",
        pd.DataFrame(refusal_rows, columns=["fixture_id", "refusal_code", "detail"]),
    )
    _write_csv(output / "stage3d_d2_size8_stress_audit.csv", pd.DataFrame(stress_rows))
    _write_csv(output / "stage3d_d2_runtime.csv", pd.DataFrame(runtime_rows))

    print("[D2] Computing source provenance hashes", flush=True)
    source_hashes = _source_hashes(root)
    write_json(
        output / "stage3d_d2_source_hashes.json",
        {"script_version": D2_VERSION, "files": source_hashes, "tree_hash": tree_hash(source_hashes)},
    )
    write_json(output / "stage3d_d2_environment_inventory.json", environment_inventory())

    counts = {
        "full_inversion_fixture_count": len(fixture_registry["full_inversion_fixtures"]),
        "size8_stress_fixture_count": len(fixture_registry["stress_fixtures"]),
        "candidate_trace_rows": len(trace_rows),
        "boundary_rows": len(boundary_rows),
        "prediction_component_rows": len(component_rows),
        "prediction_summary_rows": len(summary_rows),
        "independent_grid_audit_rows": len(independent_grid_rows),
        "topology_validation_rows": len(topology_rows),
        "candidate_state_audit_rows": len(state_rows),
        "size8_stress_rows": len(stress_rows),
        "warning_rows": len(warning_rows),
        "refusal_rows": len(refusal_rows),
        "G3_candidate_pvalues_implemented": True,
        "prediction_sets_generated": True,
        "returned_sets_domain_truncated_outer": True,
        "unbounded_sets_claimed": False,
        "coverage_evaluated": False,
        "M3_implemented": False,
        "M4_implemented": False,
        "M6_implemented": False,
        "production_experiments_run": False,
    }
    write_json(output / "stage3d_d2_counts.json", counts)

    manifest_outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name in {"stage3d_d2_manifest.json", "STAGE3D_D2_VERIFICATION.json"}:
            continue
        manifest_outputs[path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "stage": "3D_D2",
        "script_version": D2_VERSION,
        "scope": "G3_candidate_inversion_reference_domain_truncated_outer",
        "outputs": manifest_outputs,
        "elapsed_seconds": time.perf_counter() - started,
        "coverage_evaluated": False,
        "production_experiments_run": False,
    }
    write_json(output / "stage3d_d2_manifest.json", manifest)
    return counts
