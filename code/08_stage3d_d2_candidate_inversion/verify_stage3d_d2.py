from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from geodose_stage3d.candidate_inversion import point_in_components, summarize_components
from geodose_stage3d.d2_io import (
    derive_seed,
    frozen_orbit_seed,
    registered_candidate_domain,
    verify_d2_inputs,
)
from geodose_stage3d.d2_version import D2_VERSION
from geodose_stage3d.io import read_yaml, require
from geodose_stage3d.provenance import environment_inventory, package_file_hashes, sha256_file, tree_hash, write_json


def read_csv(output: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(output / name)


def _component_records(frame: pd.DataFrame) -> list[dict]:
    return frame.sort_values("component_id").to_dict("records")


def main() -> None:
    root = ROOT
    output = root / "outputs_stage3d_d2"
    require(output.is_dir(), "D2 output directory missing")
    input_audit = verify_d2_inputs(root)
    candidate_contract = read_yaml(root / "configs" / "stage3d_d2_candidate_contract.yaml")
    tolerances = read_yaml(root / "configs" / "stage3d_d2_tolerances.yaml")

    checks: dict[str, str] = {}

    def passed(name: str, condition: bool) -> None:
        require(bool(condition), f"D2 verification failed: {name}")
        checks[name] = "pass"

    observed_environment = environment_inventory()
    required_versions = {
        "python": "3.10",
        "numpy": "2.2.6",
        "pandas": "2.3.3",
        "scipy": "1.15.3",
        "PyYAML": "6.0.3",
    }
    exact = (
        sys.version_info[:2] == (3, 10)
        and observed_environment.get("numpy") == required_versions["numpy"]
        and observed_environment.get("pandas") == required_versions["pandas"]
        and observed_environment.get("scipy") == required_versions["scipy"]
        and observed_environment.get("PyYAML") == required_versions["PyYAML"]
    )
    dev_override = os.environ.get("GEODOSE_ALLOW_DEV_RUNTIME") == "1"
    passed("environment_gate", exact or dev_override)
    passed("input_hashes", all(v["expected_sha256"] == v["observed_sha256"] for v in input_audit.values()))

    manifest = json.loads((output / "stage3d_d2_manifest.json").read_text(encoding="utf-8"))
    passed("manifest_stage", manifest["stage"] == "3D_D2")
    passed("manifest_version", manifest["script_version"] == D2_VERSION)
    passed("manifest_scope", manifest["scope"] == "G3_candidate_inversion_reference_domain_truncated_outer")
    manifest_ok = True
    for name, meta in manifest["outputs"].items():
        path = output / name
        manifest_ok &= (
            path.is_file()
            and path.stat().st_size == int(meta["size_bytes"])
            and sha256_file(path) == meta["sha256"]
        )
    passed("output_manifest_hashes", manifest_ok)

    counts = json.loads((output / "stage3d_d2_counts.json").read_text(encoding="utf-8"))
    passed("G3_pvalues_implemented", counts["G3_candidate_pvalues_implemented"] is True)
    passed("prediction_sets_generated", counts["prediction_sets_generated"] is True)
    passed("returned_sets_domain_truncated", counts["returned_sets_domain_truncated_outer"] is True)
    passed("no_unbounded_claim", counts["unbounded_sets_claimed"] is False)
    passed("coverage_not_evaluated", counts["coverage_evaluated"] is False)
    passed("M3_deferred", counts["M3_implemented"] is False)
    passed("M4_deferred", counts["M4_implemented"] is False)
    passed("M6_deferred", counts["M6_implemented"] is False)
    passed("production_not_run", counts["production_experiments_run"] is False)

    grid = read_csv(output, "stage3d_d2_candidate_grid_registry.csv")
    trace = read_csv(output, "stage3d_d2_candidate_pvalue_trace.csv.gz")
    boundaries = read_csv(output, "stage3d_d2_boundary_refinement_audit.csv")
    components = read_csv(output, "stage3d_d2_prediction_set_components.csv")
    summaries = read_csv(output, "stage3d_d2_prediction_set_summary.csv")
    independent = read_csv(output, "stage3d_d2_independent_grid_audit.csv")
    topology = read_csv(output, "stage3d_d2_topology_validation.csv")
    truth = read_csv(output, "stage3d_d2_truth_membership_diagnostic.csv")
    tie = read_csv(output, "stage3d_d2_tie_randomization_audit.csv")
    state = read_csv(output, "stage3d_d2_candidate_state_audit.csv.gz")
    warnings = read_csv(output, "stage3d_d2_warning_log.csv")
    refusals = read_csv(output, "stage3d_d2_refusal_log.csv")
    stress = read_csv(output, "stage3d_d2_size8_stress_audit.csv")

    passed("five_full_inversion_fixtures", grid["fixture_id"].nunique() == 5)
    passed("candidate_trace_unique", not trace.duplicated(["fixture_id", "candidate_hex"]).any())
    passed("two_tie_modes", set(components["tie_mode"]) == {"randomized", "conservative"})

    domain = registered_candidate_domain(candidate_contract, "audit")
    passed("fixed_domain_bounds", (grid["lower"] == float(domain["lower"])).all() and (grid["upper"] == float(domain["upper"])).all())
    passed("domain_type", (grid["domain_type"] == "fixed_registered_domain_truncated_reference").all())
    passed("domain_no_outcome_leakage", (
        ~grid["target_truth_used"].astype(bool)
        & ~grid["calibration_outcomes_used"].astype(bool)
        & ~grid["support_audit_outcomes_used"].astype(bool)
        & ~grid["nuisance_training_outcomes_used"].astype(bool)
        & grid["domain_frozen_before_truth_audit"].astype(bool)
    ).all())
    passed("domain_truncated_flags", grid["domain_truncated_set"].astype(bool).all())
    passed("domain_no_full_line_claim", (~grid["full_real_line_claim"].astype(bool) & ~grid["unbounded_claim"].astype(bool)).all())

    nonsingular = ~trace["singular_conservative_inclusion"].astype(bool)
    passed("pvalues_bounded", (
        trace["conservative_p"].between(0.0, 1.0)
        & trace["randomized_p"].between(0.0, 1.0)
    ).all())
    passed("randomized_not_above_conservative", (trace["randomized_p"] <= trace["conservative_p"] + 1e-13).all())
    alpha = float(candidate_contract["alpha_primary"])
    passed("acceptance_rule_randomized", (trace["randomized_accept"].astype(bool) == (trace["randomized_p"] > alpha)).all())
    passed("acceptance_rule_conservative", (trace["conservative_accept"].astype(bool) == (trace["conservative_p"] > alpha)).all())
    passed("randomized_accept_subset", (~trace["randomized_accept"].astype(bool) | trace["conservative_accept"].astype(bool)).all())

    mass_sum = (
        trace.loc[nonsingular, "certified_strict_mass"]
        + trace.loc[nonsingular, "structural_tie_mass"]
        + trace.loc[nonsingular, "numerically_ambiguous_mass"]
        + trace.loc[nonsingular, "certified_less_mass"]
    )
    passed("score_category_mass_sum", np.max(np.abs(mass_sum - 1.0)) <= 5e-12)
    conservative_expected = (
        trace.loc[nonsingular, "certified_strict_mass"]
        + trace.loc[nonsingular, "numerically_ambiguous_mass"]
        + trace.loc[nonsingular, "structural_tie_mass"]
    )
    randomized_expected = (
        trace.loc[nonsingular, "certified_strict_mass"]
        + trace.loc[nonsingular, "numerically_ambiguous_mass"]
        + trace.loc[nonsingular, "tie_uniform"] * trace.loc[nonsingular, "structural_tie_mass"]
    )
    passed("conservative_mass_identity", np.max(np.abs(trace.loc[nonsingular, "conservative_p"] - conservative_expected)) <= 3e-12)
    passed("randomized_mass_identity", np.max(np.abs(trace.loc[nonsingular, "randomized_p"] - randomized_expected)) <= 3e-12)
    exact_claim_expected = trace["numerically_ambiguous_mass"].fillna(np.inf) == 0.0
    exact_claim_expected.loc[~nonsingular] = False
    passed("randomized_exact_claim_honest", (trace["randomized_exact_tie_claim"].astype(bool) == exact_claim_expected).all())
    passed("ambiguous_mass_nonnegative", (trace.loc[nonsingular, "numerically_ambiguous_mass"] >= -1e-15).all())

    base_seed = frozen_orbit_seed(root, "S4", 1)
    seed_ok = True
    uniform_ok = True
    for row in tie.itertuples(index=False):
        expected_seed = derive_seed(base_seed, "D2_G3_TIE_UNIFORM", row.fixture_id)
        expected_uniform = float(np.random.default_rng(expected_seed).random())
        seed_ok &= int(row.derived_tie_seed) == expected_seed and int(row.base_stage3a_orbit_seed) == base_seed
        uniform_ok &= str(row.tie_uniform_hex) == float(expected_uniform).hex()
    passed("frozen_tie_seed", seed_ok)
    passed("one_uniform_per_fixture", uniform_ok and tie["uniform_held_fixed_across_all_candidates"].astype(bool).all())
    passed("ambiguous_included_rule", tie["ambiguous_mass_included_in_full"].astype(bool).all())
    passed("conservative_primary_rule", tie["conservative_p_is_primary_certified"].astype(bool).all())
    passed("uniform_held_across_candidates", int(trace.groupby("fixture_id")["tie_uniform"].nunique().max()) == 1)

    boundary_limit = np.maximum(
        float(tolerances["boundary_absolute_tolerance"]),
        float(tolerances["boundary_relative_tolerance"]) * np.maximum(1.0, boundaries["midpoint"].abs()),
    )
    passed("boundary_tolerance", (boundaries["width"] <= boundary_limit + 1e-15).all())
    passed("boundary_outer_endpoint_is_rejected_side", np.max(np.abs(boundaries["outer_endpoint"] - boundaries["rejected_y"])) <= 1e-15)
    passed("boundary_outer_endpoint_open", (~boundaries["outer_endpoint_closed"].astype(bool)).all())
    passed("boundary_representation", (boundaries["representation"] == "coverage_preserving_outer_boundary_bracket").all())

    passed("components_finite", np.isfinite(components[["lower", "upper", "width"]].to_numpy(float)).all())
    passed("component_bounds_ordered", (components["lower"] <= components["upper"]).all())
    passed("component_width_identity", np.max(np.abs(components["width"] - (components["upper"] - components["lower"]))) <= 2e-12)
    passed("component_no_unbounded_claim", (~summaries["unbounded_claim"].astype(bool)).all())
    passed("component_domain_truncated", summaries["domain_truncated_set"].astype(bool).all())
    passed("component_representation_declared", summaries["returned_set_representation"].eq("coverage_preserving_outer_numerical_set_intersect_registered_domain").all())
    passed("no_duplicate_component_ids", not components.duplicated(["fixture_id", "tie_mode", "component_id"]).any())

    # Recompute summaries and verify randomized outer set is a subset of the
    # conservative outer set at every archived and independent-grid candidate.
    summary_ok = True
    subset_ok = True
    for fixture_id in grid["fixture_id"]:
        by_mode: dict[str, list[dict]] = {}
        domain_row = grid.loc[grid["fixture_id"] == fixture_id].iloc[0]
        for mode in ("randomized", "conservative"):
            records = _component_records(components.loc[(components["fixture_id"] == fixture_id) & (components["tie_mode"] == mode)])
            by_mode[mode] = records
            observed = summaries.loc[(summaries["fixture_id"] == fixture_id) & (summaries["tie_mode"] == mode)].iloc[0]
            calc = summarize_components(records, float(domain_row["lower"]), float(domain_row["upper"]))
            for key in ("component_count", "raw_total_width", "hull_width", "hull_inflation"):
                expected = calc[key]
                value = observed[key]
                if expected is None:
                    summary_ok &= pd.isna(value)
                else:
                    summary_ok &= abs(float(value) - float(expected)) <= 2e-12
        fixture_trace = trace.loc[trace["fixture_id"] == fixture_id]
        for row in fixture_trace.itertuples(index=False):
            randomized_member = point_in_components(float(row.candidate_y), by_mode["randomized"])
            conservative_member = point_in_components(float(row.candidate_y), by_mode["conservative"])
            subset_ok &= (not randomized_member) or conservative_member
    passed("component_summaries_recomputed", summary_ok)
    passed("randomized_outer_components_subset", subset_ok)

    passed("independent_grid_rows", len(independent) == 10)
    passed("independent_grid_no_false_negatives", (independent["false_negative_count"] == 0).all())
    passed("independent_grid_no_unexplained_excess", (independent["unexplained_outer_false_positive_count"] == 0).all())
    passed("independent_grid_pass", independent["audit_pass"].astype(bool).all())

    passed("topology_fixture_count", len(topology) == 5)
    passed("topology_all_pass", topology["validation_pass"].astype(bool).all())
    passed("topology_no_false_negatives", (topology["false_negative_count"] == 0).all())
    passed("topology_no_false_positives", (topology["false_positive_count"] == 0).all())
    passed("topology_open_closed_match", topology["openness_closedness_match"].astype(bool).all())
    topology_ids = set(topology["fixture_id"])
    passed("topology_required_cases", topology_ids == {
        "continuous_open_interval",
        "disconnected_two_interval",
        "empty_set",
        "certified_isolated_point",
        "closed_step_discontinuity",
    })

    passed("truth_post_inversion_only", truth["post_inversion_diagnostic_only"].astype(bool).all() and (~truth["coverage_evidence"].astype(bool)).all())
    passed("truth_fixture_count", truth["fixture_id"].nunique() == 5)

    # Independently reconstruct state-level category masses.
    state_ok = True
    identity_ok = True
    for (fixture_id, candidate_hex), group in state.groupby(["fixture_id", "candidate_hex"], sort=False):
        group = group.sort_values("state_index")
        expected_identity = ",".join(str(i) for i in range(len(str(group.iloc[0]["assignment"]).split(","))))
        identity_ok &= int(group.iloc[0]["state_index"]) == 0 and str(group.iloc[0]["assignment"]) == expected_identity
        probability = group["state_probability"].to_numpy(float)
        strict_mass = float(probability[group["certified_strict_greater"].astype(bool).to_numpy()].sum())
        tie_mass = float(probability[group["certified_structural_tie"].astype(bool).to_numpy()].sum())
        ambiguous_mass = float(probability[group["ambiguous_included"].astype(bool).to_numpy()].sum())
        less_mass = float(probability[group["certified_strict_less"].astype(bool).to_numpy()].sum())
        trace_row = trace.loc[(trace["fixture_id"] == fixture_id) & (trace["candidate_hex"].astype(str) == str(candidate_hex))]
        if len(trace_row) == 1:
            row = trace_row.iloc[0]
            tol = float(tolerances["candidate_probability_replay_abs_tolerance"])
            state_ok &= abs(strict_mass - float(row["certified_strict_mass"])) <= tol
            state_ok &= abs(tie_mass - float(row["structural_tie_mass"])) <= tol
            state_ok &= abs(ambiguous_mass - float(row["numerically_ambiguous_mass"])) <= tol
            state_ok &= abs(less_mass - float(row["certified_less_mass"])) <= tol
    passed("identity_observed_assignment", identity_ok)
    passed("state_audit_pvalue_replay", state_ok)

    passed("singular_rule", ((~trace["singular_conservative_inclusion"].astype(bool)) | ((trace["randomized_p"] == 1.0) & (trace["conservative_p"] == 1.0))).all())
    if trace["singular_conservative_inclusion"].astype(bool).any():
        passed("singular_warning_present", "D2_TRANSFORM_SINGULAR_POINT_INCLUDED" in set(warnings["warning_code"]))
    else:
        checks["singular_warning_present"] = "pass"
    passed("domain_warning_per_fixture", warnings.loc[warnings["warning_code"] == "D2_DOMAIN_TRUNCATED_SET", "fixture_id"].nunique() == 5)
    passed("no_refusals_in_reference", len(refusals) == 0)

    passed("size8_rows", len(stress) == 5)
    passed("size8_state_count", (stress["distinct_states"] == 40320).all())
    passed("size8_no_full_inversion", (~stress["full_inversion_performed"].astype(bool)).all())

    source_record = json.loads((output / "stage3d_d2_source_hashes.json").read_text(encoding="utf-8"))
    excluded = {".venv", "outputs_stage3d_d2", "__pycache__", ".pytest_cache"}
    actual_all = package_file_hashes(root, excluded_parts=excluded)
    actual = {
        name: value
        for name, value in actual_all.items()
        if not name.startswith("inputs/upstream/") and not name.startswith("inputs/math/")
    }
    passed("source_hashes", source_record["files"] == actual)
    passed("source_tree_hash", source_record["tree_hash"] == tree_hash(actual))

    passed("candidate_trace_count", int(counts["candidate_trace_rows"]) == len(trace))
    passed("candidate_state_count", int(counts["candidate_state_audit_rows"]) == len(state))
    passed("topology_count", int(counts["topology_validation_rows"]) == len(topology))

    verification = {
        "stage": "3D_D2",
        "script_version": D2_VERSION,
        "status": "verified_complete",
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": 0,
        "exact_environment": bool(exact),
        "development_runtime_override": bool(dev_override and not exact),
        "required_environment": required_versions,
        "observed_environment": observed_environment,
        "full_inversion_fixture_count": int(grid["fixture_id"].nunique()),
        "candidate_trace_rows": int(len(trace)),
        "prediction_set_component_rows": int(len(components)),
        "topology_validation_rows": int(len(topology)),
        "size8_stress_rows": int(len(stress)),
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
    write_json(output / "STAGE3D_D2_VERIFICATION.json", verification)
    print("STAGE 3D D2 G3 CANDIDATE INVERSION VERIFIED COMPLETE")
    print(f"Verification checks: {len(checks)}/{len(checks)}")
    print(f"Full inversion fixtures: {grid['fixture_id'].nunique()}")
    print(f"Candidate trace rows: {len(trace):,}")
    print(f"Prediction-set components: {len(components):,}")
    print(f"Topology validation fixtures: {len(topology)}")
    print("G3 candidate p-values implemented: yes")
    print("Domain-truncated outer prediction sets generated: yes")
    print("Unbounded prediction sets claimed: no")
    print("Coverage evaluated: no")
    print("M3 implemented: no")
    print("M4 implemented: no")
    print("M6 implemented: no")
    print("Production experiments run: no")


if __name__ == "__main__":
    main()
