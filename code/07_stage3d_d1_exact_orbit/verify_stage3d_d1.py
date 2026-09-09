from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.special import betaln, logsumexp, ndtr

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs_stage3d_d1"
EXPECTED_ENV = {
    "python": "3.10",
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "PyYAML": "6.0.3",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(name: str):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT / name
    if name.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return pd.read_csv(handle)
    return pd.read_csv(path)


def source_hashes_current() -> dict[str, str]:
    roots = [ROOT / "configs", ROOT / "docs", ROOT / "src", ROOT / "tests"]
    files = [
        ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "requirements_py310.txt",
        ROOT / "RUN_STAGE3D_D1.bat", ROOT / "RUN_STAGE3D_D1.ps1",
        ROOT / "stage3d_d1_run.py", ROOT / "verify_stage3d_d1.py", ROOT / "package_stage3d_outputs.py",
    ]
    result = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix not in {".pyc", ".pyo"} and "__pycache__" not in path.parts:
                result[path.relative_to(ROOT).as_posix()] = sha256(path)
    for path in files:
        result[path.relative_to(ROOT).as_posix()] = sha256(path)
    return dict(sorted(result.items()))


def check_environment(allow_development_runtime: bool) -> tuple[bool, dict[str, str]]:
    observed = {
        "python": ".".join(platform.python_version().split(".")[:2]),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "PyYAML": yaml.__version__,
    }
    exact = observed == EXPECTED_ENV
    if not exact and not allow_development_runtime:
        raise RuntimeError(f"Exact Stage3D environment mismatch: observed={observed}, expected={EXPECTED_ENV}")
    return exact, observed


def add(checks: dict[str, str], name: str, condition: bool) -> None:
    checks[name] = "pass" if bool(condition) else "fail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-runtime-override", action="store_true")
    args = parser.parse_args()

    exact_environment, observed_environment = check_environment(args.development_runtime_override)
    checks: dict[str, str] = {}

    required = {
        "stage3d_d1_contract_summary.json",
        "stage3d_d1_counts.json",
        "stage3d_d1_input_audit.json",
        "stage3d_d1_source_hashes.json",
        "stage3d_d1_environment_inventory.json",
        "stage3d_d1_fixture_registry.csv",
        "stage3d_d1_orbit_state_audit.csv.gz",
        "stage3d_d1_probability_audit.csv",
        "stage3d_d1_scalar_vectorized_audit.csv",
        "stage3d_d1_full_local_cancellation_audit.csv",
        "stage3d_d1_mixed_measure_normalization_audit.csv",
        "stage3d_d1_boundary_edge_audit.csv",
        "stage3d_d1_endpoint_atom_audit.csv",
        "stage3d_d1_duplicate_orbit_audit.csv",
        "stage3d_d1_treatment_factor_spot_audit.csv",
        "stage3d_d1_target_tilt_audit.csv",
        "stage3d_d1_global_positivity_audit.csv",
        "stage3d_d1_full_target_law_audit.csv",
        "stage3d_d1_full_local_state_audit.csv.gz",
        "stage3d_d1_refusal_audit.csv",
        "stage3d_d1_candidate_algebra_audit.json",
        "stage3d_d1_candidate_quotient_audit.json",
        "stage3d_d1_jacobian_audit.json",
        "stage3d_d1_runtime.csv",
        "stage3d_d1_manifest.json",
    }
    add(checks, "required_outputs_present", all((OUTPUT / name).is_file() for name in required))

    contract = read_json("stage3d_d1_contract_summary.json")
    counts = read_json("stage3d_d1_counts.json")
    manifest = read_json("stage3d_d1_manifest.json")
    input_audit = read_json("stage3d_d1_input_audit.json")
    source_record = read_json("stage3d_d1_source_hashes.json")
    candidate_audit = read_json("stage3d_d1_candidate_algebra_audit.json")
    candidate_quotient_audit = read_json("stage3d_d1_candidate_quotient_audit.json")
    jacobian_audit = read_json("stage3d_d1_jacobian_audit.json")
    fixtures = read_csv("stage3d_d1_fixture_registry.csv")
    states = read_csv("stage3d_d1_orbit_state_audit.csv.gz")
    probabilities = read_csv("stage3d_d1_probability_audit.csv")
    comparisons = read_csv("stage3d_d1_scalar_vectorized_audit.csv")
    cancellation = read_csv("stage3d_d1_full_local_cancellation_audit.csv")
    normalization = read_csv("stage3d_d1_mixed_measure_normalization_audit.csv")
    boundaries = read_csv("stage3d_d1_boundary_edge_audit.csv")
    endpoints = read_csv("stage3d_d1_endpoint_atom_audit.csv")
    duplicate = read_csv("stage3d_d1_duplicate_orbit_audit.csv")
    treatment_spot = read_csv("stage3d_d1_treatment_factor_spot_audit.csv")
    target_tilt = read_csv("stage3d_d1_target_tilt_audit.csv")
    global_positivity = read_csv("stage3d_d1_global_positivity_audit.csv")
    full_target_law = read_csv("stage3d_d1_full_target_law_audit.csv")
    full_local_detail = read_csv("stage3d_d1_full_local_state_audit.csv.gz")
    refusal = read_csv("stage3d_d1_refusal_audit.csv")
    runtime = read_csv("stage3d_d1_runtime.csv")

    add(checks, "stage_and_gate", contract["stage"] == "3D" and contract["gate"] == "D0_D1")
    add(checks, "G3_not_implemented", contract["G3_implemented"] is False)
    add(checks, "prediction_sets_not_generated", contract["prediction_sets_generated"] is False)
    add(checks, "coverage_not_evaluated", contract["coverage_evaluated"] is False)
    add(checks, "M3_M4_M6_deferred", not contract["M3_implemented"] and not contract["M4_implemented"] and not contract["M6_implemented"])
    add(checks, "no_dense_precision_inverse_claim", contract["dense_precision_inverse_formed"] is False)
    add(checks, "oracle_nuisance_algebra_only", contract.get("nuisance_regime") == "oracle_generator_truth_algebra_only")
    add(checks, "numeric_payload_equivalence", contract.get("payload_equivalence") == "exact numerical pair (A,Y_candidate); source origin excluded")

    add(checks, "evaluation_count", len(fixtures) == 7 and counts["evaluation_count"] == 7)
    expected_counts = {
        "G6_GAUSS_INTERIOR_TRUTH": 720,
        "G6_GAUSS_INTERIOR_SHIFTED": 720,
        "G8_GAUSS_INTERIOR_TRUTH": 40320,
        "G6_GAUSS_DUPLICATE_TRUTH": 360,
        "G6_GAUSS_ENDPOINT0": 720,
        "G6_GAUSS_ENDPOINT1": 720,
        "G6_NONGAUSSIAN_INTERIOR_TRUTH": 720,
    }
    observed_counts = fixtures.set_index("evaluation_id")["distinct_states"].astype(int).to_dict()
    add(checks, "exact_orbit_counts", observed_counts == expected_counts)
    add(checks, "total_state_rows", len(states) == sum(expected_counts.values()) == counts["total_orbit_state_rows"])
    add(checks, "unique_state_keys", not states.duplicated(["evaluation_id", "state_index"]).any())
    identity_ok = True
    for evaluation_id, group in states.groupby("evaluation_id"):
        row = group.loc[group["state_index"] == 0]
        if len(row) != 1:
            identity_ok = False
            break
        block_size = int(fixtures.loc[fixtures["evaluation_id"] == evaluation_id, "block_size"].iloc[0])
        if str(row.iloc[0]["assignment"]) != ",".join(str(i) for i in range(block_size)):
            identity_ok = False
            break
    add(checks, "identity_state_first", identity_ok)

    probability_sum_ok = np.allclose(probabilities["probability_sum"], 1.0, atol=5e-12, rtol=0.0)
    add(checks, "probability_summary_sums", probability_sum_ok)
    state_prob_ok = True
    logweight_identity_ok = True
    logshift_invariance_ok = True
    for evaluation_id, group in states.groupby("evaluation_id", sort=False):
        p = group["normalized_probability"].to_numpy(dtype=float)
        log_t = group["log_treatment_factor"].to_numpy(dtype=float)
        log_j = group["log_outcome_jacobian_factor"].to_numpy(dtype=float)
        log_r = group["log_residual_factor"].to_numpy(dtype=float)
        log_w = group["log_total_weight"].to_numpy(dtype=float)
        finite_components = np.isfinite(log_t) & np.isfinite(log_j) & np.isfinite(log_r)
        if not np.allclose(log_w[finite_components], (log_t + log_j + log_r)[finite_components], atol=2e-12, rtol=0.0):
            logweight_identity_ok = False
        finite = np.isfinite(log_w)
        recomputed = np.zeros(len(log_w))
        recomputed[finite] = np.exp(log_w[finite] - logsumexp(log_w[finite]))
        if not np.allclose(p, recomputed, atol=2e-12, rtol=0.0):
            state_prob_ok = False
        shifted = np.zeros(len(log_w))
        shifted[finite] = np.exp((log_w[finite] + 17.3) - logsumexp(log_w[finite] + 17.3))
        if not np.allclose(p, shifted, atol=2e-12, rtol=0.0):
            logshift_invariance_ok = False
    add(checks, "state_probability_recomputation", state_prob_ok)
    add(checks, "logweight_component_identity", logweight_identity_ok)
    add(checks, "logweight_constant_invariance", logshift_invariance_ok)
    add(checks, "probabilities_nonnegative_finite", np.isfinite(states["normalized_probability"]).all() and (states["normalized_probability"] >= 0.0).all())

    add(checks, "scalar_vectorized_treatment", float(comparisons["max_abs_log_treatment_difference"].max()) <= 2e-10)
    add(checks, "scalar_vectorized_jacobian", float(comparisons["max_abs_log_outcome_jacobian_difference"].max()) <= 2e-10)
    add(checks, "scalar_vectorized_residual", float(comparisons["max_abs_log_residual_difference"].max()) <= 2e-10)
    add(checks, "scalar_vectorized_total", float(comparisons["max_abs_log_total_difference"].max()) <= 2e-10)
    add(checks, "full_local_cancellation", float(cancellation["full_minus_local_constant_range"].max()) <= 2e-9)
    add(checks, "full_target_law_probability_match", len(full_target_law) >= 1 and float(full_target_law["maximum_absolute_probability_difference"].max()) <= 2e-11)
    add(checks, "full_target_law_normalization", np.allclose(full_target_law["full_probability_sum"], 1.0, atol=5e-12, rtol=0.0))
    detail_ok = True
    for evaluation_id, group in full_local_detail.groupby("evaluation_id"):
        observed_range = float(group["full_minus_local"].max() - group["full_minus_local"].min())
        expected_range = float(cancellation.loc[cancellation["evaluation_id"] == evaluation_id, "full_minus_local_constant_range"].iloc[0])
        if abs(observed_range - expected_range) > 2e-12:
            detail_ok = False
    add(checks, "full_local_state_details", detail_ok and len(full_local_detail) == int(cancellation["audited_state_count"].sum()))
    add(checks, "mixed_measure_normalization", float(normalization["absolute_error"].max()) <= 2e-8)
    add(checks, "positive_precision", float(probabilities["minimum_precision_eigenvalue"].min()) > 1e-10)
    add(checks, "boundary_edges_present", len(boundaries) > 0 and (boundaries["boundary_node_count"] > 0).all())

    endpoint_ok = True
    for row in endpoints.itertuples(index=False):
        if int(row.states_with_matching_endpoint_at_target_slot) != int(row.positive_probability_states):
            endpoint_ok = False
        group = states.loc[states["evaluation_id"] == row.evaluation_id]
        expected = np.isclose(group["target_slot_A"].astype(float), float(row.target_dose), atol=0.0, rtol=0.0)
        observed = group["normalized_probability"].astype(float).to_numpy() > 0.0
        if not np.array_equal(np.asarray(expected, dtype=bool), observed):
            endpoint_ok = False
    add(checks, "endpoint_atom_exact_support", endpoint_ok)
    interior_group = states.loc[states["evaluation_id"] == "G6_GAUSS_INTERIOR_TRUTH"]
    add(checks, "interior_tilt_endpoint_zero", (interior_group.loc[interior_group["target_slot_A"].isin([0.0, 1.0]), "normalized_probability"] == 0.0).all())

    add(checks, "duplicate_quotient_count", len(duplicate) == 1 and int(duplicate.iloc[0]["distinct_states"]) == 360 and int(duplicate.iloc[0]["full_factorial_states"]) == 720)
    add(checks, "candidate_treatment_invariant", float(candidate_audit["max_abs_log_treatment_change"]) == 0.0)
    add(checks, "candidate_jacobian_invariant", float(candidate_audit["max_abs_log_outcome_jacobian_change"]) == 0.0)
    add(checks, "candidate_residual_changes", float(candidate_audit["max_abs_log_residual_change"]) > 0.0)
    add(checks, "candidate_replacement_rule", candidate_audit["rule"] == "candidate_y_replaces_only_the_augmented_source_response_before_candidate_specific_quotienting")
    add(checks, "candidate_origin_excluded_from_equivalence", candidate_audit.get("candidate_origin_part_of_payload_equivalence") is False)
    add(checks, "candidate_specific_quotient", candidate_quotient_audit["noncollision_distinct_states"] == 6 and candidate_quotient_audit["target_calibration_collision_distinct_states"] == 3 and candidate_quotient_audit["source_origin_part_of_equivalence"] is False)
    add(checks, "inverse_jacobian_direction", float(jacobian_audit["absolute_error"]) <= 1e-12 and jacobian_audit["direction"] == "inverse_outcome_jacobian_product_1_over_scale")
    add(checks, "non_gaussian_route_present", "G6_NONGAUSSIAN_INTERIOR_TRUTH" in set(fixtures["evaluation_id"]) and np.isclose(fixtures.loc[fixtures["evaluation_id"] == "G6_NONGAUSSIAN_INTERIOR_TRUTH", "transform_power"].iloc[0], 1.5))
    add(checks, "refusal_paths", len(refusal) == 5 and refusal["test_passed"].astype(bool).all() and "D1_POSITIVITY_FAILURE" in set(refusal["refusal_code"]))

    # Treatment-factor spot traces must sum to the archived state factor.
    spot_sum_ok = True
    for (evaluation_id, state_index), group in treatment_spot.groupby(["evaluation_id", "state_index"]):
        values = group["log_density_or_mass"].to_numpy(dtype=float)
        observed_sum = -np.inf if np.any(np.isneginf(values)) else float(np.sum(values))
        archived = float(states.loc[(states["evaluation_id"] == evaluation_id) & (states["state_index"] == state_index), "log_treatment_factor"].iloc[0])
        if (np.isneginf(observed_sum) and np.isneginf(archived)):
            continue
        if not np.isclose(observed_sum, archived, atol=2e-12, rtol=0.0):
            spot_sum_ok = False
    add(checks, "treatment_spot_sums", spot_sum_ok)

    # Independently reconstruct each archived spot density from accepted Stage3B
    # slot parameters and the frozen target intervention, without importing the
    # Stage3D treatment module.
    with zipfile.ZipFile(ROOT / "inputs" / "upstream" / "GeoDose_Stage3B_OUTPUTS.zip") as zf:
        units = pd.read_csv(gzip.GzipFile(fileobj=io.BytesIO(zf.read("stage3b_validation_units.csv.gz"))))
        fixture_json_independent = json.loads(zf.read("stage3b_exact_orbit_fixtures.json").decode("utf-8"))
    fixture_lookup = fixtures.set_index("evaluation_id")
    treatment_recompute_ok = True
    for row in treatment_spot.itertuples(index=False):
        meta = fixture_lookup.loc[row.evaluation_id]
        base_name = str(meta.fixture_source).replace("_derived", "")
        raw_fixture = fixture_json_independent[base_name]
        node = int(raw_fixture["fixed_slots"][int(row.slot_position)]["node_index"])
        unit = units.loc[(units["case_id"] == meta.case_id) & (units["node_index"] == node)]
        if len(unit) != 1:
            treatment_recompute_ok = False
            continue
        unit = unit.iloc[0]
        a = float(row.payload_A)
        if row.factor_type == "target_intervention_q":
            dose = float(meta.target_dose)
            if dose in {0.0, 1.0}:
                density = 1.0 if a == dose else 0.0
            else:
                h = float(meta.bandwidth)
                if not (0.0 < a < 1.0):
                    density = 0.0
                else:
                    z_norm = float(ndtr((1.0 - dose) / h) - ndtr((0.0 - dose) / h))
                    density = float(np.exp(-0.5 * ((a - dose) / h) ** 2) / (h * np.sqrt(2.0 * np.pi) * z_norm))
        else:
            if a == 0.0:
                density = float(unit["pi_atom_0"])
            elif a == 1.0:
                density = float(unit["pi_atom_1"])
            else:
                alpha = float(unit["beta_alpha"])
                beta = float(unit["beta_beta"])
                beta_log = (alpha - 1.0) * np.log(a) + (beta - 1.0) * np.log1p(-a) - betaln(alpha, beta)
                density = float(unit["pi_interior"] * np.exp(beta_log))
        if not np.isclose(density, float(row.density_or_mass), atol=2e-13, rtol=2e-13):
            treatment_recompute_ok = False
    add(checks, "treatment_factor_independent_recompute", treatment_recompute_ok)
    add(checks, "target_treatment_cancellation_trace", set(treatment_spot.loc[treatment_spot["slot_position"] == treatment_spot["evaluation_id"].map(fixture_lookup["target_slot_position"]), "factor_type"]) == {"target_intervention_q"})
    add(checks, "target_tilt_positivity", target_tilt["positivity_satisfied"].astype(bool).all())
    add(checks, "target_tilt_cancellation", float(target_tilt["absolute_error"].max()) <= 2e-14)
    add(checks, "target_tilt_required_rows_have_positive_g", (target_tilt.loc[target_tilt["positivity_required"].astype(bool), "target_observational_g"] > 0.0).all())
    add(checks, "complete_intervention_support_positivity", len(global_positivity) == 7 and global_positivity["complete_support_positivity_satisfied"].astype(bool).all())
    add(checks, "interior_support_component_positive", (global_positivity.loc[global_positivity["intervention_support"] == "open_interval_0_1", "target_slot_pi_interior"] > 0.0).all())
    add(checks, "endpoint_support_component_positive", (global_positivity.loc[global_positivity["intervention_support"] == "atom_0", "target_slot_pi_atom_0"] > 0.0).all() and (global_positivity.loc[global_positivity["intervention_support"] == "atom_1", "target_slot_pi_atom_1"] > 0.0).all())

    add(checks, "target_payload_position_valid", ((states["target_payload_position"] >= 0) & (states["target_payload_position"] < states["evaluation_id"].map(fixtures.set_index("evaluation_id")["block_size"]))).all())
    add(checks, "target_slot_payload_index_valid", ((states["target_slot_payload_index"] >= 0) & (states["target_slot_payload_index"] < states["evaluation_id"].map(fixtures.set_index("evaluation_id")["block_size"]))).all())

    # Upstream hash audit.
    expected_input_hashes = json.loads((ROOT / "configs" / "input_hashes.json").read_text(encoding="utf-8"))
    add(checks, "input_audit_names", set(input_audit) == set(expected_input_hashes))
    add(checks, "input_audit_hashes", all(input_audit[name]["observed_sha256"] == expected_input_hashes[name] for name in expected_input_hashes))

    current_source = source_hashes_current()
    add(checks, "source_file_hashes", source_record["files"] == current_source)
    canonical = json.dumps(current_source, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    add(checks, "source_tree_hash", hashlib.sha256(canonical).hexdigest() == source_record["tree_sha256"] == manifest["source_tree_sha256"])

    manifest_ok = True
    for name, meta in manifest["outputs"].items():
        path = OUTPUT / name
        if not path.is_file() or path.stat().st_size != int(meta["size_bytes"]) or sha256(path) != meta["sha256"]:
            manifest_ok = False
            break
    add(checks, "output_manifest_hashes", manifest_ok)
    add(checks, "manifest_scope", manifest["stage"] == "3D_D0_D1" and manifest["production_experiments_run"] is False)
    add(checks, "runtime_rows", len(runtime) == 7 and (runtime["seconds"] >= 0.0).all())

    # Scan source for prohibited exact-branch shortcuts.
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((ROOT / "src").rglob("*.py")))
    add(checks, "no_dense_inverse", "np.linalg.inv(" not in source_text and "numpy.linalg.inv(" not in source_text and "scipy.linalg.inv(" not in source_text)
    add(checks, "no_network_code", "requests." not in source_text and "urllib.request" not in source_text and "socket." not in source_text)
    add(checks, "no_unsafe_deserialization", "pickle.load" not in source_text and "eval(" not in source_text and "exec(" not in source_text)

    # Stage 3B fixture source remains exact and verified.
    with zipfile.ZipFile(ROOT / "inputs" / "upstream" / "GeoDose_Stage3B_OUTPUTS.zip") as zf:
        stage3b_verification = json.loads(zf.read("STAGE3B_VERIFICATION.json").decode("utf-8"))
        fixture_json = json.loads(zf.read("stage3b_exact_orbit_fixtures.json").decode("utf-8"))
    add(checks, "stage3b_verified", stage3b_verification["status"] == "verified_complete")
    add(checks, "stage3b_fixture_contract", fixture_json["candidate_inversion_rule"].startswith("replace target response by candidate y"))
    candidate_contract = yaml.safe_load((ROOT / "configs" / "stage3d_candidate_inversion_contract.yaml").read_text(encoding="utf-8"))
    add(checks, "candidate_domain_frozen_before_calibration", candidate_contract["candidate_domain"]["calibration_outcomes_allowed"] is False and candidate_contract["candidate_domain"]["data_allowed"] == "independent_nuisance_training_outcomes_only")
    orbit_contract = yaml.safe_load((ROOT / "configs" / "stage3d_orbit_contract.yaml").read_text(encoding="utf-8"))
    add(checks, "orbit_contract_candidate_specific", orbit_contract["candidate_specific_orbit"] is True and orbit_contract["candidate_origin_part_of_payload_equivalence"] is False)

    add(checks, "exact_environment", exact_environment or args.development_runtime_override)
    add(checks, "production_not_run", counts["production_experiments_run"] is False)

    failed = [name for name, value in checks.items() if value != "pass"]
    status = "verified_complete" if not failed else "failed"
    record = {
        "stage": "3D_D0_D1",
        "script_version": contract["script_version"],
        "status": status,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "exact_environment": exact_environment,
        "development_runtime_override": bool(args.development_runtime_override),
        "observed_environment": observed_environment,
        "required_environment": EXPECTED_ENV,
        "evaluation_count": len(fixtures),
        "orbit_state_rows": len(states),
        "G3_implemented": False,
        "prediction_sets_generated": False,
        "coverage_evaluated": False,
        "production_experiments_run": False,
    }
    (OUTPUT / "STAGE3D_D1_VERIFICATION.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"Stage 3D D1 verification failed: {failed}")
    print("STAGE 3D D0/D1 EXACT-ORBIT ALGEBRA VERIFIED COMPLETE")
    print(f"Verification checks: {len(checks)}/{len(checks)}")
    print(f"Evaluations: {len(fixtures)}")
    print(f"Orbit state rows: {len(states):,}")
    print("G3 implemented: no")
    print("Prediction sets generated: no")
    print("Coverage evaluated: no")
    print("Production experiments run: no")


if __name__ == "__main__":
    main()
