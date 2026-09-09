from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geodose_stage3a.core import *


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="inputs")
    parser.add_argument("--output", default="outputs_stage3a")
    args = parser.parse_args()

    input_root = (PACKAGE_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input).resolve()
    output = (PACKAGE_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()

    required = [
        "stage3_registry.yaml",
        "scenario_registry.csv",
        "method_registry.csv",
        "factor_registry.csv",
        "nuisance_model_registry.csv",
        "ablation_registry.csv",
        "metric_registry.csv",
        "reduction_test_registry.csv",
        "additional_stress_test_registry.csv",
        "mandatory_gate_registry.csv",
        "reporting_contract_registry.csv",
        "source_alignment_matrix.csv",
        "deferred_feature_registry.csv",
        "residual_law_registry.csv",
        "seed_registry.csv",
        "mine_role_registry.csv",
        "feature_registry.csv",
        "target_registry.csv",
        "assumption_registry.csv",
        "proof_test_registry.csv",
        "refusal_registry.json",
        "target_draw_contract.json",
        "target_population_contract.json",
        "stage3a_environment_inventory.json",
        "data_contract.json",
        "input_manifest.json",
        "stage3_prescreen_report.md",
        "stage3a_generator_preview.csv.gz",
        "stage3a_generator_truth.csv.gz",
        "stage3a_localized_target_truth.csv.gz",
        "stage3a_preview_graph_edges.csv",
        "stage3a_generator_diagnostics.json",
        "stage3a_manifest.json",
    ]
    for name in required:
        require((output / name).exists(), f"Missing Stage 3A output: {name}")

    input_manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    require(input_manifest["stage2b_archive_sha256"] == EXPECTED_STAGE2B_ARCHIVE_SHA256, "Stage 2B archive hash is not the accepted final hash")
    for relative, info in input_manifest["files"].items():
        require(sha256_file(input_root / relative) == info["sha256"], f"Frozen input checksum changed: {relative}")

    registry = yaml.safe_load((output / "stage3_registry.yaml").read_text(encoding="utf-8"))
    require(registry["status"]["architecture_frozen"] is True, "Architecture not frozen")
    require(registry["status"]["methods_implemented"] is False, "Stage 3A must not claim method implementation")
    require(registry["scientific_scope"]["primary_estimand"] == "localized stochastic potential outcome under a supported mixed intervention", "Primary target misregistered")

    scenarios = pd.read_csv(output / "scenario_registry.csv", encoding="utf-8-sig")
    methods = pd.read_csv(output / "method_registry.csv", encoding="utf-8-sig")
    factors = pd.read_csv(output / "factor_registry.csv", encoding="utf-8-sig")
    nuisance_models = pd.read_csv(output / "nuisance_model_registry.csv", encoding="utf-8-sig")
    ablations = pd.read_csv(output / "ablation_registry.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(output / "metric_registry.csv", encoding="utf-8-sig")
    reductions = pd.read_csv(output / "reduction_test_registry.csv", encoding="utf-8-sig")
    stress_tests = pd.read_csv(output / "additional_stress_test_registry.csv", encoding="utf-8-sig")
    mandatory_gates = pd.read_csv(output / "mandatory_gate_registry.csv", encoding="utf-8-sig")
    reporting_contract = pd.read_csv(output / "reporting_contract_registry.csv", encoding="utf-8-sig")
    alignment = pd.read_csv(output / "source_alignment_matrix.csv", encoding="utf-8-sig")
    deferred_features = pd.read_csv(output / "deferred_feature_registry.csv", encoding="utf-8-sig")
    residual_laws = pd.read_csv(output / "residual_law_registry.csv", encoding="utf-8-sig")
    seeds = pd.read_csv(output / "seed_registry.csv", encoding="utf-8-sig")
    roles = pd.read_csv(output / "mine_role_registry.csv", encoding="utf-8-sig")
    features = pd.read_csv(output / "feature_registry.csv", encoding="utf-8-sig")
    targets = pd.read_csv(output / "target_registry.csv", encoding="utf-8-sig")
    assumptions = pd.read_csv(output / "assumption_registry.csv", encoding="utf-8-sig")
    proof_tests = pd.read_csv(output / "proof_test_registry.csv", encoding="utf-8-sig")

    require(scenarios.scenario_id.tolist() == [f"S{i}" for i in range(1, 11)], "Scenario registry mismatch")
    require(methods.method_id.tolist() == [f"M{i}" for i in range(1, 7)], "Method registry mismatch")
    require(factors.factor_id.tolist() == [f"F{i:02d}" for i in range(1, 13)], "Factor registry mismatch")
    require(nuisance_models.model_id.tolist() == [f"NUI{i}" for i in range(1, 7)], "Nuisance model registry mismatch")
    require(ablations.ablation_id.tolist() == [f"ABL{i}" for i in range(0, 10)], "Ablation registry mismatch")
    require(metrics.metric_id.tolist() == [f"MET{i:02d}" for i in range(1, 24)], "Metric registry mismatch")
    require(reductions.reduction_id.tolist() == [f"R{i}" for i in range(1, 5)], "Reduction registry mismatch")
    require(stress_tests.stress_id.tolist() == ["ST1", "ST2", "ST3"], "Additional stress-test registry mismatch")
    require(set(stress_tests.stress_name) == {"mixed_endpoint_atoms", "repeated_mine_year_temporal_dependence", "spatial_confounding_C2_negative_control"}, "Required additional stress tests are incomplete")
    require(mandatory_gates.gate_id.tolist() == [f"GATE{i:02d}" for i in range(1, 18)], "Mandatory-gate registry mismatch")
    require(len(reporting_contract) == 16 and reporting_contract.report_id.is_unique, "Reporting contract registry mismatch")
    require(len(alignment) == 17 and alignment.status.str.startswith("aligned").all(), "Source alignment matrix incomplete")
    require(len(deferred_features) == 4 and deferred_features.availability_status.eq("deferred_minimum_route").all(), "Deferred context registry mismatch")
    require(residual_laws.residual_law_id.tolist() == ["RL1", "RL2", "RL3", "RL4"], "Residual-law registry mismatch")
    require((residual_laws.residual_law_name == "monotone_transformed_gmrf").any(), "Non-Gaussian G2 validation route is missing")
    require(len(seeds) == 3200, "Seed registry should contain 200 pilot + 3000 provisional production rows")
    require(not seeds[["phase", "scenario_id", "replication"]].duplicated().any(), "Duplicate seed registry key")
    seed_columns = ["data_seed", "split_seed", "nuisance_seed", "orbit_seed", "measurement_seed", "target_draw_seed"]
    require(set(seed_columns).issubset(seeds.columns), "Dedicated target-draw seed stream missing")
    require(seeds[seed_columns].notna().all().all(), "Missing seed values")
    require((seeds[seed_columns].nunique(axis=1) == len(seed_columns)).all(), "Seed streams collide within a replication")
    require(seeds[seed_columns].stack().nunique() == len(seeds) * len(seed_columns), "Seed values collide across replications or streams")
    require(len(roles) == 25, "Mine role registry row count mismatch")
    splits = pd.read_csv(input_root / "stage2a/leave_one_mine_out_splits.csv", encoding="utf-8-sig")
    held = splits[["fold_id", "held_out_MineID"]].drop_duplicates().set_index("fold_id")["held_out_MineID"].to_dict()
    for fold, group in roles.groupby("fold_id"):
        counts = group.role.value_counts().to_dict()
        require(counts == {"nuisance_training": 2, "test_target": 1, "support_audit": 1, "calibration": 1}, f"Invalid roles in {fold}: {counts}")
        require(group.MineID.nunique() == 5, f"Mine role duplication in {fold}")
        require(group.loc[group.role == "test_target", "MineID"].iloc[0] == held[fold], f"Test target does not match held-out mine in {fold}")

    require(set(assumptions.assumption_id) == {"C1", "C2", "C3", "C4", "C5", "F1"}, "Causal assumption registry mismatch")
    require(proof_tests.test_id.tolist() == [f"U{i}" for i in range(1, 19)], "Proof test registry mismatch")
    require(proof_tests.loc[proof_tests.test_id == "U16", "status"].iloc[0] == "implemented_preview_numeric_check", "U16 implementation status mismatch")

    refusals = json.loads((output / "refusal_registry.json").read_text(encoding="utf-8"))
    refusal_codes = {row["code"] for row in refusals}
    required_refusals = {
        "R01_UNSUPPORTED_DOSE", "R02_ENDPOINT_NOT_AUDITED", "R06_SPATIAL_PRECISION_NOT_PD",
        "R07_EMPTY_NUISANCE_CERTIFICATE", "R08_CERTIFIED_DEFICIT_VACUOUS",
        "R09_ORBIT_PROPOSAL_NO_SUPPORT", "R11_EO_QUALITY_INSUFFICIENT",
        "R13_EXACT_TREATMENT_LIKELIHOOD_UNAVAILABLE", "R14_NUISANCE_THEOREM_REGIME_MISMATCH",
        "R15_VECCHIA_CONDITIONS_UNVERIFIED", "R16_DESIGN_RULE_NOT_FROZEN",
        "R17_TARGET_UNIT_OUTSIDE_SUPPORTED_POPULATION",
    }
    require(required_refusals.issubset(refusal_codes), "Mandatory theorem refusal codes are incomplete")
    require(set(mandatory_gates.refusal_code).issubset(refusal_codes), "Mandatory gate references an unknown refusal code")
    required_ablation_names = {
        "without_treatment_correction", "without_target_design_transport", "without_spatial_law",
        "without_eligibility_refusal", "geographic_distance_heuristic", "oracle_treatment", "oracle_spatial_law",
    }
    require(required_ablation_names.issubset(set(ablations.ablation_name)), "Mandatory ablations are incomplete")
    required_metrics = {
        "marginal_coverage", "absolute_calibration_error", "mean_width_at_matched_coverage",
        "weighted_interval_score", "theorem_eligibility_rate", "refusal_rate", "false_support_rate",
        "empirical_semivariogram", "morans_I", "spatial_leakage_optimism",
    }
    require(required_metrics.issubset(set(metrics.metric_name)), "Mandatory metrics/diagnostics are incomplete")

    mapped = features.loc[features.feature_name == "mapped_rehabilitation_fraction"].iloc[0]
    require(not bool(mapped.allowed_in_synthetic_assignment), "Snapshot mapped fraction must not enter synthetic assignment")
    require(not bool(mapped.allowed_in_fitted_treatment_model), "Snapshot mapped fraction must not enter fitted treatment model")
    require(not bool(mapped.allowed_as_outcome_model_predictor), "Snapshot mapped fraction must not enter outcome predictor")
    require(bool(mapped.allowed_in_support_stratification), "Snapshot mapped fraction should remain available for support stratification")

    post_quality_names = {
        "ue_2025", "valid_observation_count_2025", "valid_pixel_fraction_2025",
        "water_fraction_2025", "unclear_fraction_2025",
        "high_ue_fraction_20_2025", "high_ue_fraction_25_2025", "high_ue_fraction_30_2025",
    }
    post_quality = features.loc[features.feature_name.isin(post_quality_names)]
    require(set(post_quality.feature_name) == post_quality_names, "Post-assignment 2025 quality registry is incomplete")
    require((~post_quality.allowed_in_synthetic_assignment.astype(bool)).all(), "2025 quality leaked into assignment")
    require((~post_quality.allowed_in_fitted_treatment_model.astype(bool)).all(), "2025 quality leaked into propensity")
    require((~post_quality.allowed_as_outcome_model_predictor.astype(bool)).all(), "2025 quality leaked into outcome predictor")
    require(post_quality.allowed_in_measurement_error_mechanism.astype(bool).all(), "2025 quality is not registered for S9 measurement error")
    require(post_quality.allowed_in_outcome_quality_diagnostics.astype(bool).all(), "2025 quality is not registered for diagnostics")

    require(set(targets.estimand_type) == {"localized_stochastic_potential_outcome"}, "Target registry estimand mismatch")
    interior_targets = targets.loc[~targets.endpoint_atom.astype(bool)]
    require(interior_targets.bandwidth_primary.eq(PRIMARY_BANDWIDTH).all(), "Interior target bandwidth mismatch")
    endpoint_targets = targets.loc[targets.endpoint_atom.astype(bool)]
    require(endpoint_targets.intervention_type.eq("audited_endpoint_atom").all(), "Endpoint intervention mismatch")
    require(endpoint_targets.endpoint_audit_scope.str.contains("synthetic treatment mechanism", regex=False).all(), "Endpoint audit scope is ambiguous")
    require(targets.target_object.str.contains("random Y_i(A_star)", regex=False).all(), "Primary random target object is not explicit")
    draw_contract = json.loads((output / "target_draw_contract.json").read_text(encoding="utf-8"))
    require(draw_contract["stage3a_truth_artifact"] == "analytic conditional mean functional only", "Stage 3A truth scope is ambiguous")
    require("realized A_star" in draw_contract["coverage_truth_required_in_stage3b"], "Stage 3B target draws are not mandated")
    target_population = json.loads((output / "target_population_contract.json").read_text(encoding="utf-8"))
    require("held-out mine" in target_population["minedosebench_target_population"], "MineDoseBench target population is not frozen")
    require("post-assignment EO quality" in target_population["eo_quality_gate"], "EO quality gate/causal eligibility separation missing")
    environment = json.loads((output / "stage3a_environment_inventory.json").read_text(encoding="utf-8"))
    require(environment["internet_used"] is False and environment["production_inference_run"] is False, "Stage 3A environment scope mismatch")
    require(environment["packages"]["numpy"] is not None and environment["packages"]["pandas"] is not None, "Environment versions missing")

    preview = pd.read_csv(output / "stage3a_generator_preview.csv.gz", compression="gzip")
    hard_truth = pd.read_csv(output / "stage3a_generator_truth.csv.gz", compression="gzip")
    localized_truth = pd.read_csv(output / "stage3a_localized_target_truth.csv.gz", compression="gzip")
    edges = pd.read_csv(output / "stage3a_preview_graph_edges.csv", encoding="utf-8-sig")
    diagnostics = json.loads((output / "stage3a_generator_diagnostics.json").read_text(encoding="utf-8"))

    require(len(preview) == 625 and preview.unit_id.is_unique, "Generator preview size/key failure")
    require(len(edges) == 2352 and not edges.duplicated().any(), "Preview queen graph failure")
    require(len(hard_truth) == 4375 and set(hard_truth.dose.unique()) == set(PRIMARY_DOSE_GRID), "Hard truth table failure")
    require(len(localized_truth) == 4375 and set(localized_truth.target_dose.unique()) == set(PRIMARY_DOSE_GRID), "Localized truth table failure")
    require(set(localized_truth.target_object) == {"random_localized_stochastic_potential_outcome"}, "Localized target object mismatch")
    require(set(localized_truth.artifact_scope) == {"analytic_conditional_mean_functional_only"}, "Localized truth artifact overclaims coverage truth")
    require(set(localized_truth.coverage_truth_status) == {"realized_target_draws_deferred_to_stage3b"}, "Stage 3B draw requirement missing")
    require((preview.loc[preview.treatment_category == "atom_0", "A"] == 0).all(), "Zero atoms altered")
    require((preview.loc[preview.treatment_category == "atom_1", "A"] == 1).all(), "One atoms altered")
    interior = preview.loc[preview.treatment_category == "interior", "A"]
    require(interior.between(0, 1, inclusive="neither").all(), "Interior treatment outside open unit interval")
    require(preview.treatment_category.value_counts().get("atom_0", 0) > 0, "Zero endpoint atom absent")
    require(preview.treatment_category.value_counts().get("atom_1", 0) > 0, "One endpoint atom absent")
    require(np.isfinite(preview.oracle_g_mixed_at_A).all() and (preview.oracle_g_mixed_at_A > 0).all(), "Oracle mixed density invalid")
    require(diagnostics["mixed_treatment_normalization_max_abs_error"] < 1e-6, "Mixed treatment normalization failed")
    require(min(diagnostics["precision_min_eigenvalues"].values()) > 0, "Generator precision is not positive definite")
    require(diagnostics["realization_dependent_centering_or_scaling"] is False, "GMRF realization was normalized after drawing")
    require(diagnostics["dense_inverse_formed"] is False, "Dense inverse rule violated")
    require(diagnostics["realized_target_draws_required_before_coverage"] is True, "Coverage draw gate missing")

    # Deterministic replay: compare generated scientific content, not only coarse summaries.
    replay_preview, replay_hard, replay_local, replay_edges, replay_diag = generate_preview()
    require(preview["unit_id"].equals(replay_preview["unit_id"]), "Preview deterministic keys changed")
    for col in preview.columns:
        if pd.api.types.is_numeric_dtype(preview[col]):
            require(np.allclose(preview[col].to_numpy(dtype=float), replay_preview[col].to_numpy(dtype=float), equal_nan=True), f"Preview deterministic replay mismatch: {col}")
        else:
            require(preview[col].astype(str).equals(replay_preview[col].astype(str)), f"Preview deterministic replay mismatch: {col}")
    require(np.allclose(hard_truth.select_dtypes(include=[np.number]).to_numpy(), replay_hard.select_dtypes(include=[np.number]).to_numpy(), equal_nan=True), "Hard truth deterministic replay mismatch")
    require(np.allclose(localized_truth.select_dtypes(include=[np.number]).to_numpy(), replay_local.select_dtypes(include=[np.number]).to_numpy(), equal_nan=True), "Localized truth deterministic replay mismatch")
    require(edges.equals(replay_edges), "Preview graph deterministic replay mismatch")

    # Endpoint localized interventions must exactly agree with endpoint hard-dose truth.
    for endpoint in [0.0, 1.0]:
        hard = hard_truth.loc[hard_truth.dose == endpoint, ["unit_id", "Y_true"]].sort_values("unit_id")
        localized = localized_truth.loc[localized_truth.target_dose == endpoint, ["unit_id", "localized_Y_true_mean"]].sort_values("unit_id")
        require(np.allclose(hard.Y_true.to_numpy(), localized.localized_Y_true_mean.to_numpy()), f"Endpoint localized truth mismatch at {endpoint}")

    # Interior localized means should not be mislabeled as hard-dose values.
    hard_mid = hard_truth.loc[hard_truth.dose == 0.5, "Y_true"].to_numpy()
    local_mid = localized_truth.loc[localized_truth.target_dose == 0.5, "localized_Y_true_mean"].to_numpy()
    require(not np.allclose(hard_mid, local_mid), "Localized interior target collapsed to hard-dose truth")

    manifest = json.loads((output / "stage3a_manifest.json").read_text(encoding="utf-8"))
    require(manifest["script_version"] == SCRIPT_VERSION, "Stage 3A script version mismatch")
    require(manifest["primary_estimand"] == "localized stochastic potential outcome", "Manifest target mismatch")
    for name, info in manifest["outputs"].items():
        require((output / name).exists(), f"Manifest output missing: {name}")
        require(sha256_file(output / name) == info["sha256"], f"Stage 3A output hash mismatch: {name}")

    verification = {
        "status": "verified_complete",
        "verified_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "stage": "3A",
        "frozen_input_count": len(input_manifest["files"]),
        "mine_count": 5,
        "block_count": EXPECTED_BLOCKS,
        "block_year_count": EXPECTED_BLOCK_YEARS,
        "scenario_count": 10,
        "method_count": 6,
        "factor_count": 12,
        "metric_count": 23,
        "ablation_count": 10,
        "additional_stress_test_count": 3,
        "mandatory_gate_count": 17,
        "reporting_contract_item_count": 16,
        "reduction_test_count": 4,
        "seed_count": 3200,
        "preview_unit_count": 625,
        "production_experiments_run": False,
        "checks": {
            "frozen_input_hashes": "pass",
            "accepted_stage2b_archive": "pass",
            "stage2a_stage2b_contract": "pass",
            "scenario_registry": "pass",
            "method_registry": "pass",
            "factor_registry": "pass",
            "nuisance_model_registry": "pass",
            "ablation_registry": "pass",
            "metric_registry": "pass",
            "reduction_registry": "pass",
            "additional_stress_test_registry": "pass",
            "mandatory_gate_registry": "pass",
            "reporting_contract_registry": "pass",
            "source_alignment_matrix": "pass",
            "deferred_feature_registry": "pass",
            "residual_law_registry": "pass",
            "seed_registry": "pass",
            "global_seed_uniqueness": "pass",
            "target_draw_seed_stream": "pass",
            "mine_role_separation": "pass",
            "causal_assumption_registry": "pass",
            "proof_test_registry": "pass",
            "snapshot_feature_temporal_guard": "pass",
            "post_assignment_quality_guard": "pass",
            "localized_target_draw_contract": "pass",
            "target_population_contract": "pass",
            "environment_inventory": "pass",
            "localized_target_registry": "pass",
            "mixed_treatment_atoms": "pass",
            "mixed_treatment_normalization": "pass",
            "oracle_density": "pass",
            "proper_gmrf_draw": "pass",
            "potential_outcome_truth": "pass",
            "localized_target_truth": "pass",
            "deterministic_generator_replay": "pass",
            "no_dense_inverse": "pass",
            "manifest_hashes": "pass",
        },
    }
    write_json(output / "STAGE3A_VERIFICATION.json", verification)

    archive = PACKAGE_ROOT / "GeoDose_Stage3A_OUTPUTS.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output.parent))

    print("STAGE 3A VERIFIED COMPLETE")
    print("Frozen mines: 5")
    print("Frozen blocks: 27,042")
    print("Registered scenarios: 10")
    print("Registered methods: 6")
    print("Generator preview: 625 units")
    print("Primary target: localized stochastic potential outcome")
    print("Production inference run: no")
    print(f"Output archive: {archive}")


if __name__ == "__main__":
    try:
        main()
    except Stage3AError as exc:
        print(f"STAGE 3A VERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
