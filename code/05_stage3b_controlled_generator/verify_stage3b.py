from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geodose_stage3b.core import (  # noqa: E402
    EXPECTED_STAGE3A_SHA256,
    PRIMARY_DOSES,
    SCRIPT_VERSION,
    Stage3BError,
    dataframe_hash,
    derived_seed,
    frame_to_cases,
    generate_case,
    read_stage3a_archive,
    require,
    sha256_file,
    source_seed_row,
    utc_now,
    validate_stage3a_contract,
    write_json,
)


def read_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")


def verify_hashes(output_dir: Path, manifest: Dict[str, Any]) -> None:
    for name, info in manifest["outputs"].items():
        path = output_dir / name
        require(path.exists(), f"Missing manifest output: {name}")
        require(path.stat().st_size == int(info["size_bytes"]), f"Size mismatch: {name}")
        require(sha256_file(path) == info["sha256"], f"SHA-256 mismatch: {name}")


def verify_implementation_hashes(manifest: Dict[str, Any]) -> None:
    expected_paths = {
        "stage3b_generate.py": ROOT / "stage3b_generate.py",
        "src/geodose_stage3b/core.py": ROOT / "src" / "geodose_stage3b" / "core.py",
        "verify_stage3b.py": ROOT / "verify_stage3b.py",
        "tests/test_stage3b.py": ROOT / "tests" / "test_stage3b.py",
        "requirements_frozen_py310.txt": ROOT / "requirements_frozen_py310.txt",
    }
    require(set(manifest["implementation_files"]) == set(expected_paths), "Implementation-file manifest is incomplete")
    for name, path in expected_paths.items():
        require(path.exists(), f"Missing implementation file: {name}")
        require(sha256_file(path) == manifest["implementation_files"][name], f"Implementation SHA mismatch: {name}")
    payload = json.dumps(manifest["implementation_files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    require(hashlib.sha256(payload).hexdigest() == manifest["implementation_fingerprint_sha256"], "Implementation fingerprint mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GeoDose-CP Stage 3B outputs")
    parser.add_argument("--input", default="inputs/stage3a/GeoDose_Stage3A_OUTPUTS.zip")
    parser.add_argument("--output", default="outputs_stage3b")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    require(output_dir.exists(), f"Missing output directory: {output_dir}")
    stage3a = read_stage3a_archive(input_path)

    manifest = json.loads((output_dir / "stage3b_manifest.json").read_text(encoding="utf-8"))
    require(manifest["script_version"] == SCRIPT_VERSION, "Unexpected Stage 3B script version")
    require(manifest["input_stage3a_archive_sha256"] == EXPECTED_STAGE3A_SHA256, "Manifest Stage 3A SHA mismatch")
    require(manifest["production_experiments_run"] is False, "Production experiments were unexpectedly run")
    require(manifest["methods_implemented"] is False, "Methods unexpectedly marked implemented")
    require(manifest["coverage_evaluated"] is False, "Coverage unexpectedly evaluated")
    require(int(manifest["replication"]) == 1 and manifest["seed_phase"] == "pilot", "Validation replication contract failed")
    verify_implementation_hashes(manifest)
    verify_hashes(output_dir, manifest)

    cases = pd.read_csv(output_dir / "stage3b_case_registry.csv", encoding="utf-8-sig")
    alignment = json.loads((output_dir / "stage3b_stage3a_alignment.json").read_text(encoding="utf-8"))
    require(alignment["status"] == "aligned", "Stage 3A alignment report failed")
    require(validate_stage3a_contract(stage3a, cases)["status"] == "aligned", "Independent Stage 3A contract alignment failed")

    units = read_gz(output_dir / "stage3b_validation_units.csv.gz")
    hard = read_gz(output_dir / "stage3b_hard_truth.csv.gz")
    treatment_transport = read_gz(output_dir / "stage3b_treatment_transport_truth.csv.gz")
    target_design = read_gz(output_dir / "stage3b_target_design_truth.csv.gz")
    targets = read_gz(output_dir / "stage3b_realized_target_draws.csv.gz")
    obs_targets = read_gz(output_dir / "stage3b_observational_target_draws.csv.gz")
    support = pd.read_csv(output_dir / "stage3b_oracle_support_diagnostics.csv", encoding="utf-8-sig")
    temporal = read_gz(output_dir / "stage3b_temporal_stress.csv.gz")
    fitted_edges = read_gz(output_dir / "stage3b_fitted_graph_edges.csv.gz")
    true_edges = read_gz(output_dir / "stage3b_true_graph_edges_by_case.csv.gz")
    diagnostics = pd.read_csv(output_dir / "stage3b_scenario_diagnostics.csv", encoding="utf-8-sig")
    counts = json.loads((output_dir / "stage3b_counts.json").read_text(encoding="utf-8"))
    contract = json.loads((output_dir / "stage3b_generator_contract.json").read_text(encoding="utf-8"))
    feature_contract = json.loads((output_dir / "stage3b_method_feature_contract.json").read_text(encoding="utf-8"))
    paired_s10 = read_gz(output_dir / "stage3b_s10_paired_aggregation.csv.gz")

    require(len(cases) == 27 and cases.case_id.is_unique, "Case registry failed")
    require(set(f"S{i}" for i in range(1, 11)).issubset(set(cases.scenario_id)), "S1-S10 missing")
    require({"ST1", "ST2", "ST3"}.issubset(set(cases.scenario_id)), "ST1-ST3 missing")
    require(contract["production_experiments_run"] is False, "Generator contract production flag failed")
    require(contract["methods_implemented"] is False, "Generator contract method flag failed")
    require(contract["coverage_evaluated"] is False, "Generator contract coverage flag failed")
    require(contract["supported_replications"] == list(range(1, 21)), "Supported pilot replication registry failed")
    require(int(counts["replication"]) == 1 and counts["seed_phase"] == "pilot", "Output replication count contract failed")

    expected_units = int((cases.n_rows.astype(int) * cases.n_cols.astype(int)).sum())
    require(len(units) == expected_units == counts["unit_rows"], "Unit row count failed")
    require(not units.duplicated(["case_id", "unit_id"]).any(), "Duplicate case-unit rows")
    require(set(units.case_id) == set(cases.case_id), "Unit case coverage failed")
    require(units["A"].between(0, 1, inclusive="both").all(), "Treatment outside [0,1]")
    require(set(units.replication.astype(int)) == {1} and set(units.seed_phase) == {"pilot"}, "Unit replication fields failed")
    require(np.allclose(units.beta_mean, units.beta_alpha / (units.beta_alpha + units.beta_beta), atol=1e-12), "Beta mean parameterization failed")
    require(np.isfinite(units["oracle_g_mixed_at_A"]).all() and (units["oracle_g_mixed_at_A"] > 0).all(), "Oracle g failed")
    require(np.allclose(units[["pi_atom_0", "pi_atom_1", "pi_interior"]].sum(axis=1), 1.0, atol=1e-12), "Mixed category probabilities failed")
    cat_ok = (
        ((units.treatment_category == "atom_0") & (units.A == 0.0))
        | ((units.treatment_category == "atom_1") & (units.A == 1.0))
        | ((units.treatment_category == "interior") & units.A.between(0, 1, inclusive="neither"))
    )
    require(cat_ok.all(), "Treatment category/value mismatch")
    require(units["treatment_latent_interior_draw"].between(0, 1, inclusive="neither").all(), "Latent interior treatment draw outside (0,1)")
    require(units["treatment_category_uniform"].between(0, 1, inclusive="left").all(), "Treatment category uniform outside [0,1)")
    interior_rows = units.treatment_category == "interior"
    require(np.allclose(units.loc[interior_rows, "A"], units.loc[interior_rows, "treatment_latent_interior_draw"], atol=0, rtol=0), "Interior treatment does not equal latent draw")
    require((units.loc[units.case_id == "ST1_INTERIOR_ONLY", "treatment_category"] == "interior").all(), "Interior-only stress failed")
    require((units.loc[units.case_id == "S4_RHO060_NONGAUSSIAN", "residual_transform_power"] == 1.5).all(), "Non-Gaussian route failed")
    require(units.loc[units.case_id == "ST3_HIDDEN_C2", "hidden_spatial_confounder"].std() > 0.0, "Hidden C2 field missing")
    require(not units["hidden_confounder_observed_by_method"].astype(bool).any(), "Hidden confounder leaked")
    require("hidden_spatial_confounder" in feature_contract["forbidden_method_predictors"], "Hidden feature is not forbidden")
    require("oracle_g_mixed_at_A" in feature_contract["forbidden_method_predictors"], "Oracle density leakage guard missing")
    require(set(feature_contract["allowed_observed_nuisance_features"]).issubset(units.columns), "Allowed nuisance feature contract failed")
    require(feature_contract["target_design_ratio_features"] == ["X_nonlinear_1"], "Target-design feature contract failed")
    require({"x_coord", "y_coord"}.issubset(set(feature_contract["target_design_ratio_excluded_fields"])), "Coordinate exclusion from target-design ratio failed")
    require("oracle_target_design_ratio" in feature_contract["forbidden_method_predictors"], "Oracle design-ratio leakage guard missing")
    require({"X_nonlinear_1_base", "target_design_class", "ue_proxy", "measurement_error_mode"}.issubset(set(feature_contract["forbidden_method_predictors"])), "Generator-truth leakage guard incomplete")
    require(float(cases.loc[cases.case_id == "S1_BASE", "primary_bandwidth"].iloc[0]) == 0.10, "S1 primary bandwidth drifted from frozen Stage 3A")

    # Reduction cases isolate the intended factor: IID covariates in S1/S2/S3/S7,
    # IID residuals in S1/S3/S7, and spatial residual dependence only in S2.
    for case_id in ["S1_BASE", "S2_SPATIAL", "S3_SHIFT", "S7_EXCHANGEABLE"]:
        require(set(units.loc[units.case_id == case_id, "covariate_law"]) == {"iid"}, f"IID covariate reduction failed: {case_id}")
    for case_id in ["S1_BASE", "S3_SHIFT", "S7_EXCHANGEABLE"]:
        require(set(cases.loc[cases.case_id == case_id, "residual_law"]) == {"iid_continuous"}, f"IID residual reduction failed: {case_id}")
    require(np.isclose(float(cases.loc[cases.case_id == "S2_SPATIAL", "spatial_rho"].iloc[0]), 0.6, atol=1e-12), "S2 spatial dependence mapping failed")

    # Common-random factor variants isolate only the registered factor.
    common_groups = {
        "S4": ["S4_RHO000", "S4_RHO020", "S4_RHO040", "S4_RHO060", "S4_RHO080", "S4_RHO060_NONGAUSSIAN"],
        "S6": ["S6_ROOK", "S6_OMIT50"],
        "S9": ["S9_RANDOM", "S9_TREATMENT05", "S9_TREATMENT10", "S9_SUBSTRATE", "S9_COMBINED05"],
        "ST1": ["ST1_MIXED_ATOMS", "ST1_INTERIOR_ONLY"],
    }
    base_shared_cols = [
        "X_spatial_1", "X_spatial_2", "X_nonlinear_1_base", "X_nonlinear_2",
        "outcome_baseline", "substrate_proxy_hidden", "ue_proxy",
        "residual_base_innovation", "measurement_base_innovation",
        "treatment_latent_interior_draw", "treatment_category_uniform",
    ]
    for label, group_cases in common_groups.items():
        reference = units[units.case_id == group_cases[0]].sort_values("node_index")
        for other_id in group_cases[1:]:
            other = units[units.case_id == other_id].sort_values("node_index")
            for col in base_shared_cols:
                require(np.allclose(reference[col], other[col], atol=0, rtol=0), f"Common-random isolation failed: {label}/{col}")
    # When the registered residual law and rho are unchanged, the realized residual
    # must also be identical. S4 deliberately varies rho/residual transform and is
    # compared through the common base innovation instead.
    for label in ["S6", "S9", "ST1"]:
        group_cases = common_groups[label]
        reference = units[units.case_id == group_cases[0]].sort_values("node_index")
        for other_id in group_cases[1:]:
            other = units[units.case_id == other_id].sort_values("node_index")
            require(np.allclose(reference["shared_spatial_residual"], other["shared_spatial_residual"], atol=0, rtol=0), f"Residual-factor isolation failed: {label}")
    s4_gauss = units[units.case_id == "S4_RHO060"].sort_values("node_index")
    s4_nongauss = units[units.case_id == "S4_RHO060_NONGAUSSIAN"].sort_values("node_index")
    require(np.allclose(s4_gauss["residual_latent_gaussian"], s4_nongauss["residual_latent_gaussian"], atol=0, rtol=0), "S4 transformed-GMRF latent draw mismatch")
    # Treatment law is shared for S4/S6/S9; ST1 deliberately changes endpoint structure.
    for group_cases in [common_groups["S4"], common_groups["S6"], common_groups["S9"]]:
        reference = units[units.case_id == group_cases[0]].sort_values("node_index")
        for other_id in group_cases[1:]:
            other = units[units.case_id == other_id].sort_values("node_index")
            for col in ["A", "pi_atom_0", "pi_atom_1", "pi_interior", "beta_alpha", "beta_beta"]:
                require(np.allclose(reference[col], other[col], atol=0, rtol=0), f"Treatment-factor isolation failed: {group_cases[0]} vs {other_id}/{col}")

    # Dedicated N2/N3 target-design transport fixture: base draws and all
    # observational-design rows match S4_RHO060, while target-design rows have
    # the exact registered N(delta,1) location shift in X_nonlinear_1.
    design_ref = units[units.case_id == "S4_RHO060"].sort_values("node_index").reset_index(drop=True)
    design_shift = units[units.case_id == "S4_RHO060_DESIGN_SHIFT"].sort_values("node_index").reset_index(drop=True)
    require(np.allclose(design_ref.X_nonlinear_1_base, design_shift.X_nonlinear_1_base, atol=0, rtol=0), "Design-shift base covariate pairing failed")
    target_mask_design = design_shift.target_design_class == "target_design"
    obs_mask_design = ~target_mask_design
    require(np.allclose(design_shift.loc[obs_mask_design, "X_nonlinear_1"], design_ref.loc[obs_mask_design, "X_nonlinear_1"], atol=0, rtol=0), "Observational design changed in target-design factor case")
    delta = float(cases.loc[cases.case_id == "S4_RHO060_DESIGN_SHIFT", "target_design_delta"].iloc[0])
    require(np.allclose(design_shift.loc[target_mask_design, "X_nonlinear_1"] - design_ref.loc[target_mask_design, "X_nonlinear_1"], delta, atol=1e-12), "Target design shift magnitude failed")

    role_counts = units.groupby(["case_id", "role"]).size().unstack(fill_value=0)
    for role in ["nuisance_training", "support_audit", "calibration", "test_target"]:
        require(role in role_counts.columns and (role_counts[role] > 0).all(), f"Missing role: {role}")
    require(int(role_counts.loc["S8_SMALL_CAL", "calibration"]) == 25, "Small-calibration scenario failed")

    # Nuisance training and support audit are separate independent graph components;
    # calibration and test remain connected in the inference graph.
    unit_lookup = units.set_index(["case_id", "node_index"])[["information_component", "role"]]
    for case_id, edge_group in true_edges.groupby("case_id"):
        left = unit_lookup.loc[[(case_id, int(i)) for i in edge_group.source_node]]
        right = unit_lookup.loc[[(case_id, int(i)) for i in edge_group.target_node]]
        require(np.array_equal(left.information_component.to_numpy(), right.information_component.to_numpy()), f"Independent information components are graph-linked: {case_id}")
        cal_test = ((left.role.to_numpy() == "calibration") & (right.role.to_numpy() == "test_target")) | ((left.role.to_numpy() == "test_target") & (right.role.to_numpy() == "calibration"))
        if case_id not in {"S7_EXCHANGEABLE"}:
            require(cal_test.any(), f"Calibration-target spatial link missing: {case_id}")

    target_count_by_case = cases.set_index("case_id").apply(
        lambda r: len(PRIMARY_DOSES) + (1 if r.primary_target_mode == "localized" and pd.notna(r.primary_target_dose) and all(abs(float(r.primary_target_dose) - d) > 1e-12 for d in PRIMARY_DOSES) else 0),
        axis=1,
    )
    unit_count_by_case = units.groupby("case_id").size()
    expected_hard_rows = int((unit_count_by_case * target_count_by_case).sum())
    require(len(hard) == expected_hard_rows == counts["hard_truth_rows"], "Hard truth row count failed")
    require(not hard.duplicated(["case_id", "unit_id", "target_dose"]).any(), "Duplicate hard truth")
    merged_hard = hard.merge(units[["case_id", "unit_id", "shared_spatial_residual"]], on=["case_id", "unit_id"], suffixes=("", "_unit"), validate="many_to_one")
    require(np.allclose(merged_hard.Y_true, merged_hard.conditional_mean + merged_hard.shared_spatial_residual_unit, atol=1e-12), "Hard truth identity failed")

    require(len(treatment_transport) == expected_hard_rows == counts["treatment_transport_rows"], "Treatment-transport row count failed")
    require(not treatment_transport.duplicated(["case_id", "unit_id", "target_dose"]).any(), "Duplicate treatment-transport truth")
    require((treatment_transport.loc[treatment_transport.positive_target_weight, "q_at_observed_A"] > 0).all(), "Positive treatment-transport flag failed")
    require((treatment_transport.loc[~treatment_transport.positive_target_weight, "q_at_observed_A"] == 0).all(), "Zero treatment-transport flag failed")
    require("log_oracle_treatment_ratio" in treatment_transport.columns and "log_oracle_target_ratio" not in treatment_transport.columns, "Treatment/design transport naming separation failed")

    require(len(target_design) == len(units) == counts["target_design_rows"], "Target-design truth row count failed")
    require(not target_design.duplicated(["case_id", "unit_id"]).any(), "Duplicate target-design truth")
    require(np.isfinite(target_design.oracle_target_design_ratio).all() and (target_design.oracle_target_design_ratio > 0).all(), "Oracle target-design ratio failed")
    identity_design = target_design.target_design_mode == "identity"
    require(np.allclose(target_design.loc[identity_design, "oracle_target_design_ratio"], 1.0, atol=0, rtol=0), "Identity target-design ratio failed")
    shifted = target_design[target_design.case_id == "S4_RHO060_DESIGN_SHIFT"]
    require(len(shifted) == 625 and set(shifted.target_design_ratio_feature) == {"X_nonlinear_1"}, "Shifted target-design fixture failed")
    shifted_delta = shifted.target_design_delta.astype(float).to_numpy()
    expected_ratio = np.exp(shifted_delta * shifted.X_nonlinear_1.to_numpy() - 0.5 * shifted_delta * shifted_delta)
    require(np.allclose(shifted.oracle_target_design_ratio, expected_ratio, atol=1e-12, rtol=1e-12), "Analytic target-design ratio identity failed")
    require(set(shifted.loc[shifted.role == "nuisance_training", "ratio_training_role"]) == {"observational_training"}, "Design-ratio observational training role failed")
    require(set(shifted.loc[shifted.role == "support_audit", "ratio_training_role"]) == {"target_training"}, "Design-ratio target training role failed")
    for case_id, group in target_design.groupby("case_id"):
        n0 = int((group.role == "nuisance_training").sum())
        n1 = int((group.role == "support_audit").sum())
        require((group.ratio_training_n_observational.astype(int) == n0).all(), f"Target-design n0 count failed: {case_id}")
        require((group.ratio_training_n_target.astype(int) == n1).all(), f"Target-design n1 count failed: {case_id}")
        require(np.allclose(group.class_prior_odds_n0_over_n1, n0 / n1, atol=1e-12), f"Target-design prior-odds factor failed: {case_id}")
        require(set(group.ratio_training_dependence_regime) == {"iid_design_feature_X_nonlinear_1"}, f"Target-design dependence regime failed: {case_id}")

    test_counts = units[units.role == "test_target"].groupby("case_id").size()
    expected_target_rows = int((test_counts * target_count_by_case).sum())
    require(len(targets) == expected_target_rows == counts["realized_target_draw_rows"], "Target-draw row count failed")
    require(not targets.duplicated(["case_id", "unit_id", "target_dose"]).any(), "Duplicate target draws")
    drawn = targets.draw_status == "drawn"
    endpoint = drawn & targets.target_dose.isin([0.0, 1.0])
    interior = drawn & ~targets.target_dose.isin([0.0, 1.0])
    require(np.array_equal(targets.loc[endpoint, "A_star"].to_numpy(), targets.loc[endpoint, "target_dose"].to_numpy()), "Endpoint target draw failed")
    require(targets.loc[interior, "A_star"].between(0, 1, inclusive="neither").all(), "Interior target draw failed")
    require(np.isfinite(targets.loc[drawn, "q_at_A_star"]).all() and (targets.loc[drawn, "q_at_A_star"] > 0).all(), "Target q failed")
    require(np.isfinite(targets.loc[drawn, "g_at_A_star"]).all() and (targets.loc[drawn, "g_at_A_star"] > 0).all(), "Target g failed")
    require(np.allclose(targets.loc[drawn, "Y_true_at_A_star"], targets.loc[drawn, "conditional_mean_at_A_star"] + targets.loc[drawn, "shared_spatial_residual"], atol=1e-12), "Realized latent target identity failed")
    require(targets.loc[drawn, "target_draw_subseed"].notna().all(), "Target substream seed missing")
    # Every case/target uses an order-independent deterministic target substream.
    require(not targets.loc[drawn, ["case_id", "target_dose", "target_scope", "target_draw_subseed"]].drop_duplicates().duplicated(["case_id", "target_dose", "target_scope"]).any(), "Target substream mapping failed")
    target_seed_semantics = targets.loc[drawn, ["case_id", "target_dose", "target_scope", "target_draw_subseed"]].drop_duplicates().merge(
        cases[["case_id", "common_random_group"]], on="case_id", validate="many_to_one"
    )
    target_seed_semantics["semantic"] = (
        target_seed_semantics.common_random_group.astype(str) + "|"
        + target_seed_semantics.target_dose.astype(str) + "|"
        + target_seed_semantics.target_scope.astype(str)
    )
    require((target_seed_semantics.groupby("target_draw_subseed").semantic.nunique() == 1).all(), "Accidental target-substream collision across semantic targets")
    unaudited = targets.draw_status == "R02_ENDPOINT_NOT_AUDITED"
    require(set(targets.loc[unaudited, "case_id"]) == {"ST1_INTERIOR_ONLY"}, "Endpoint refusal case failed")
    require(targets.loc[unaudited, "A_star"].isna().all(), "Unaudited endpoint was drawn")

    no_me_cases = set(cases.loc[cases.measurement_error == "none", "case_id"])
    target_none = targets[drawn & targets.case_id.isin(no_me_cases)]
    require(np.allclose(target_none.Y_observed_at_A_star, target_none.Y_true_at_A_star, atol=1e-12), "No-error target identity failed")
    s9_positive = targets[drawn & targets.case_id.str.startswith("S9_")]
    require(np.mean(np.abs(s9_positive.Y_observed_at_A_star - s9_positive.Y_true_at_A_star)) > 0.0, "S9 observed/latent separation failed")
    s9_modes = set(cases.loc[cases.scenario_id == "S9", "measurement_error"])
    require(s9_modes == {"random", "treatment_correlated", "substrate_bias", "combined"}, "S9 isolated measurement mechanisms failed")
    s9_units = units[units.scenario_id == "S9"].set_index(["case_id", "node_index"])
    random_error = s9_units.loc["S9_RANDOM", "measurement_error_at_A"].to_numpy()
    treatment05_error = s9_units.loc["S9_TREATMENT05", "measurement_error_at_A"].to_numpy()
    treatment10_error = s9_units.loc["S9_TREATMENT10", "measurement_error_at_A"].to_numpy()
    substrate_error = s9_units.loc["S9_SUBSTRATE", "measurement_error_at_A"].to_numpy()
    combined_error = s9_units.loc["S9_COMBINED05", "measurement_error_at_A"].to_numpy()
    A_s9 = s9_units.loc["S9_RANDOM", "A"].to_numpy()
    substrate_s9 = s9_units.loc["S9_RANDOM", "substrate_proxy_hidden"].to_numpy()
    require(np.allclose(treatment05_error - random_error, 0.5 * (A_s9 - 0.5), atol=1e-12), "S9 treatment component isolation failed")
    require(np.allclose(treatment10_error - random_error, 1.0 * (A_s9 - 0.5), atol=1e-12), "S9 strong treatment component isolation failed")
    require(np.allclose(substrate_error - random_error, 0.10 * substrate_s9, atol=1e-12), "S9 substrate component isolation failed")
    require(np.allclose(combined_error - random_error, 0.5 * (A_s9 - 0.5) + 0.10 * substrate_s9, atol=1e-12), "S9 combined component failed")

    require(set(obs_targets.case_id) == {"S2_SPATIAL", "S7_EXCHANGEABLE"}, "Observational target cases failed")
    require(obs_targets.q_equals_g.astype(bool).all(), "Observational q=g flag failed")
    require(np.allclose(obs_targets.oracle_treatment_ratio, 1.0, atol=0), "Observational treatment ratio failed")

    require(len(support) == int(target_count_by_case.sum()), "Support diagnostic count failed")
    require((support.oracle_ess >= 0).all() and support.max_normalized_weight.between(0, 1, inclusive="both").all(), "Support metric range failed")
    require((support.operational_eligibility_status == "deferred_until_pilot_threshold_freeze").all(), "Premature eligibility decision detected")
    only_false = support.loc[~support.mathematical_positivity_holds.astype(bool)]
    require(set(only_false.case_id) == {"ST1_INTERIOR_ONLY"} and set(only_false.target_dose) == {0.0, 1.0}, "Mathematical positivity classification failed")
    require((support.loc[(support.case_id == "S8_SEVERE_TAIL") & (support.target_dose >= 0.75), "scenario_stress_label"] == "severe_weight_concentration_expected").all(), "S8 stress label failed")

    require(len(temporal) == 625 * 3 == counts["temporal_rows"], "Temporal row count failed")
    require(set(temporal.year.astype(int)) == {2023, 2024, 2025}, "Temporal years failed")
    require(not temporal.duplicated(["unit_id", "year"]).any(), "Duplicate temporal rows")
    require(temporal.row_split_forbidden.astype(bool).all(), "Temporal leakage guard failed")
    wide = temporal.pivot(index="unit_id", columns="year", values="spatiotemporal_residual")
    require(wide.corr().loc[2023, 2024] > 0.3 and wide.corr().loc[2024, 2025] > 0.3, "Temporal dependence too weak")

    require(set(fitted_edges.case_id) == set(cases.case_id), "Fitted graph case coverage failed")
    require(set(true_edges.case_id) == set(cases.case_id), "True graph case coverage failed")
    for frame, label in [(fitted_edges, "fitted"), (true_edges, "true")]:
        require(not frame.duplicated(["case_id", "source_node", "target_node"]).any(), f"Duplicate {label} edges")
        require((frame.source_node < frame.target_node).all(), f"Noncanonical {label} edge")
    diag = diagnostics.set_index("case_id")
    require(int(diag.loc["S6_ROOK", "fitted_edge_count"]) < int(diag.loc["S4_RHO060", "fitted_edge_count"]), "Rook graph not reduced")
    require(int(diag.loc["S6_OMIT50", "fitted_edge_count"]) < int(diag.loc["S4_RHO060", "fitted_edge_count"]), "Omitted graph not reduced")
    require((diagnostics.residual_min_precision_eigenvalue > 0).all(), "Residual precision failed")
    require((diagnostics.mixed_treatment_normalization_max_abs_error < 1e-5).all(), "Mixed treatment normalization failed")
    require((diagnostics.covariate_min_precision_eigenvalue > 0).all(), "Covariate precision failed")
    require(~diagnostics.dense_precision_inverse_formed.astype(bool).any(), "Dense inverse flag failed")

    # Derived substreams may be shared within a registered common-random group,
    # but not accidentally across different groups.
    seed_cols = ["data_seed", "covariate_seed", "treatment_seed", "residual_seed", "temporal_seed", "measurement_seed", "target_draw_seed"]
    for col in seed_cols:
        mapping = diagnostics.groupby(col).common_random_group.nunique()
        require((mapping == 1).all(), f"Accidental cross-group derived-seed collision: {col}")
    seed_records = []
    for col in seed_cols + ["orbit_seed"]:
        for row in diagnostics[[col, "common_random_group"]].drop_duplicates().itertuples(index=False):
            seed_records.append({"seed": int(row[0]), "stream": col, "group": str(row[1])})
    seed_frame = pd.DataFrame(seed_records)
    seed_collision = seed_frame.groupby("seed").apply(
        lambda g: g[["stream", "group"]].drop_duplicates().shape[0], include_groups=False
    )
    require((seed_collision == 1).all(), "Cross-stream derived-seed collision")
    require(diagnostics.orbit_seed.is_unique, "Orbit seed collision")

    # Audit the complete frozen 20-replication pilot seed space without running
    # methods or production experiments. Shared seeds are allowed only when the
    # semantic common-random key is identical.
    all_seed_records = []
    case_objects = frame_to_cases(cases)
    for rep in range(1, 21):
        for case in case_objects:
            source = source_seed_row(stage3a, case.scenario_id, replication=rep)
            data_seed = derived_seed(source["data_seed"], case.common_random_group)
            base = {
                "data": data_seed,
                "covariate": derived_seed(data_seed, "COVARIATES"),
                "treatment": derived_seed(data_seed, "TREATMENT"),
                "residual": derived_seed(data_seed, "RESIDUAL"),
                "temporal": derived_seed(data_seed, "TEMPORAL"),
                "measurement": derived_seed(source["measurement_seed"], case.common_random_group),
                "target": derived_seed(source["target_draw_seed"], case.common_random_group),
                "orbit": derived_seed(source["orbit_seed"], case.case_id),
            }
            for stream, value in base.items():
                semantic_group = case.case_id if stream == "orbit" else case.common_random_group
                all_seed_records.append((int(value), f"{rep}|{stream}|{semantic_group}"))
            target_specs_seed = [(float(d), "registered_grid") for d in PRIMARY_DOSES]
            if case.primary_target_mode == "localized" and case.primary_target_dose is not None and all(abs(float(case.primary_target_dose) - d) > 1e-12 for d in PRIMARY_DOSES):
                target_specs_seed.append((float(case.primary_target_dose), "scenario_primary_extra"))
            for dose, scope in target_specs_seed:
                sub = derived_seed(base["target"], f"TARGET_{dose:.12g}_{scope}")
                all_seed_records.append((int(sub), f"{rep}|target_substream|{case.common_random_group}|{dose:.12g}|{scope}"))
            if case.primary_target_mode == "observational":
                sub = derived_seed(base["target"], "OBSERVATIONAL_TARGET")
                all_seed_records.append((int(sub), f"{rep}|observational_target_substream|{case.common_random_group}"))
    all_seed_frame = pd.DataFrame(all_seed_records, columns=["seed", "semantic"]).drop_duplicates()
    require((all_seed_frame.groupby("seed").semantic.nunique() == 1).all(), "Collision in complete 20-replication derived-seed space")

    map180 = pd.read_csv(output_dir / "stage3b_support_180m_map.csv", encoding="utf-8-sig")
    require(len(map180) == 625 and map180.unit_180m.nunique() == 169, "180m support map failed")
    require(np.allclose(map180.groupby("unit_180m").aggregation_weight.sum(), 1.0, atol=1e-12), "180m aggregation weights failed")
    require(len(paired_s10) == 169 == counts["s10_paired_aggregation_rows"], "Paired S10 aggregation count failed")
    require(paired_s10.unit_180m.is_unique, "Paired S10 aggregation IDs failed")
    require((paired_s10.treatment_likelihood_status.str.contains("not_closed_form")).all(), "Paired S10 theorem boundary failed")

    expected_graph_counts = {
        "stage3b_true_queen_edges_25x25_primary.csv": 2206,
        "stage3b_rook_edges_25x25_primary.csv": 1150,
        "stage3b_true_queen_edges_25x25_small_calibration.csv": 2206,
        "stage3b_true_queen_edges_25x25_maup_aligned.csv": 2206,
        "stage3b_true_queen_edges_13x13_primary.csv": 526,
    }
    for name, expected in expected_graph_counts.items():
        require(len(pd.read_csv(output_dir / name, encoding="utf-8-sig")) == expected, f"Canonical graph count failed: {name}")

    fixtures = json.loads((output_dir / "stage3b_exact_orbit_fixtures.json").read_text(encoding="utf-8"))
    require(fixtures["fixture_version"] == "1.3", "Orbit fixture version failed")
    require(fixtures["size6_unique"]["distinct_permutations"] == math.factorial(6), "Size-6 fixture failed")
    require(fixtures["size8_unique"]["distinct_permutations"] == math.factorial(8), "Size-8 fixture failed")
    require(fixtures["size6_one_duplicate_pair"]["distinct_permutations"] == math.factorial(6) // 2, "Duplicate fixture failed")
    dup_payloads = fixtures["size6_one_duplicate_pair"]["movable_payloads"]
    require(dup_payloads[0] == dup_payloads[1], "Duplicate fixture does not contain duplicate movable payloads")
    require(fixtures["size6_one_duplicate_pair"]["fixed_slots"][0] != fixtures["size6_one_duplicate_pair"]["fixed_slots"][1], "Fixed slots were incorrectly duplicated")
    require(len(fixtures["size6_unique"]["boundary_edges"]) > 0, "Orbit boundary fixture missing")
    for fixture_name in ["size6_unique", "size8_unique", "size6_one_duplicate_pair"]:
        fixture = fixtures[fixture_name]
        roles = [slot["role"] for slot in fixture["fixed_slots"]]
        require(roles.count("test_target") == 1, f"Orbit target-slot inclusion failed: {fixture_name}")
        require(roles.count("calibration") == len(roles) - 1, f"Orbit calibration-slot contract failed: {fixture_name}")
        require(fixture["block_connected"] is True and len(fixture["internal_edges"]) >= len(roles) - 1, f"Orbit block connectedness failed: {fixture_name}")
        target_pos = int(fixture["target_slot_position"])
        require(fixture["candidate_response_replacement_required"] is True, f"Orbit candidate-replacement contract failed: {fixture_name}")
        require(fixture["movable_payloads"][target_pos]["payload_origin"] == "realized_localized_target_truth_reference", f"Orbit target payload origin failed: {fixture_name}")
        require(abs(float(fixture["target_truth_reference"]["target_dose"]) - 0.90) < 1e-12, f"Orbit target dose failed: {fixture_name}")

    replay = json.loads((output_dir / "stage3b_replay_diagnostics.json").read_text(encoding="utf-8"))
    require(replay["identical"] is True and replay["hash_first"] == replay["hash_second"], "Replay artifact failed")
    case_frame = pd.read_csv(output_dir / "stage3b_case_registry.csv", encoding="utf-8-sig")
    case = [c for c in frame_to_cases(case_frame) if c.case_id == "S1_BASE"][0]
    replay_units = generate_case(case, stage3a, {}, replication=1)["units"].sort_values("node_index").reset_index(drop=True)
    require(dataframe_hash(replay_units) == replay["hash_first"], "Independent replay failed")
    rep2_units = generate_case(case, stage3a, {}, replication=2)["units"].sort_values("node_index").reset_index(drop=True)
    require(dataframe_hash(rep2_units) != replay["hash_first"], "Distinct pilot replication stream failed")

    verification = {
        "status": "verified_complete",
        "verified_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "stage": "3B",
        "stage3a_archive_sha256": EXPECTED_STAGE3A_SHA256,
        "case_count": len(cases),
        "unit_rows": len(units),
        "hard_truth_rows": len(hard),
        "treatment_transport_rows": len(treatment_transport),
        "target_design_rows": len(target_design),
        "realized_target_draw_rows": len(targets),
        "temporal_rows": len(temporal),
        "production_experiments_run": False,
        "methods_implemented": False,
        "coverage_evaluated": False,
        "checks": {
            "frozen_stage3a_archive": "pass",
            "stage3a_contract_alignment": "pass",
            "implementation_file_hashes": "pass",
            "manifest_hashes": "pass",
            "case_registry": "pass",
            "scenario_and_stress_coverage": "pass",
            "information_component_independence": "pass",
            "calibration_target_spatial_graph": "pass",
            "iid_reduction_cases": "pass",
            "common_random_factor_isolation": "pass",
            "unit_cartesian_contract": "pass",
            "mixed_treatment_atoms": "pass",
            "mixed_treatment_normalization": "pass",
            "beta_parameterization": "pass",
            "interior_only_reference": "pass",
            "oracle_treatment_likelihood": "pass",
            "proper_gmrf": "pass",
            "non_gaussian_transformed_gmrf": "pass",
            "hidden_C2_negative_control": "pass",
            "method_feature_leakage_contract": "pass",
            "hard_dose_truth": "pass",
            "treatment_transport_truth": "pass",
            "target_design_transport_truth": "pass",
            "target_design_common_support": "pass",
            "target_design_class_prior_and_iid_training": "pass",
            "realized_localized_target_draws": "pass",
            "order_independent_target_substreams": "pass",
            "endpoint_draws_exact": "pass",
            "unaudited_endpoint_refusal": "pass",
            "observational_q_equals_g_reduction": "pass",
            "S9_latent_observed_separation": "pass",
            "S9_measurement_component_isolation": "pass",
            "positivity_vs_weight_stress_separation": "pass",
            "oracle_support_diagnostics": "pass",
            "ST2_temporal_dependence": "pass",
            "graph_misspecification_fixtures": "pass",
            "derived_seed_collision_audit": "pass",
            "complete_20_replication_seed_space": "pass",
            "S10_support_map": "pass",
            "S10_paired_aggregation_fixture": "pass",
            "exact_orbit_realized_target_candidate_and_duplicate_fixture": "pass",
            "deterministic_replay": "pass",
            "pilot_replication_engine": "pass",
            "no_dense_precision_inverse": "pass",
            "production_not_run": "pass",
            "methods_not_implemented": "pass",
            "coverage_not_evaluated": "pass",
        },
    }
    write_json(output_dir / "STAGE3B_VERIFICATION.json", verification)

    print("STAGE 3B VERIFIED COMPLETE")
    print(f"Validation cases: {len(cases)}")
    print(f"Unit rows: {len(units):,}")
    print(f"Realized target-draw rows: {len(targets):,}")
    print("Methods implemented: no")
    print("Coverage evaluated: no")
    print("Production experiments run: no")


if __name__ == "__main__":
    try:
        main()
    except Stage3BError as exc:
        print(f"STAGE 3B VERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
