from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geodose_stage3c.graph import build_graph_cache, deterministic_graph_safe_set, verify_independent_set
from geodose_stage3c.io import execution_tree_hash, load_stage3a, load_stage3b, sha256_file, write_json
from geodose_stage3c.runner import derived_seed, source_scenario_id
from geodose_stage3c.weights import weighted_conformal_quantile


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs_stage3c_reference")
    parser.add_argument("--stage3a", type=Path, default=ROOT / "inputs/stage3a/GeoDose_Stage3A_OUTPUTS.zip")
    parser.add_argument("--stage3b", type=Path, default=ROOT / "inputs/stage3b/GeoDose_Stage3B_OUTPUTS.zip")
    args = parser.parse_args()
    out = args.output
    checks: dict[str, str] = {}

    manifest = json.loads((out / "stage3c_manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for name, metadata in manifest["outputs"].items():
        path = out / name
        if not path.exists() or path.stat().st_size != metadata["size_bytes"] or sha256_file(path) != metadata["sha256"]:
            mismatches.append(name)
    checks["manifest_hashes_and_sizes"] = "pass" if not mismatches else "fail"

    intervals = read_csv(out / "stage3c_method_intervals.csv.gz")
    metrics = read_csv(out / "stage3c_reference_metrics.csv.gz")
    counts = json.loads((out / "stage3c_counts.json").read_text(encoding="utf-8"))
    method_status = read_csv(out / "stage3c_method_status.csv.gz")
    seeds_used = read_csv(out / "stage3c_seed_usage.csv.gz")
    stage3a = load_stage3a(args.stage3a)
    stage3b = load_stage3b(args.stage3b)

    checks["case_count_27"] = "pass" if intervals["case_id"].nunique() == 27 else "fail"
    checks["reference_methods_exact"] = "pass" if set(intervals["method"]) == {"M1", "M2", "H1", "H4", "M5"} else "fail"
    checks["M3_M4_M6_absent"] = "pass" if not ({"M3", "M4", "M6"} & set(intervals["method"])) else "fail"
    status_map = method_status.set_index("method_id")["implementation_status"].to_dict()
    checks["method_boundary_explicit"] = "pass" if (
        status_map.get("M3", "").startswith("deferred")
        and status_map.get("M4", "").startswith("deferred")
        and status_map.get("M6", "").startswith("deferred")
        and status_map.get("H1") == "implemented_reference"
        and status_map.get("H4") == "implemented_reference"
    ) else "fail"
    checks["production_false"] = "pass" if counts["production_experiments_run"] is False else "fail"
    checks["factorial_nuisance_variants"] = "pass" if set(intervals["nuisance_variant"]) == {"OT_TG", "ET_TG", "OT_FG", "ET_FG"} else "fail"
    checks["single_replication_only"] = "pass" if set(intervals["replication"]) == {1} else "fail"
    checks["target_design_not_used"] = "pass" if not intervals["target_design_ratio_used"].any() else "fail"

    keys = ["query_id", "predictor", "nuisance_variant", "method"]
    checks["no_duplicate_interval_keys"] = "pass" if not intervals.duplicated(keys).any() else "fail"
    task_methods = intervals.groupby(["query_id", "predictor", "nuisance_variant"])["method"].nunique()
    checks["five_reference_methods_per_task"] = "pass" if int(task_methods.min()) == 5 and int(task_methods.max()) == 5 else "fail"
    shared_outcome = intervals.groupby(["query_id", "predictor"])["outcome_fit_id"].nunique()
    checks["shared_outcome_fit_across_methods_and_variants"] = "pass" if int(shared_outcome.max()) == 1 else "fail"
    xgb_scenarios = set(intervals.loc[intervals["predictor"] == "xgb", "scenario_id"])
    checks["xgb_scope_matches_stage3a_registry"] = "pass" if xgb_scenarios == {"S1", "S4", "S5", "S8"} else "fail"

    # Frozen Stage 3A nuisance seeds and labelled substreams.
    seed_ok = True
    for row in seeds_used.itertuples(index=False):
        source = source_scenario_id(str(row.scenario_id))
        frozen = stage3a.seeds[
            (stage3a.seeds["phase"] == "pilot")
            & (stage3a.seeds["scenario_id"] == source)
            & (stage3a.seeds["replication"] == 1)
        ]
        if len(frozen) != 1 or int(frozen.iloc[0]["nuisance_seed"]) != int(row.frozen_nuisance_seed):
            seed_ok = False
            break
        expected = {
            "derived_rf_seed": derived_seed(row.frozen_nuisance_seed, "OUTCOME_RF"),
            "derived_xgb_seed": derived_seed(row.frozen_nuisance_seed, "OUTCOME_XGB"),
            "derived_propensity_seed": derived_seed(row.frozen_nuisance_seed, "MIXED_PROPENSITY_ESTIMATED"),
            "derived_propensity_misspecified_seed": derived_seed(row.frozen_nuisance_seed, "MIXED_PROPENSITY_MISSPECIFIED"),
        }
        for column, value in expected.items():
            if int(getattr(row, column)) != int(value):
                seed_ok = False
                break
    checks["frozen_stage3a_nuisance_seeds_used"] = "pass" if seed_ok else "fail"
    s4_seeds = seeds_used[seeds_used["source_scenario_id"] == "S4"]["frozen_nuisance_seed"].unique()
    checks["S4_and_stress_common_model_randomness"] = "pass" if len(s4_seeds) == 1 else "fail"

    # Method-specific invariance/factorial separation.
    m1 = intervals[intervals["method"] == "M1"]
    m1_pivot = m1.pivot_table(index=["query_id", "predictor"], columns="nuisance_variant", values="quantile", aggfunc="first")
    checks["M1_invariant_to_treatment_and_graph"] = "pass" if len(m1_pivot.dropna()) and np.allclose(
        m1_pivot.dropna().to_numpy(), m1_pivot.dropna().iloc[:, [0]].to_numpy(), equal_nan=True
    ) else "fail"

    m2 = intervals[intervals["method"] == "M2"]
    m2_pivot = m2.pivot_table(index=["query_id", "predictor", "treatment_mode"], columns="graph_mode", values="quantile", aggfunc="first").dropna()
    checks["M2_invariant_to_graph"] = "pass" if len(m2_pivot) and np.allclose(m2_pivot["true_graph"], m2_pivot["fitted_graph"], equal_nan=True) else "fail"

    h1 = intervals[intervals["method"] == "H1"]
    h1_pivot = h1.pivot_table(index=["query_id", "predictor"], columns="nuisance_variant", values="quantile", aggfunc="first").dropna()
    checks["H1_invariant_to_treatment_and_graph"] = "pass" if len(h1_pivot) and np.allclose(
        h1_pivot.to_numpy(), h1_pivot.iloc[:, [0]].to_numpy(), equal_nan=True
    ) else "fail"

    h4 = intervals[intervals["method"] == "H4"]
    h4_pivot = h4.pivot_table(index=["query_id", "predictor", "treatment_mode"], columns="graph_mode", values="quantile", aggfunc="first").dropna()
    checks["H4_invariant_to_graph"] = "pass" if len(h4_pivot) and np.allclose(
        h4_pivot["true_graph"], h4_pivot["fitted_graph"], equal_nan=True
    ) else "fail"

    method_provenance_ok = (
        intervals[intervals["method"].isin(["M1", "H1"])]["propensity_fit_id"].eq("not_used").all()
        and intervals[intervals["method"].isin(["M1", "M2"])]["graph_hash"].eq("not_used").all()
        and intervals[intervals["method"].isin(["H1", "H4"])]["graph_hash"].eq("geographic_distance_no_graph").all()
        and intervals[intervals["method"].eq("M5")]["graph_hash"].ne("not_used").all()
        and intervals[intervals["method"].eq("M5")]["graph_hash"].ne("geographic_distance_no_graph").all()
        and intervals[(intervals["method"].isin(["M2", "H4", "M5"])) & (intervals["treatment_mode"] == "oracle_treatment")]["propensity_fit_id"].eq("oracle").all()
        and intervals[(intervals["method"].isin(["M2", "H4", "M5"])) & (intervals["treatment_mode"] == "estimated_treatment")]["propensity_fit_id"].ne("oracle").all()
    )
    checks["method_specific_provenance"] = "pass" if method_provenance_ok else "fail"

    estimated_m2 = intervals[(intervals["method"] == "M2") & (intervals["treatment_mode"] == "estimated_treatment") & (intervals["interval_status"] != "refused")]
    target_variation = estimated_m2.groupby(["case_id", "target_scope", "target_dose_group", "predictor"])["target_normalized_weight"].nunique()
    checks["target_pseudoweight_is_unit_specific"] = "pass" if len(target_variation) and int(target_variation.max()) > 1 else "fail"

    observational_metrics = metrics[metrics["target_scope"] == "observational_law"]
    checks["observational_law_grouped_once"] = "pass" if set(observational_metrics["target_dose_group"]) == {"observational_law"} and int(observational_metrics["requested_n"].max()) > 1 else "fail"

    # Coverage and false-support identities.
    returned = intervals["interval_status"].ne("refused")
    recomputed_observed = returned & (intervals["lower"] <= intervals["Y_observed_at_A_star"]) & (intervals["Y_observed_at_A_star"] <= intervals["upper"])
    recomputed_latent = returned & (intervals["lower"] <= intervals["Y_true_at_A_star"]) & (intervals["Y_true_at_A_star"] <= intervals["upper"])
    checks["observed_coverage_recomputed"] = "pass" if (recomputed_observed == intervals["covered_observed"]).all() else "fail"
    checks["latent_coverage_recomputed"] = "pass" if (recomputed_latent == intervals["covered_latent"]).all() else "fail"
    recomputed_false_support = returned & ~intervals["Y_observed_at_A_star"].notna()
    # Unsupported rows have missing target truth and must be refused, so false-support must remain zero.
    checks["false_support_zero_for_reference"] = "pass" if not intervals["false_support"].any() and not recomputed_false_support.any() else "fail"

    # Eligibility rules are pre-outcome and honest about heuristic/GMRF limitations.
    checks["heuristics_never_theorem_eligible"] = "pass" if not intervals[intervals["method"].isin(["H1", "H4"])]["theorem_eligible_preoutcome"].any() else "fail"
    case_law = stage3b.cases.set_index("case_id")["residual_law"]
    m5 = intervals[intervals["method"] == "M5"].copy()
    m5["residual_law"] = m5["case_id"].map(case_law)
    checks["M5_gmrf_sampling_assumption_not_claimed"] = "pass" if not m5[m5["residual_law"] != "iid_continuous"]["theorem_eligible_preoutcome"].any() else "fail"
    estimated_weighted = intervals[(intervals["method"].isin(["M2", "M5"])) & (intervals["treatment_mode"] == "estimated_treatment")]
    checks["estimated_treatment_not_theorem_certified"] = "pass" if not estimated_weighted["theorem_eligible_preoutcome"].any() else "fail"
    spatial_covariate_weighted = intervals[(intervals["method"].isin(["M2", "M5"])) & intervals["case_id"].map(stage3b.cases.set_index("case_id")["covariate_law"]).ne("iid")]
    checks["spatial_covariates_not_treated_as_iid"] = "pass" if not spatial_covariate_weighted["theorem_eligible_preoutcome"].any() else "fail"
    checks["ST3_not_causally_eligible"] = "pass" if not intervals[intervals["scenario_id"] == "ST3"]["theorem_eligible_preoutcome"].any() else "fail"

    # Temporal and S10 paired-support audits.
    temporal = read_csv(out / "stage3c_temporal_audit.csv.gz").iloc[0]
    checks["ST2_temporal_fixture_consumed"] = "pass" if (
        int(temporal["row_count"]) == 1875
        and int(temporal["unit_count"]) == 625
        and bool(temporal["role_unique_per_unit"])
        and bool(temporal["complete_2023_2025_per_unit"])
        and bool(temporal["row_split_forbidden_all"])
        and temporal["interval_path_status"] == "cross_sectional_proxy_only_temporal_cluster_method_deferred"
    ) else "fail"
    s10 = read_csv(out / "stage3c_s10_paired_support_audit.csv.gz").iloc[0]
    checks["S10_paired_aggregation_consumed"] = "pass" if (
        int(s10["map_row_count"]) == 625
        and int(s10["paired_unit_count"]) == 169
        and float(s10["map_weight_sum_max_abs_error"]) < 1e-12
        and float(s10["max_abs_aggregate_difference"]) < 1e-12
        and s10["treatment_likelihood_status"] == "not_closed_form_diagnostic_only"
    ) else "fail"

    # Propensity normalization and no leakage.
    norm = read_csv(out / "stage3c_propensity_normalization_audit.csv.gz")
    checks["mixed_propensity_normalization"] = "pass" if (
        np.allclose(norm["interior_integral"], 1.0, atol=1e-12)
        and np.allclose(norm["total_mixed_mass"], 1.0, atol=1e-10)
        and (norm[["pi_atom_0_hat", "pi_atom_1_hat", "pi_interior_hat"]].to_numpy() > 0).all()
        and (norm[["beta_alpha_hat", "beta_beta_hat"]].to_numpy() > 0).all()
    ) else "fail"
    nuisance = read_csv(out / "stage3c_nuisance_fit_registry.csv.gz")
    prop = read_csv(out / "stage3c_propensity_diagnostics.csv.gz")
    checks["no_support_or_target_fit_leakage"] = "pass" if ((~nuisance["support_audit_used_for_fit"]) & (~nuisance["target_used_for_fit"])).all() else "fail"
    checks["propensity_no_support_or_target_fit_leakage"] = "pass" if ((~prop["support_audit_used_for_fit"]) & (~prop["target_used_for_fit"])).all() else "fail"
    checks["fit_ids_content_addressed_present"] = "pass" if nuisance["training_data_hash"].str.len().eq(64).all() and prop["training_data_hash"].str.len().eq(64).all() else "fail"
    checks["hyperparameters_machine_readable"] = "pass" if (
        nuisance["hyperparameters_json"].str.len().gt(2).all()
        and prop["category_hyperparameters_json"].str.len().gt(2).all()
        and prop["interior_hyperparameters_json"].str.len().gt(2).all()
    ) else "fail"

    ratio = read_csv(out / "stage3c_target_design_ratio_diagnostics.csv.gz")
    checks["target_ratio_coordinates_excluded"] = "pass" if ratio["raw_coordinates_excluded"].all() else "fail"
    checks["target_ratio_not_used_by_reference"] = "pass" if (~ratio["used_by_reference_methods"]).all() else "fail"

    spatial = read_csv(out / "stage3c_spatial_diagnostics.csv.gz")
    semivariogram = read_csv(out / "stage3c_semivariogram.csv.gz")
    checks["morans_I_diagnostic_exported"] = "pass" if len(spatial) > 0 and spatial["diagnostic_only"].all() and spatial["not_a_conformal_deficit"].all() else "fail"
    checks["semivariogram_diagnostic_exported"] = "pass" if len(semivariogram) > 0 and semivariogram["diagnostic_only"].all() else "fail"

    failure_log = read_csv(out / "stage3c_computational_failure_log.csv.gz")
    checks["computational_failures_separate_and_zero"] = "pass" if len(failure_log) == 0 and int(counts["computational_failure_rows"]) == 0 else "fail"
    checks["thresholds_deferred"] = "pass" if set(intervals["threshold_status"]) == {"deferred_until_20_replication_pilot"} else "fail"

    # Independent weighted-quantile reconstruction from archived calibration trace.
    audit = read_csv(out / "stage3c_calibration_weight_audit.csv.gz")
    audit_ok = True
    for _, group in audit.groupby(["query_id", "predictor", "nuisance_variant", "method"], sort=False):
        order = np.argsort(group["score"].to_numpy(float), kind="mergesort")
        result = weighted_conformal_quantile(
            group["score"].to_numpy(float)[order],
            group["log_weight"].to_numpy(float)[order],
            float(group["target_log_weight"].iloc[0]),
            float(group["alpha"].iloc[0]),
        )
        archived = float(group["archived_quantile"].iloc[0])
        if not ((np.isinf(result.q) and np.isinf(archived)) or np.isclose(result.q, archived, rtol=0, atol=1e-12, equal_nan=True)):
            audit_ok = False
            break
    checks["independent_quantile_reconstruction"] = "pass" if audit_ok else "fail"

    # H1 must use the frozen absolute-scale Euclidean Gaussian kernel; the
    # target pseudo-weight is k(0)=1, hence log-weight zero.
    h1_kernel_ok = True
    interval_lookup = intervals.set_index("query_id")
    unit_lookup = stage3b.units.set_index(["case_id", "unit_id"])
    h1_audit = audit[audit["method"] == "H1"]
    for query_id, group in h1_audit.groupby("query_id", sort=False):
        interval_row = interval_lookup.loc[query_id]
        if isinstance(interval_row, pd.DataFrame):
            interval_row = interval_row.iloc[0]
        case_id = str(interval_row["case_id"])
        target_unit_id = str(interval_row["unit_id"])
        target_xy = unit_lookup.loc[(case_id, target_unit_id), ["x_coord", "y_coord"]].to_numpy(float)
        calibration_units = stage3b.units[(stage3b.units["case_id"] == case_id) & (stage3b.units["role"] == "calibration")].set_index("unit_id")
        calibration_xy = calibration_units.loc[group["calibration_unit_id"], ["x_coord", "y_coord"]].to_numpy(float)
        expected = -np.sum((calibration_xy - target_xy.reshape(1, 2)) ** 2, axis=1) / (2.0 * 0.25 ** 2)
        if not np.allclose(group["log_weight"].to_numpy(float), expected, atol=1e-12, rtol=0):
            h1_kernel_ok = False
            break
        if not np.allclose(group["target_log_weight"].to_numpy(float), 0.0, atol=0, rtol=0):
            h1_kernel_ok = False
            break
    checks["H1_exact_absolute_scale_geographic_kernel"] = "pass" if h1_kernel_ok else "fail"

    # H4 = M2 + H1 log weights for matching graph/treatment branches.
    h4_ok = True
    audit_key = ["query_id", "predictor", "calibration_unit_id"]
    for variant, (treatment_variant, graph_variant) in {
        "OT_TG": ("OT_TG", "OT_TG"),
        "ET_TG": ("ET_TG", "ET_TG"),
        "OT_FG": ("OT_FG", "OT_FG"),
        "ET_FG": ("ET_FG", "ET_FG"),
    }.items():
        subset = audit[audit["nuisance_variant"] == variant]
        m2w = subset[subset["method"] == "M2"].set_index(audit_key)["log_weight"]
        h1w = subset[subset["method"] == "H1"].set_index(audit_key)["log_weight"]
        h4w = subset[subset["method"] == "H4"].set_index(audit_key)["log_weight"]
        common = m2w.index.intersection(h1w.index).intersection(h4w.index)
        lhs = h4w.loc[common].to_numpy(float)
        rhs = m2w.loc[common].to_numpy(float) + h1w.loc[common].to_numpy(float)
        finite = np.isfinite(lhs) & np.isfinite(rhs)
        same_neg_inf = np.isneginf(lhs) & np.isneginf(rhs)
        if not np.all(finite | same_neg_inf) or (finite.any() and not np.allclose(lhs[finite], rhs[finite], atol=1e-12, rtol=0)):
            h4_ok = False
            break
    checks["H4_exact_naive_product_identity"] = "pass" if h4_ok else "fail"

    m5_weight_ok = True
    for variant in sorted(audit["nuisance_variant"].unique()):
        subset = audit[audit["nuisance_variant"] == variant]
        m2w = subset[subset["method"] == "M2"].set_index(audit_key)["log_weight"]
        m5w = subset[subset["method"] == "M5"].set_index(audit_key)["log_weight"]
        common = m2w.index.intersection(m5w.index)
        selected = np.isfinite(m5w.loc[common].to_numpy(float))
        if selected.any() and not np.allclose(
            m5w.loc[common].to_numpy(float)[selected],
            m2w.loc[common].to_numpy(float)[selected],
            atol=1e-12,
            rtol=0,
        ):
            m5_weight_ok = False
            break
    checks["M5_uses_M2_weights_on_graph_safe_subset"] = "pass" if m5_weight_ok else "fail"

    # Graph-safe sets are recomputed from accepted Stage 3B graphs.
    graph_safe_ok = True
    reference_units = intervals[["case_id", "unit_id", "graph_mode"]].drop_duplicates()
    for (case_id, graph_mode), group in reference_units.groupby(["case_id", "graph_mode"]):
        case_units = stage3b.units[stage3b.units["case_id"] == case_id]
        edges = stage3b.true_edges if graph_mode == "true_graph" else stage3b.fitted_edges
        case_edges = edges[edges["case_id"] == case_id]
        node_map = case_units.set_index("unit_id")["node_index"]
        target_nodes = np.sort(node_map.loc[group["unit_id"]].to_numpy(int))
        cache = build_graph_cache(case_units, case_edges, target_nodes)
        calibration_nodes = case_units.loc[case_units["role"] == "calibration", "node_index"].to_numpy(int)
        for node in target_nodes:
            selected = deterministic_graph_safe_set(cache, int(node), calibration_nodes)
            if not verify_independent_set(cache, int(node), selected):
                graph_safe_ok = False
                break
    checks["graph_safe_sets_recomputed"] = "pass" if graph_safe_ok else "fail"

    environment = json.loads((out / "stage3c_environment_inventory.json").read_text(encoding="utf-8"))
    required_environment_fields = {"python", "numpy", "pandas", "scipy", "scikit_learn", "xgboost", "joblib", "PyYAML"}
    checks["complete_environment_inventory"] = "pass" if required_environment_fields.issubset(environment) and environment.get("official_runner_enforces_exact_environment") is True else "fail"

    # Source/config/environment provenance.
    source_record = json.loads((out / "stage3c_source_hashes.json").read_text(encoding="utf-8"))
    execution_files = sorted(source_record["files"])
    tree_hash, observed_hashes = execution_tree_hash(ROOT, execution_files)
    checks["execution_file_hashes_exact"] = "pass" if observed_hashes == source_record["files"] else "fail"
    checks["package_tree_hash_exact"] = "pass" if tree_hash == source_record["package_tree_sha256"] == manifest["package_tree_sha256"] else "fail"

    # Hand calculations for target pseudo-mass.
    scores = np.array([1.0, 2.0, 3.0])
    q_inf = weighted_conformal_quantile(scores, np.zeros(3), 0.0, 0.20)
    q_fin = weighted_conformal_quantile(scores, np.zeros(3), np.log(0.01), 0.25)
    checks["weighted_quantile_target_pseudomass"] = "pass" if np.isinf(q_inf.q) else "fail"
    checks["weighted_quantile_finite_manual"] = "pass" if q_fin.q == 3.0 else "fail"

    failed = {name: value for name, value in checks.items() if value != "pass"}
    verification = {
        "status": "verified_complete" if not failed else "failed",
        "stage": "3C-reference-infrastructure",
        "script_version": manifest["script_version"],
        "checks": checks,
        "failed_checks": failed,
        "validation_cases": int(intervals["case_id"].nunique()),
        "target_queries": int(intervals["query_id"].nunique()),
        "interval_rows": int(len(intervals)),
        "implemented_principal_baselines": ["M1", "M2", "M5"],
        "implemented_heuristic_ablations": ["H1", "H4"],
        "M3_implemented": False,
        "M4_implemented": False,
        "M6_implemented": False,
        "production_experiments_run": False,
        "scientific_interpretation": "Gate C1 reference infrastructure only; no production or Monte Carlo evidence",
    }
    write_json(out / "STAGE3C_REFERENCE_VERIFICATION.json", verification)
    if failed:
        raise SystemExit(f"Verification failed: {failed}")
    print("STAGE 3C REFERENCE INFRASTRUCTURE VERIFIED COMPLETE")
    print(f"Verification checks: {len(checks)}/{len(checks)}")
    print(f"Validation cases: {verification['validation_cases']}")
    print(f"Target queries: {verification['target_queries']:,}")
    print(f"Interval rows: {verification['interval_rows']:,}")
    print("Implemented principal baselines: M1, M2, M5")
    print("Implemented heuristic ablations: H1, H4")
    print("M3 implemented: no")
    print("M4 implemented: no")
    print("M6 implemented: no")
    print("Production experiments run: no")


if __name__ == "__main__":
    main()
