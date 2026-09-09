from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib
import yaml
import xgboost

from . import __version__
from .graph import (
    build_graph_cache,
    deterministic_graph_safe_set,
    geographic_distance_logweights,
    graph_semivariogram,
    morans_i,
    verify_independent_set,
)
from .io import (
    Stage3CError,
    execution_tree_hash,
    load_stage3a,
    load_stage3b,
    output_manifest,
    safe_prepare_output_dir,
    sha256_file,
    verify_frozen_inputs,
    write_csv_gz_deterministic,
    write_json,
)
from .methods import METHODS, build_interval
from .models import (
    FEATURES,
    MixedPropensityModel,
    OutcomeRegressor,
    TargetDesignRatioModel,
    outcome_diagnostics,
)
from .weights import intervention_density, interval_score

ALPHA = 0.10
GEOGRAPHIC_BANDWIDTH = 0.25
REFERENCE_PREDICTORS = {"rf": "all", "xgb": "prespecified_subset"}
XGB_CASES = {
    # Frozen Stage 3A NUI2 scope: S1, S4, S5, and S8.  Gate C1 uses
    # every controlled case belonging to those scenario families.
    "S1_BASE",
    "S4_RHO000",
    "S4_RHO020",
    "S4_RHO040",
    "S4_RHO060",
    "S4_RHO080",
    "S4_RHO060_NONGAUSSIAN",
    "S4_RHO060_DESIGN_SHIFT",
    "S5_POOR_OVERLAP",
    "S8_SEVERE_TAIL",
    "S8_SMALL_CAL",
}
NUISANCE_VARIANTS = {
    "OT_TG": ("oracle_treatment", "true_graph"),
    "ET_TG": ("estimated_treatment", "true_graph"),
    "OT_FG": ("oracle_treatment", "fitted_graph"),
    "ET_FG": ("estimated_treatment", "fitted_graph"),
}


def derived_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def source_scenario_id(scenario_id: str) -> str:
    return scenario_id if scenario_id.startswith("S") and scenario_id[1:].isdigit() else "S4"


def frozen_nuisance_seed(stage3a_seeds: pd.DataFrame, scenario_id: str, replication: int) -> tuple[str, int]:
    source = source_scenario_id(str(scenario_id))
    rows = stage3a_seeds[
        (stage3a_seeds["phase"] == "pilot")
        & (stage3a_seeds["scenario_id"] == source)
        & (stage3a_seeds["replication"] == int(replication))
    ]
    if len(rows) != 1:
        raise Stage3CError(f"Missing frozen nuisance seed for {source}/replication {replication}")
    return source, int(rows.iloc[0]["nuisance_seed"])


def _unit_lookup(units: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    lookup = units.set_index("unit_id", drop=False)
    missing = set(targets["unit_id"]) - set(lookup.index)
    if missing:
        raise Stage3CError(f"Missing target unit IDs: {list(missing)[:3]}")
    return lookup.loc[targets["unit_id"]].reset_index(drop=True)


def _target_query_id(df: pd.DataFrame) -> pd.Series:
    def one(row: pd.Series) -> str:
        dose = "NA" if pd.isna(row["target_dose"]) else f"{float(row['target_dose']):.12g}"
        subseed = row.get("target_draw_subseed", "NA")
        text = f"{row['case_id']}|{row['unit_id']}|{row['target_scope']}|{dose}|{subseed}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return df.apply(one, axis=1)


def _reference_target_subset(case_targets: pd.DataFrame, case_units: pd.DataFrame) -> pd.DataFrame:
    test = case_units[case_units["role"] == "test_target"].sort_values("node_index")
    ids = test["unit_id"].tolist()
    if len(ids) <= 5:
        chosen = ids
    else:
        positions = np.linspace(0, len(ids) - 1, 5).round().astype(int)
        chosen = [ids[index] for index in sorted(set(positions.tolist()))]
    out = case_targets[case_targets["unit_id"].isin(chosen)].copy()
    if out.empty:
        raise Stage3CError("Reference target subset is empty")
    return out.sort_values(["unit_id", "target_scope", "target_dose"], kind="mergesort").reset_index(drop=True)


def _oracle_calibration_logweights(
    treatment_truth: pd.DataFrame,
    target_row: pd.Series,
    calibration_units: pd.DataFrame,
) -> np.ndarray:
    if target_row["target_scope"] == "observational_law":
        return np.zeros(len(calibration_units))
    subset = treatment_truth[
        (treatment_truth["case_id"] == target_row["case_id"])
        & (treatment_truth["target_scope"] == target_row["target_scope"])
        & np.isclose(treatment_truth["target_dose"].astype(float), float(target_row["target_dose"]))
        & (treatment_truth["role"] == "calibration")
    ].set_index("unit_id")
    return subset.loc[calibration_units["unit_id"], "log_oracle_treatment_ratio"].to_numpy(float)


def _estimated_logweights(
    propensity: MixedPropensityModel,
    calibration_units: pd.DataFrame,
    target_row: pd.Series,
    target_unit: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    if target_row["target_scope"] == "observational_law":
        return np.zeros(len(calibration_units)), 0.0
    dose = float(target_row["target_dose"])
    bandwidth = float(target_row["bandwidth"]) if pd.notna(target_row["bandwidth"]) else None
    endpoint_audited = bool(target_unit["endpoint_audited"].iloc[0])
    q_calibration = intervention_density(calibration_units["A"].to_numpy(float), dose, bandwidth, endpoint_audited)
    g_calibration = propensity.density(calibration_units)
    log_calibration = np.full(len(calibration_units), -np.inf)
    positive = (q_calibration > 0) & (g_calibration > 0) & np.isfinite(g_calibration)
    log_calibration[positive] = np.log(q_calibration[positive]) - np.log(g_calibration[positive])
    a_star = np.array([float(target_row["A_star"])])
    q_target = intervention_density(a_star, dose, bandwidth, endpoint_audited)[0]
    g_target = propensity.density(target_unit, a=a_star)[0]
    log_target = float(np.log(q_target) - np.log(g_target)) if q_target > 0 and g_target > 0 and np.isfinite(g_target) else -np.inf
    return log_calibration, log_target


def _oracle_target_logweight(target_row: pd.Series) -> float:
    return 0.0 if target_row["target_scope"] == "observational_law" else float(target_row["log_oracle_treatment_ratio_at_A_star"])


def _estimated_target_logweight(
    propensity: MixedPropensityModel,
    target_row: pd.Series,
    target_unit: pd.DataFrame,
) -> float:
    if target_row["target_scope"] == "observational_law":
        return 0.0
    dose = float(target_row["target_dose"])
    bandwidth = float(target_row["bandwidth"]) if pd.notna(target_row["bandwidth"]) else None
    endpoint_audited = bool(target_unit["endpoint_audited"].iloc[0])
    a_star = np.array([float(target_row["A_star"])])
    q_target = intervention_density(a_star, dose, bandwidth, endpoint_audited)[0]
    g_target = propensity.density(target_unit, a=a_star)[0]
    return float(np.log(q_target) - np.log(g_target)) if q_target > 0 and g_target > 0 and np.isfinite(g_target) else -np.inf


def _method_eligibility(
    case_row: pd.Series,
    target_row: pd.Series,
    method: str,
    safe_count: int,
    treatment_mode: str,
) -> tuple[bool, str]:
    if str(target_row["draw_status"]) != "drawn" or not bool(target_row["target_supported_by_generator"]):
        return False, "generator_support_failed"
    if method in {"H1", "H4"}:
        return False, "geographic_heuristic_not_theorem_backed"
    iid_residual = str(case_row["residual_law"]) == "iid_continuous"
    iid_covariates = str(case_row["covariate_law"]) == "iid"
    no_hidden_c2 = str(case_row["confounding"]) != "hidden_C2_negative_control"
    oracle_treatment = treatment_mode == "oracle_treatment"
    if method == "M1":
        exact_exchangeable = (
            iid_residual
            and str(case_row["covariate_law"]) == "iid"
            and str(case_row["target_shift"]) == "none"
            and str(target_row["target_scope"]) == "observational_law"
            and no_hidden_c2
        )
        return exact_exchangeable, "exchangeable_case" if exact_exchangeable else "exchangeability_not_established"
    if method == "M2":
        eligible = iid_residual and iid_covariates and no_hidden_c2 and oracle_treatment
        if not oracle_treatment:
            return False, "estimated_treatment_no_finite_sample_certificate"
        return eligible, "iid_units_oracle_treatment_shift_only" if eligible else "spatial_or_causal_assumption_not_established"
    if method == "M5":
        eligible = iid_residual and iid_covariates and no_hidden_c2 and oracle_treatment and safe_count > 0
        if not oracle_treatment:
            return False, "estimated_treatment_no_finite_sample_certificate"
        return eligible, "iid_units_graph_safe_subset" if eligible else "graph_safe_subset_not_independent_or_causal_assumption_failed"
    return False, "unknown_method"


def _metrics_from_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "case_id",
        "scenario_id",
        "predictor",
        "nuisance_variant",
        "method",
        "target_scope",
        "target_dose_group",
    ]
    for key, group in intervals.groupby(keys, dropna=False):
        returned = group["interval_status"].ne("refused")
        finite = returned & np.isfinite(group["lower"]) & np.isfinite(group["upper"])
        covered_observed = group.loc[returned, "covered_observed"].mean() if returned.any() else np.nan
        covered_latent = group.loc[returned, "covered_latent"].mean() if returned.any() else np.nan
        widths = group.loc[finite, "upper"] - group.loc[finite, "lower"]
        rows.append(
            dict(zip(keys, key))
            | {
                "requested_n": len(group),
                "returned_n": int(returned.sum()),
                "finite_n": int(finite.sum()),
                "refusal_rate": float(1.0 - returned.mean()),
                "computational_failure_rate": float(group["computational_failure"].mean()),
                "theorem_eligibility_rate": float(group["theorem_eligible_preoutcome"].mean()),
                "false_support_rate": float(group["false_support"].mean()),
                "infinite_rate": float((returned & ~finite).mean()),
                "coverage_observed_returned": float(covered_observed) if pd.notna(covered_observed) else np.nan,
                "coverage_latent_returned": float(covered_latent) if pd.notna(covered_latent) else np.nan,
                "absolute_calibration_error_observed": float(abs(covered_observed - (1.0 - ALPHA))) if pd.notna(covered_observed) else np.nan,
                "absolute_calibration_error_latent": float(abs(covered_latent - (1.0 - ALPHA))) if pd.notna(covered_latent) else np.nan,
                "mean_width_finite": float(widths.mean()) if len(widths) else np.nan,
                "median_ess": float(group.loc[returned, "ess"].median()) if returned.any() else np.nan,
                "median_max_normalized_weight": float(group.loc[returned, "max_normalized_weight"].median()) if returned.any() else np.nan,
                "mean_interval_score_observed_finite": float(group.loc[finite, "interval_score_observed"].mean()) if finite.any() else np.nan,
                "mean_interval_score_latent_finite": float(group.loc[finite, "interval_score_latent"].mean()) if finite.any() else np.nan,
                "interpretation": "single_replication_reference_diagnostic_only_no_MCSE",
            }
        )
    return pd.DataFrame(rows)


def _temporal_audit(temporal: pd.DataFrame) -> pd.DataFrame:
    group = temporal.groupby("unit_id", sort=True)
    role_unique = group["role"].nunique()
    year_count = group["year"].nunique()
    complete_years = group["year"].apply(lambda x: set(map(int, x)) == {2023, 2024, 2025})
    lag_pairs: list[tuple[float, float]] = []
    for _, frame in group:
        values = frame.sort_values("year")["spatiotemporal_residual"].to_numpy(float)
        lag_pairs.extend(zip(values[:-1], values[1:]))
    if lag_pairs:
        first = np.array([x for x, _ in lag_pairs])
        second = np.array([y for _, y in lag_pairs])
        lag1 = float(np.corrcoef(first, second)[0, 1])
    else:
        lag1 = float("nan")
    return pd.DataFrame(
        [
            {
                "case_id": "ST2_TEMPORAL",
                "row_count": len(temporal),
                "unit_count": temporal["unit_id"].nunique(),
                "years_per_unit_min": int(year_count.min()),
                "years_per_unit_max": int(year_count.max()),
                "role_unique_per_unit": bool((role_unique == 1).all()),
                "complete_2023_2025_per_unit": bool(complete_years.all()),
                "row_split_forbidden_all": bool(temporal["row_split_forbidden"].all()),
                "lag1_residual_correlation": lag1,
                "interval_path_status": "cross_sectional_proxy_only_temporal_cluster_method_deferred",
            }
        ]
    )


def _s10_audit(units: pd.DataFrame, mapping: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    map_sums = mapping.groupby("unit_180m")["aggregation_weight"].sum()
    source = units[units["case_id"] == "S10_90M"].copy()
    source["unit_90m"] = source.apply(lambda row: f"R{int(row['grid_row']):02d}_C{int(row['grid_col']):02d}", axis=1)
    merged = source.merge(mapping[["unit_90m", "unit_180m", "aggregation_weight"]], on="unit_90m", how="inner", validate="one_to_one")
    columns = [
        "X_spatial_1",
        "X_spatial_2",
        "X_nonlinear_1",
        "X_nonlinear_2",
        "A",
        "outcome_baseline",
        "conditional_mean_at_A",
        "shared_spatial_residual",
        "Y_true_at_A",
        "ue_proxy",
        "measurement_error_at_A",
        "Y_observed_at_A",
    ]
    recomputed = []
    for unit_180m, frame in merged.groupby("unit_180m", sort=True):
        row: dict[str, Any] = {"unit_180m": unit_180m}
        weights = frame["aggregation_weight"].to_numpy(float)
        for column in columns:
            row[column] = float(np.sum(frame[column].to_numpy(float) * weights))
        recomputed.append(row)
    rec = pd.DataFrame(recomputed).set_index("unit_180m")
    archived = paired.set_index("unit_180m")
    common = rec.index.intersection(archived.index)
    max_diff = 0.0
    for column in columns:
        max_diff = max(max_diff, float(np.max(np.abs(rec.loc[common, column] - archived.loc[common, column]))))
    return pd.DataFrame(
        [
            {
                "map_row_count": len(mapping),
                "paired_unit_count": paired["unit_180m"].nunique(),
                "map_weight_sum_max_abs_error": float(np.max(np.abs(map_sums - 1.0))),
                "recomputed_paired_unit_count": len(rec),
                "max_abs_aggregate_difference": max_diff,
                "treatment_likelihood_status": "not_closed_form_diagnostic_only",
                "theorem_branch_support": "S10_180M_support_level_rerun",
            }
        ]
    )


def run_reference(
    stage3a_zip: Path,
    stage3b_zip: Path,
    stage3b_source_zip: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    package_root = Path(__file__).resolve().parents[2]
    safe_prepare_output_dir(output_dir, package_root, overwrite)
    input_audit = verify_frozen_inputs(stage3a_zip, stage3b_zip, stage3b_source_zip)
    stage3a = load_stage3a(stage3a_zip)
    data = load_stage3b(stage3b_zip)

    allowed = set(data.feature_contract["allowed_observed_nuisance_features"])
    if set(FEATURES) != allowed:
        raise Stage3CError(f"Feature whitelist mismatch: code={FEATURES}, contract={sorted(allowed)}")

    units = data.units.copy()
    targets = data.targets.copy()
    targets["query_id"] = _target_query_id(targets)
    if targets["query_id"].duplicated().any():
        raise Stage3CError("Duplicate target query IDs")

    interval_rows: list[dict[str, Any]] = []
    nuisance_rows: list[dict[str, Any]] = []
    propensity_rows: list[dict[str, Any]] = []
    propensity_norm_parts: list[pd.DataFrame] = []
    ratio_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    spatial_diag_rows: list[dict[str, Any]] = []
    semivariogram_parts: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    calibration_audit_rows: list[dict[str, Any]] = []
    computation_failures: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    # Target-design ratio is audited as separate N2/N3 infrastructure and never
    # injected into the reference methods.
    for case_id, case_units in units.groupby("case_id", sort=True):
        scenario_id = str(case_units["scenario_id"].iloc[0])
        source_scenario, base_seed = frozen_nuisance_seed(stage3a.seeds, scenario_id, 1)
        ratio_seed = derived_seed(base_seed, "TARGET_DESIGN_RATIO")
        observational = case_units[case_units["role"] == "nuisance_training"]
        target_design = case_units[case_units["role"] == "support_audit"]
        evaluation = case_units[case_units["role"].isin(["calibration", "test_target"])].copy()
        model = TargetDesignRatioModel(ratio_seed).fit(observational, target_design)
        estimated = model.ratio(evaluation)
        truth = evaluation["oracle_target_design_ratio"].to_numpy(float)
        ratio_rows.append(
            {
                "case_id": case_id,
                "fit_id": model.fit_id,
                "training_data_hash": model.training_data_hash,
                "frozen_source_scenario": source_scenario,
                "frozen_nuisance_seed": base_seed,
                "derived_ratio_seed": ratio_seed,
                "feature": "X_nonlinear_1",
                "raw_coordinates_excluded": True,
                "used_by_reference_methods": False,
                "hyperparameters_json": json.dumps(model.hyperparameters, sort_keys=True, separators=(",", ":")),
                "rmse_log_ratio": float(np.sqrt(np.mean((np.log(estimated) - np.log(truth)) ** 2))),
                "mean_abs_log_ratio_error": float(np.mean(np.abs(np.log(estimated) - np.log(truth)))),
                "n_observational_train": len(observational),
                "n_target_train": len(target_design),
            }
        )

    for case_id in sorted(units["case_id"].unique()):
        case_units = units[units["case_id"] == case_id].copy().reset_index(drop=True)
        case_row = data.cases[data.cases["case_id"] == case_id].iloc[0]
        scenario_id = str(case_row["scenario_id"])
        source_scenario, base_seed = frozen_nuisance_seed(stage3a.seeds, scenario_id, 1)
        seeds = {
            "rf": derived_seed(base_seed, "OUTCOME_RF"),
            "xgb": derived_seed(base_seed, "OUTCOME_XGB"),
            "propensity": derived_seed(base_seed, "MIXED_PROPENSITY_ESTIMATED"),
            "propensity_misspecified": derived_seed(base_seed, "MIXED_PROPENSITY_MISSPECIFIED"),
        }
        seed_rows.append(
            {
                "case_id": case_id,
                "scenario_id": scenario_id,
                "source_scenario_id": source_scenario,
                "replication": 1,
                "frozen_nuisance_seed": base_seed,
                **{f"derived_{key}_seed": value for key, value in seeds.items()},
            }
        )

        case_targets_all = targets[targets["case_id"] == case_id].copy().reset_index(drop=True)
        case_targets = _reference_target_subset(case_targets_all, case_units)
        training = case_units[case_units["role"] == "nuisance_training"].copy()
        support_audit = case_units[case_units["role"] == "support_audit"].copy()
        calibration = case_units[case_units["role"] == "calibration"].copy().reset_index(drop=True)
        target_units = _unit_lookup(case_units, case_targets)
        calibration_nodes = calibration["node_index"].to_numpy(int)
        target_nodes = np.sort(case_units.loc[case_units["role"] == "test_target", "node_index"].unique())
        true_edges = data.true_edges[data.true_edges["case_id"] == case_id]
        fitted_edges = data.fitted_edges[data.fitted_edges["case_id"] == case_id]
        graph_true = build_graph_cache(case_units, true_edges, target_nodes)
        graph_fitted = build_graph_cache(case_units, fitted_edges, target_nodes)
        graph_rows.append(
            {
                "case_id": case_id,
                "true_edge_hash": graph_true.edge_hash,
                "fitted_edge_hash": graph_fitted.edge_hash,
                "true_edge_count": len(true_edges),
                "fitted_edge_count": len(fitted_edges),
            }
        )

        propensity = MixedPropensityModel("estimated", seeds["propensity"]).fit(training)
        propensity_misspecified = MixedPropensityModel("misspecified", seeds["propensity_misspecified"]).fit(training)
        for mode, model in [("estimated", propensity), ("misspecified_diagnostic_only", propensity_misspecified)]:
            diagnostics = model.diagnostics(support_audit)
            propensity_rows.append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "propensity_fit_id": model.fit_id,
                    "training_data_hash": model.training_data_hash,
                    "frozen_source_scenario": source_scenario,
                    "frozen_nuisance_seed": base_seed,
                    "derived_model_seed": model.random_state,
                    "training_role": "nuisance_training",
                    "support_audit_used_for_fit": False,
                    "target_used_for_fit": False,
                    "n_train": len(training),
                    "category_hyperparameters_json": json.dumps(model.category.hyperparameters, sort_keys=True, separators=(",", ":")),
                    "interior_hyperparameters_json": json.dumps(model.interior.hyperparameters, sort_keys=True, separators=(",", ":")),
                    **diagnostics,
                }
            )
        norm_audit = propensity.normalization_audit(support_audit.sort_values("unit_id").head(3)).copy()
        norm_audit.insert(0, "case_id", case_id)
        norm_audit.insert(1, "propensity_fit_id", propensity.fit_id)
        propensity_norm_parts.append(norm_audit)

        predictors = ["rf"] + (["xgb"] if case_id in XGB_CASES else [])
        for predictor in predictors:
            started = time.time()
            outcome = OutcomeRegressor(predictor, seeds[predictor]).fit(training)
            calibration_prediction = outcome.predict_factual(calibration)
            scores = np.abs(calibration["Y_observed_at_A"].to_numpy(float) - calibration_prediction)
            sort_index = np.argsort(scores, kind="mergesort")
            sorted_scores = scores[sort_index]
            target_prediction = np.full(len(case_targets), np.nan, dtype=float)
            prediction_mask = (
                (case_targets["draw_status"].astype(str) == "drawn")
                & case_targets["target_supported_by_generator"].astype(bool)
                & case_targets["A_star"].notna()
            ).to_numpy(bool)
            if prediction_mask.any():
                target_prediction[prediction_mask] = outcome.predict_target(
                    target_units.loc[prediction_mask].reset_index(drop=True),
                    case_targets.loc[prediction_mask, "A_star"].to_numpy(float),
                )
            nuisance_rows.append(
                {
                    "case_id": case_id,
                    "predictor": predictor,
                    "outcome_fit_id": outcome.fit_id,
                    "training_data_hash": outcome.training_data_hash,
                    "frozen_source_scenario": source_scenario,
                    "frozen_nuisance_seed": base_seed,
                    "derived_model_seed": seeds[predictor],
                    "training_role": "nuisance_training",
                    "support_audit_used_for_fit": False,
                    "target_used_for_fit": False,
                    "n_train": len(training),
                    "n_calibration": len(calibration),
                    "hyperparameters_json": json.dumps(outcome.hyperparameters, sort_keys=True, separators=(",", ":")),
                    **outcome_diagnostics(outcome, support_audit),
                }
            )

            # Residual spatial diagnostics on support-audit data.  These are not
            # conformal deficits and do not tune the reference methods.
            audit_residual = support_audit["Y_observed_at_A"].to_numpy(float) - outcome.predict_factual(support_audit)
            residual_series = pd.Series(audit_residual, index=support_audit["node_index"].to_numpy(int))
            for graph_label, graph_cache, edge_frame in [
                ("true_graph", build_graph_cache(case_units, true_edges, np.array([], dtype=int)), true_edges),
                ("fitted_graph", build_graph_cache(case_units, fitted_edges, np.array([], dtype=int)), fitted_edges),
            ]:
                m_i = morans_i(residual_series, edge_frame)
                spatial_diag_rows.append(
                    {
                        "case_id": case_id,
                        "predictor": predictor,
                        "graph_mode": graph_label,
                        "morans_I": m_i,
                        "diagnostic_only": True,
                        "not_a_conformal_deficit": True,
                    }
                )
                variogram = graph_semivariogram(residual_series, graph_cache)
                variogram.insert(0, "case_id", case_id)
                variogram.insert(1, "predictor", predictor)
                variogram.insert(2, "graph_mode", graph_label)
                variogram["diagnostic_only"] = True
                semivariogram_parts.append(variogram)

            geographic_cache: dict[int, np.ndarray] = {}
            safe_cache: dict[tuple[str, int], np.ndarray] = {}
            calibration_xy = calibration[["x_coord", "y_coord"]].to_numpy(float)
            target_xy_lookup = case_units.set_index("node_index")[["x_coord", "y_coord"]]
            for node in target_nodes:
                geographic_cache[int(node)] = geographic_distance_logweights(
                    target_xy_lookup.loc[int(node)].to_numpy(float),
                    calibration_xy,
                    GEOGRAPHIC_BANDWIDTH,
                )
            for graph_mode, graph in [("true_graph", graph_true), ("fitted_graph", graph_fitted)]:
                for node in target_nodes:
                    selected = deterministic_graph_safe_set(graph, int(node), calibration_nodes)
                    if not verify_independent_set(graph, int(node), selected):
                        raise Stage3CError(f"Invalid graph-safe set in {case_id}/{graph_mode}/{node}")
                    safe_cache[(graph_mode, int(node))] = np.isin(calibration_nodes, selected)

            dose_cache: dict[tuple[str, str, float], np.ndarray] = {}
            audit_query_id = None
            supported_for_audit = case_targets[
                (case_targets["draw_status"].astype(str) == "drawn")
                & case_targets["target_supported_by_generator"].astype(bool)
            ]
            if len(supported_for_audit):
                audit_query_id = str(supported_for_audit.iloc[0]["query_id"])

            for i, target_row in case_targets.iterrows():
                query_id = str(target_row["query_id"])
                target_unit = target_units.iloc[[i]]
                node = int(target_unit["node_index"].iloc[0])
                center = float(target_prediction[i]) if pd.notna(target_prediction[i]) else np.nan
                supported = (
                    str(target_row["draw_status"]) == "drawn"
                    and bool(target_row["target_supported_by_generator"])
                    and pd.notna(target_row["A_star"])
                )
                for nuisance_variant, (treatment_mode, graph_mode) in NUISANCE_VARIANTS.items():
                    graph = graph_true if graph_mode == "true_graph" else graph_fitted
                    if not supported:
                        for method in METHODS:
                            eligible, eligibility_reason = _method_eligibility(case_row, target_row, method, 0, treatment_mode)
                            interval_rows.append(
                                {
                                    "query_id": query_id,
                                    "case_id": case_id,
                                    "scenario_id": target_row["scenario_id"],
                                    "replication": int(target_row["replication"]),
                                    "unit_id": target_row["unit_id"],
                                    "target_scope": target_row["target_scope"],
                                    "target_dose": float(target_row["target_dose"]),
                                    "target_dose_group": target_row["target_dose_group"],
                                    "A_star": np.nan,
                                    "predictor": predictor,
                                    "nuisance_variant": nuisance_variant,
                                    "treatment_mode": treatment_mode,
                                    "graph_mode": graph_mode,
                                    "method": method,
                                    "method_family_status": "heuristic_ablation" if method in {"H1", "H4"} else "principal_baseline",
                                    "outcome_fit_id": outcome.fit_id,
                                    "propensity_fit_id": "not_used" if method in {"M1", "H1"} else ("oracle" if treatment_mode == "oracle_treatment" else propensity.fit_id),
                                    "graph_hash": (
                                        "not_used" if method in {"M1", "M2"}
                                        else "geographic_distance_no_graph" if method in {"H1", "H4"}
                                        else graph.edge_hash
                                    ),
                                    "center": np.nan,
                                    "lower": np.nan,
                                    "upper": np.nan,
                                    "quantile": np.nan,
                                    "interval_status": "refused",
                                    "refusal_code": "R02_ENDPOINT_NOT_AUDITED",
                                    "positive_weight_count": 0,
                                    "ess": 0.0,
                                    "max_normalized_weight": 1.0,
                                    "target_normalized_weight": 1.0,
                                    "max_all_normalized_weight": 1.0,
                                    "log_weight_range": np.inf,
                                    "Y_observed_at_A_star": np.nan,
                                    "Y_true_at_A_star": np.nan,
                                    "covered_observed": False,
                                    "covered_latent": False,
                                    "interval_score_observed": np.inf,
                                    "interval_score_latent": np.inf,
                                    "target_design_ratio_used": False,
                                    "threshold_status": "deferred_until_20_replication_pilot",
                                    "theorem_eligible_preoutcome": eligible,
                                    "eligibility_reason": eligibility_reason,
                                    "false_support": False,
                                    "computational_failure": False,
                                    "temporal_interpretation": "cross_sectional_proxy_only" if scenario_id == "ST2" else "not_temporal_stress",
                                }
                            )
                        continue

                    cache_key = (treatment_mode, str(target_row["target_scope"]), float(target_row["target_dose"]))
                    if cache_key not in dose_cache:
                        if treatment_mode == "oracle_treatment":
                            dose_cache[cache_key] = _oracle_calibration_logweights(data.treatment_truth, target_row, calibration)
                        else:
                            dose_cache[cache_key] = _estimated_logweights(propensity, calibration, target_row, target_unit)[0]
                    dose_log = dose_cache[cache_key]
                    target_dose_log = (
                        _oracle_target_logweight(target_row)
                        if treatment_mode == "oracle_treatment"
                        else _estimated_target_logweight(propensity, target_row, target_unit)
                    )
                    spatial_log = geographic_cache[node]
                    safe_mask = safe_cache[(graph_mode, node)]
                    method_log_weights = {
                        "M1": np.zeros(len(calibration)),
                        "M2": dose_log,
                        "H1": spatial_log,
                        "H4": dose_log + spatial_log,
                        "M5": np.where(safe_mask, dose_log, -np.inf),
                    }
                    method_target_log_weight = {
                        "M1": 0.0,
                        "M2": target_dose_log,
                        "H1": 0.0,
                        "H4": target_dose_log,
                        "M5": target_dose_log,
                    }
                    y_observed = float(target_row["Y_observed_at_A_star"])
                    y_true = float(target_row["Y_true_at_A_star"])
                    for method in METHODS:
                        safe_count = int(safe_mask.sum()) if method == "M5" else len(calibration)
                        eligible, eligibility_reason = _method_eligibility(case_row, target_row, method, safe_count, treatment_mode)
                        try:
                            result = build_interval(
                                center,
                                sorted_scores,
                                method_log_weights[method][sort_index],
                                method_target_log_weight[method],
                                ALPHA,
                            )
                            computational_failure = False
                        except Exception as exc:  # pragma: no cover - defensive release path
                            computational_failure = True
                            computation_failures.append(
                                {
                                    "query_id": query_id,
                                    "case_id": case_id,
                                    "predictor": predictor,
                                    "nuisance_variant": nuisance_variant,
                                    "method": method,
                                    "exception_type": type(exc).__name__,
                                    "exception_message": str(exc),
                                }
                            )
                            result = build_interval(center, np.array([]), np.array([]), -np.inf, ALPHA)
                        returned = result.interval_status != "refused"
                        covered_observed = bool(returned and result.lower <= y_observed <= result.upper)
                        covered_latent = bool(returned and result.lower <= y_true <= result.upper)
                        false_support = bool(returned and not bool(target_row["target_supported_by_generator"]))
                        interval_rows.append(
                            {
                                "query_id": query_id,
                                "case_id": case_id,
                                "scenario_id": target_row["scenario_id"],
                                "replication": int(target_row["replication"]),
                                "unit_id": target_row["unit_id"],
                                "target_scope": target_row["target_scope"],
                                "target_dose": float(target_row["target_dose"]),
                                "target_dose_group": target_row["target_dose_group"],
                                "A_star": float(target_row["A_star"]),
                                "predictor": predictor,
                                "nuisance_variant": nuisance_variant,
                                "treatment_mode": treatment_mode,
                                "graph_mode": graph_mode,
                                "method": method,
                                "method_family_status": "heuristic_ablation" if method in {"H1", "H4"} else "principal_baseline",
                                "outcome_fit_id": outcome.fit_id,
                                "propensity_fit_id": "not_used" if method in {"M1", "H1"} else ("oracle" if treatment_mode == "oracle_treatment" else propensity.fit_id),
                                "graph_hash": (
                                        "not_used" if method in {"M1", "M2"}
                                        else "geographic_distance_no_graph" if method in {"H1", "H4"}
                                        else graph.edge_hash
                                    ),
                                "center": result.center,
                                "lower": result.lower,
                                "upper": result.upper,
                                "quantile": result.quantile,
                                "interval_status": result.interval_status,
                                "refusal_code": result.refusal_code,
                                "positive_weight_count": result.positive_weight_count,
                                "ess": result.ess,
                                "max_normalized_weight": result.max_normalized_weight,
                                "target_normalized_weight": result.target_normalized_weight,
                                "max_all_normalized_weight": result.max_all_normalized_weight,
                                "log_weight_range": result.log_weight_range,
                                "Y_observed_at_A_star": y_observed,
                                "Y_true_at_A_star": y_true,
                                "covered_observed": covered_observed,
                                "covered_latent": covered_latent,
                                "interval_score_observed": interval_score(result.lower, result.upper, y_observed, ALPHA),
                                "interval_score_latent": interval_score(result.lower, result.upper, y_true, ALPHA),
                                "target_design_ratio_used": False,
                                "threshold_status": "deferred_until_20_replication_pilot",
                                "theorem_eligible_preoutcome": eligible,
                                "eligibility_reason": eligibility_reason,
                                "false_support": false_support,
                                "computational_failure": computational_failure,
                                "temporal_interpretation": "cross_sectional_proxy_only" if scenario_id == "ST2" else "not_temporal_stress",
                            }
                        )
                        if predictor == "rf" and query_id == audit_query_id:
                            original_logw = method_log_weights[method]
                            for calibration_index, calibration_row in calibration.iterrows():
                                calibration_audit_rows.append(
                                    {
                                        "query_id": query_id,
                                        "case_id": case_id,
                                        "scenario_id": scenario_id,
                                        "predictor": predictor,
                                        "nuisance_variant": nuisance_variant,
                                        "method": method,
                                        "calibration_unit_id": calibration_row["unit_id"],
                                        "calibration_node": int(calibration_row["node_index"]),
                                        "score": float(scores[calibration_index]),
                                        "log_weight": float(original_logw[calibration_index]),
                                        "target_log_weight": float(method_target_log_weight[method]),
                                        "archived_quantile": float(result.quantile),
                                        "archived_status": result.interval_status,
                                        "center": center,
                                        "alpha": ALPHA,
                                    }
                                )
            runtime_rows.append(
                {
                    "case_id": case_id,
                    "predictor": predictor,
                    "elapsed_seconds": time.time() - started,
                }
            )

    intervals = pd.DataFrame(interval_rows)
    metrics = _metrics_from_intervals(intervals)
    nuisance = pd.DataFrame(nuisance_rows)
    propensity_diagnostics = pd.DataFrame(propensity_rows)
    propensity_normalization = pd.concat(propensity_norm_parts, ignore_index=True)
    ratio = pd.DataFrame(ratio_rows)
    graph_diagnostics = pd.DataFrame(graph_rows)
    spatial_diagnostics = pd.DataFrame(spatial_diag_rows)
    semivariogram = pd.concat(semivariogram_parts, ignore_index=True)
    runtime = pd.DataFrame(runtime_rows)
    calibration_audit = pd.DataFrame(calibration_audit_rows)
    temporal_audit = _temporal_audit(data.temporal)
    s10_audit = _s10_audit(units, data.s10_map, data.s10_paired)
    seeds_used = pd.DataFrame(seed_rows)
    computational_failure_log = pd.DataFrame(computation_failures)
    if computational_failure_log.empty:
        computational_failure_log = pd.DataFrame(
            columns=["query_id", "case_id", "predictor", "nuisance_variant", "method", "exception_type", "exception_message"]
        )

    refusals = intervals[intervals["interval_status"] == "refused"][
        [
            "query_id",
            "case_id",
            "scenario_id",
            "unit_id",
            "target_scope",
            "target_dose_group",
            "predictor",
            "nuisance_variant",
            "method",
            "refusal_code",
            "eligibility_reason",
        ]
    ].copy()
    support = intervals[
        [
            "query_id",
            "case_id",
            "unit_id",
            "target_scope",
            "target_dose_group",
            "predictor",
            "nuisance_variant",
            "method",
            "positive_weight_count",
            "ess",
            "max_normalized_weight",
            "target_normalized_weight",
            "max_all_normalized_weight",
            "log_weight_range",
            "interval_status",
            "theorem_eligible_preoutcome",
            "false_support",
        ]
    ].copy()

    method_status = pd.DataFrame(
        [
            ["M1", "standard_split_conformal", "implemented_reference", "principal_baseline"],
            ["M2", "dose_only_weighted_conformal", "implemented_reference", "principal_baseline"],
            ["M3", "spatial_only_generalized_conformal", "deferred_to_stage3d_graph_local_residual_law", "principal_method"],
            ["M4", "naive_product_using_final_spatial_law", "deferred_until_M3_spatial_law_exists", "principal_method"],
            ["M5", "conservative_graph_safe_weighted_conformal", "implemented_reference_with_sampling_eligibility_flag", "principal_baseline"],
            ["M6", "full_geodose_cp", "deferred_stage3d_stage3e", "proposed_method"],
            ["H1", "geographic_euclidean_distance_heuristic", "implemented_reference", "ABL7_heuristic"],
            ["H4", "naive_product_M2_times_H1", "implemented_reference", "ABL4_plus_ABL7_heuristic"],
        ],
        columns=["method_id", "method_name", "implementation_status", "scientific_role"],
    )

    metric_applicability = pd.DataFrame(
        [
            ["MET02", "coverage_mcse", "deferred", "requires 20 independent pilot replications"],
            ["MET20", "raw_set_components", "not_applicable", "M6 only"],
            ["MET23", "spatial_leakage_optimism", "not_applicable", "MineDoseBench and real-data split comparison"],
        ],
        columns=["metric_id", "metric_name", "status", "reason"],
    )

    outputs = {
        "stage3c_method_intervals.csv.gz": intervals,
        "stage3c_reference_metrics.csv.gz": metrics,
        "stage3c_nuisance_fit_registry.csv.gz": nuisance,
        "stage3c_propensity_diagnostics.csv.gz": propensity_diagnostics,
        "stage3c_propensity_normalization_audit.csv.gz": propensity_normalization,
        "stage3c_target_design_ratio_diagnostics.csv.gz": ratio,
        "stage3c_graph_diagnostics.csv.gz": graph_diagnostics,
        "stage3c_spatial_diagnostics.csv.gz": spatial_diagnostics,
        "stage3c_semivariogram.csv.gz": semivariogram,
        "stage3c_temporal_audit.csv.gz": temporal_audit,
        "stage3c_s10_paired_support_audit.csv.gz": s10_audit,
        "stage3c_runtime.csv.gz": runtime,
        "stage3c_refusal_log.csv.gz": refusals,
        "stage3c_support_diagnostics.csv.gz": support,
        "stage3c_calibration_weight_audit.csv.gz": calibration_audit,
        "stage3c_computational_failure_log.csv.gz": computational_failure_log,
        "stage3c_seed_usage.csv.gz": seeds_used,
        "stage3c_method_status.csv.gz": method_status,
        "stage3c_metric_applicability.csv.gz": metric_applicability,
    }
    for filename, frame in outputs.items():
        write_csv_gz_deterministic(frame, output_dir / filename)

    write_json(output_dir / "stage3c_input_audit.json", input_audit)
    write_json(
        output_dir / "stage3c_environment_inventory.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
            "PyYAML": yaml.__version__,
            "official_runner_enforces_exact_environment": True,
            "script_version": __version__,
            "alpha": ALPHA,
            "geographic_bandwidth": GEOGRAPHIC_BANDWIDTH,
            "reference_predictors": REFERENCE_PREDICTORS,
            "nuisance_variants": NUISANCE_VARIANTS,
            "production_experiments_run": False,
            "single_replication_reference_only": True,
            "M3_implemented": False,
            "M4_implemented": False,
            "M6_implemented": False,
        },
    )

    counts = {
        "validation_cases": int(intervals["case_id"].nunique()),
        "target_queries": int(intervals["query_id"].nunique()),
        "reference_target_units_per_case_max": 5,
        "interval_rows": int(len(intervals)),
        "implemented_reference_procedures": list(METHODS),
        "principal_methods_implemented": ["M1", "M2", "M5"],
        "heuristic_ablations_implemented": ["H1", "H4"],
        "principal_methods_deferred": ["M3", "M4", "M6"],
        "predictors": sorted(intervals["predictor"].unique().tolist()),
        "nuisance_variants": sorted(intervals["nuisance_variant"].unique().tolist()),
        "refusal_rows": int(len(refusals)),
        "infinite_interval_rows": int(np.isinf(intervals["upper"]).sum()),
        "computational_failure_rows": int(intervals["computational_failure"].sum()),
        "production_experiments_run": False,
        "coverage_interpretation": "single_replication_reference_diagnostic_only",
        "temporal_method_status": "cross_sectional_proxy_only_temporal_cluster_method_deferred",
    }
    write_json(output_dir / "stage3c_counts.json", counts)

    execution_files = [
        "README.md",
        "RUN_STAGE3C.bat",
        "RUN_STAGE3C.ps1",
        "requirements_py310.txt",
        "package_stage3c_outputs.py",
        "stage3c_run.py",
        "verify_stage3c.py",
        "tests/test_stage3c.py",
        "configs/stage3c_feature_contract.json",
        "configs/stage3c_method_contract.yaml",
        "configs/stage3c_pilot_threshold_plan.yaml",
        "src/geodose_stage3c/__init__.py",
        "src/geodose_stage3c/io.py",
        "src/geodose_stage3c/models.py",
        "src/geodose_stage3c/weights.py",
        "src/geodose_stage3c/graph.py",
        "src/geodose_stage3c/methods.py",
        "src/geodose_stage3c/runner.py",
    ]
    tree_hash, file_hashes = execution_tree_hash(package_root, execution_files)
    write_json(
        output_dir / "stage3c_source_hashes.json",
        {"package_tree_sha256": tree_hash, "files": file_hashes},
    )
    manifest = {
        "stage": "3C-reference",
        "script_version": __version__,
        "package_tree_sha256": tree_hash,
        "outputs": output_manifest(output_dir, {"stage3c_manifest.json", "STAGE3C_REFERENCE_VERIFICATION.json"}),
    }
    write_json(output_dir / "stage3c_manifest.json", manifest)
    return counts
