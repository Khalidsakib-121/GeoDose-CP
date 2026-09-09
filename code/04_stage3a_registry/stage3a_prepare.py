from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import platform
import json
import shutil
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geodose_stage3a.core import *


def validate_stage2a(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    p = input_root / "stage2a"
    required = [
        "mine_blocks_90m.gpkg",
        "block_attributes.parquet",
        "block_graph_edges.csv",
        "leave_one_mine_out_splits.csv",
        "stage2_manifest.json",
        "STAGE2A_VERIFICATION.json",
    ]
    for name in required:
        require((p / name).exists(), f"Missing Stage 2A input: {name}")

    manifest = json.loads((p / "stage2_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((p / "STAGE2A_VERIFICATION.json").read_text(encoding="utf-8"))
    require(verification.get("status") == "verified_complete", "Stage 2A is not verified complete")
    require(verification.get("graph_frozen_before_dea") is True, "Stage 2A graph was not frozen before DEA")

    for name, info in manifest["outputs"].items():
        path = p / name
        require(path.exists(), f"Stage 2A manifest output is missing: {name}")
        require(sha256_file(path) == info["sha256"], f"Stage 2A checksum mismatch: {name}")

    blocks = gpd.read_file(p / "mine_blocks_90m.gpkg", layer="blocks_90m", ignore_geometry=True)
    edges = pd.read_csv(p / "block_graph_edges.csv", encoding="utf-8-sig")
    splits = pd.read_csv(p / "leave_one_mine_out_splits.csv", encoding="utf-8-sig")

    require(len(blocks) == EXPECTED_BLOCKS, "Stage 2A block count mismatch")
    require(blocks["block_id"].is_unique, "Duplicate Stage 2A block IDs")
    require(blocks["MineID"].nunique() == EXPECTED_MINES, "Stage 2A mine count mismatch")
    require(len(edges) == EXPECTED_EDGES, "Stage 2A edge count mismatch")
    require(not edges[["source_block_id", "target_block_id"]].duplicated().any(), "Duplicate Stage 2A graph edge")
    require((edges["source_block_id"] < edges["target_block_id"]).all(), "Noncanonical Stage 2A edge")

    ids = set(blocks["block_id"].astype(str))
    require(set(edges["source_block_id"]).issubset(ids), "Unknown source graph endpoint")
    require(set(edges["target_block_id"]).issubset(ids), "Unknown target graph endpoint")
    mine_map = blocks.set_index("block_id")["MineID"].to_dict()
    require(
        all(mine_map[a] == mine_map[b] for a, b in zip(edges["source_block_id"], edges["target_block_id"])),
        "Cross-mine graph edge",
    )

    degree = (
        pd.concat([edges["source_block_id"], edges["target_block_id"]])
        .value_counts()
        .reindex(blocks["block_id"], fill_value=0)
        .to_numpy()
    )
    require(np.array_equal(degree.astype(int), blocks["graph_degree"].astype(int).to_numpy()), "Graph degree mismatch")
    require(len(splits) == EXPECTED_FOLDS * EXPECTED_BLOCKS, "LOMO row count mismatch")
    require(splits["fold_id"].nunique() == EXPECTED_FOLDS, "LOMO fold count mismatch")
    require(blocks["graph_component_id"].astype(str).str.contains("_CC", regex=False).all(), "Malformed component IDs")
    return blocks, edges, splits, manifest


def validate_stage2b(
    input_root: Path, blocks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    path = input_root / "stage2b" / "GeoDose_Stage2B_OUTPUTS_FINAL.zip"
    require(path.exists(), "Missing final Stage 2B ZIP")
    require(sha256_file(path) == EXPECTED_STAGE2B_ARCHIVE_SHA256, "Unexpected Stage 2B final archive hash")

    with zipfile.ZipFile(path) as zf:
        require(zf.testzip() is None, "Stage 2B ZIP integrity failed")
        names = normalized_zip_map(zf)
        mandatory = [
            "dea_block_year_prescreen.csv.gz",
            "dea_mine_prescreen_summary.csv",
            "selected_three_mines.csv",
            "stage2b_manifest.json",
            "STAGE2B_VERIFICATION.json",
            "stage2b_config.json",
            "dea_environment_inventory.json",
            "Stage2B_Acceptance_Decision.md",
        ]
        for name in mandatory:
            require(name in names, f"Missing Stage 2B archive member: {name}")

        manifest = read_stage2b_json(zf, names, "stage2b_manifest.json")
        verification = read_stage2b_json(zf, names, "STAGE2B_VERIFICATION.json")
        config = read_stage2b_json(zf, names, "stage2b_config.json")
        require(verification.get("status") == "verified_complete", "Stage 2B is not verified complete")
        require(verification.get("selection_uses_pv_outcome") is False, "Stage 2B selection was not outcome-blind")

        for name, info in manifest["outputs"].items():
            require(name in names, f"Stage 2B manifest member missing: {name}")
            raw = zf.read(names[name])
            require(hashlib.sha256(raw).hexdigest() == info["sha256"], f"Stage 2B internal hash mismatch: {name}")

        by = read_stage2b_csv(zf, names, "dea_block_year_prescreen.csv.gz")
        summary = read_stage2b_csv(zf, names, "dea_mine_prescreen_summary.csv")
        selected = read_stage2b_csv(zf, names, "selected_three_mines.csv")

    require(len(by) == EXPECTED_BLOCK_YEARS, "Stage 2B row count mismatch")
    require(by["block_id"].nunique() == EXPECTED_BLOCKS, "Stage 2B block count mismatch")
    require(set(by["year"].astype(int).unique()) == set(EXPECTED_YEARS), "Stage 2B year set mismatch")
    require(not by.duplicated(["block_id", "year"]).any(), "Duplicate Stage 2B block-year")
    require(set(by["block_id"].astype(str)) == set(blocks["block_id"].astype(str)), "Stage 2A/2B block ID mismatch")
    require(len(selected) == 3 and selected["MineID"].nunique() == 3, "Stage 2B selected-mine table invalid")
    all_year = by.assign(block_year_eligible=as_bool(by["block_year_eligible"]))
    eligible = int(all_year.groupby("block_id")["block_year_eligible"].all().sum())
    require(eligible == EXPECTED_ALL_YEAR_ELIGIBLE, "Stage 2B all-year eligibility count mismatch")
    return by, summary, selected, manifest, config


def scenario_registry() -> pd.DataFrame:
    rows = [
        ("S1", "Weak shift; negligible dependence", "0.00", "weak", "good", "true", "none", "baseline reduction"),
        ("S2", "Spatial dependence only", "0.60", "none", "good", "true", "none", "spatial correction sufficiency"),
        ("S3", "Treatment shift only", "0.00", "strong", "moderate", "true", "none", "dose correction sufficiency"),
        ("S4", "Combined dependence and treatment shift", "0.00;0.20;0.40;0.60;0.80", "strong", "moderate", "true", "none", "hero coverage-width curve"),
        ("S5", "Poor overlap and concentrated weights", "0.40", "tail target", "poor", "true", "none", "ESS/support/refusal"),
        ("S6", "Misspecified dependency graph", "0.60", "moderate", "moderate", "rook or 30-50% edge omission", "none", "graph robustness"),
        ("S7", "Benign exchangeable setting", "0.00", "none", "good", "true", "none", "no artificial advantage"),
        ("S8", "Severe non-overlap / too little information", "0.60", "outside/near-boundary", "severe", "true", "none", "mandatory refusal"),
        ("S9", "Treatment-correlated EO error", "0.40", "moderate", "moderate", "true", "lambda_A=0;0.5;1.0", "observed vs latent target"),
        ("S10", "Alternative spatial support / MAUP", "0.40", "moderate", "moderate", "true", "none", "90m vs anchored 180m"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "scenario_id",
            "scenario_name",
            "spatial_rho",
            "target_shift",
            "overlap",
            "fitted_graph",
            "measurement_error",
            "primary_purpose",
        ],
    ).assign(registry_status="architecture_frozen_pre_pilot")


def method_registry() -> pd.DataFrame:
    rows = [
        ("M1", "Standard split conformal", "none", "none", "unweighted split quantile", "registered_not_implemented"),
        ("M2", "Dose-only weighted conformal", "mixed treatment ratio", "none", "weighted quantile", "registered_not_implemented"),
        ("M3", "Spatial-only generalized conformal", "none", "graph-local residual law", "generalized conformal", "registered_not_implemented"),
        ("M4", "Naive dose × spatial product", "mixed treatment ratio", "spatial weight", "ordinary weighted quantile; no certified deficit", "registered_not_implemented"),
        ("M5", "Conservative graph-safe weighted conformal", "mixed treatment ratio", "independent-set or separated blocks", "graph-safe weighted quantile", "registered_not_implemented"),
        ("M6", "Full GeoDose-CP", "mixed treatment ratio", "exact G2/G3 or scalable N2/N3", "certified calibration and refusal", "registered_not_implemented"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "method_id",
            "method_name",
            "treatment_adjustment",
            "spatial_adjustment",
            "calibration_rule",
            "implementation_status",
        ],
    )




def factor_registry() -> pd.DataFrame:
    rows = [
        ("F01", "spatial_dependence", "rho=0;0.2;0.4;0.6;0.8", "S1-S10 as registered", "S4 hero curve fixed; remaining combinations finalized after pilot"),
        ("F02", "treatment_target_shift", "none;weak;moderate;strong;tail;outside", "S1;S3;S4;S5;S7;S8", "operational DGP mapping required in Stage3B"),
        ("F03", "overlap", "good;moderate;poor;severe", "S1-S10", "support and refusal thresholds freeze after pilot"),
        ("F04", "confounding_and_identification", "observed_none;observed_moderate;observed_strong;hidden_C2_negative_control", "controlled and MineDoseBench", "hidden negative control must show conformal validity does not repair causal bias"),
        ("F05", "outcome_nonlinearity", "linear_reference;nonlinear_primary", "controlled and MineDoseBench", "nonlinear primary; linear reduction check"),
        ("F06", "sample_size", "small;primary;large", "controlled simulation", "exact sizes freeze after runtime pilot"),
        ("F07", "propensity_specification", "oracle;correct_estimated;misspecified", "controlled simulation", "separates calibration principle from nuisance error"),
        ("F08", "graph_specification", "true;rook;edge_omission;wrong_residual_law", "S6;S8", "misspecification is never relabeled as certified"),
        ("F09", "measurement_error", "none;random;treatment_correlated;substrate_bias", "S9", "observed-product and latent-outcome coverage reported separately"),
        ("F10", "spatial_support", "90m_primary;180m_anchored", "S10", "complete analysis rerun after support change"),
        ("F11", "temporal_dependence", "none;repeated_mine_year", "additional stress test and MineDoseBench", "same-mine repeated-year dependence must be preserved and audited"),
        ("F12", "endpoint_structure", "interior_only_reference;mixed_atoms_0_1", "additional stress test", "genuine atoms preserved; no jitter or clipping"),
    ]
    return pd.DataFrame(rows, columns=["factor_id","factor_name","registered_levels","scenario_scope","freeze_status"])

def nuisance_model_registry() -> pd.DataFrame:
    rows = [
        ("NUI1", "outcome_reference", "RandomForestRegressor", "all S1-S10 and MineDoseBench", "shared across M1-M6 within a split", "registered_not_implemented"),
        ("NUI2", "outcome_flexible", "XGBoost regressor", "S1;S4;S5;S8 and principal MineDoseBench settings", "model-agnostic confirmation", "registered_not_implemented"),
        ("NUI3", "mixed_treatment_oracle", "known endpoint masses plus known interior density", "controlled oracle diagnostics", "exact G2 likelihood and oracle comparisons", "generator_available"),
        ("NUI4", "mixed_treatment_estimated", "three-part endpoint/interior model with normalized interior density", "controlled and MineDoseBench", "exact branch requires theorem-backed likelihood; scalable branch may use target-design ratio", "registered_not_implemented"),
        ("NUI5", "mixed_treatment_misspecified", "prespecified omitted/nonlinear misspecification", "controlled simulation", "nuisance-error stress test", "registered_not_implemented"),
        ("NUI6", "residual_scale_optional", "cross-fitted positive scale model", "sensitivity only when pilot residual diagnostics justify", "not mandatory", "provisional"),
    ]
    return pd.DataFrame(rows, columns=["model_id","model_role","model_class","scenario_scope","scientific_rule","status"])



def ablation_registry() -> pd.DataFrame:
    rows = [
        ("ABL0", "full_geodose_cp", "none", "controlled;MineDoseBench;real demonstration", "reference proposed method"),
        ("ABL1", "without_treatment_correction", "remove treatment-likelihood/transport correction", "controlled", "necessity of treatment-shift correction"),
        ("ABL2", "without_target_design_transport", "set target-design ratio to one in the scalable branch", "controlled;MineDoseBench", "necessity of target-population transport distinct from exact treatment likelihood"),
        ("ABL3", "without_spatial_law", "remove graph-local residual-law correction", "controlled", "necessity of the spatial law"),
        ("ABL4", "naive_product", "replace derived target law with product of marginal dose and spatial weights", "controlled;MineDoseBench", "tests nontriviality beyond weight multiplication"),
        ("ABL5", "without_dependence_penalty", "keep target weights but use ordinary weighted quantile", "controlled", "isolates certified dependence adjustment"),
        ("ABL6", "without_eligibility_refusal", "force output at unsupported targets", "controlled", "tests value of theorem eligibility and abstention"),
        ("ABL7", "geographic_distance_heuristic", "replace residual-law correction with a prespecified distance-decay heuristic", "controlled;MineDoseBench", "tests whether graph-law correction adds value beyond geography alone"),
        ("ABL8", "oracle_treatment", "use true mixed treatment likelihood or true target ratio as route-appropriate", "controlled", "separates treatment-nuisance estimation from structural calibration"),
        ("ABL9", "oracle_spatial_law", "use true graph and residual-law parameters", "controlled", "separates spatial estimation from structural approximation"),
    ]
    return pd.DataFrame(rows, columns=["ablation_id","ablation_name","change","experiment_scope","scientific_purpose"])


def metric_registry() -> pd.DataFrame:
    rows = [
        ("MET01", "marginal_coverage", "primary", "all", "coverage of the registered target population"),
        ("MET02", "coverage_mcse", "mandatory", "controlled;MineDoseBench", "Monte Carlo uncertainty of empirical coverage"),
        ("MET03", "absolute_calibration_error", "primary", "controlled;MineDoseBench", "absolute difference between empirical and nominal coverage"),
        ("MET04", "dose_bin_coverage", "primary", "all known-truth experiments", "validity across supported doses"),
        ("MET05", "mine_or_block_coverage", "primary", "MineDoseBench;real demonstration", "local failure hidden by aggregate coverage"),
        ("MET06", "fifth_percentile_local_coverage", "primary", "controlled;MineDoseBench", "robust local-validity summary without conditional-coverage claim"),
        ("MET07", "mean_width_at_matched_coverage", "primary", "all", "efficiency only at approximately equal coverage"),
        ("MET08", "weighted_interval_score", "primary", "all", "joint calibration and sharpness"),
        ("MET09", "dose_response_rmse", "primary", "controlled;MineDoseBench", "error against known target mean function"),
        ("MET10", "theorem_eligibility_rate", "primary", "all", "fraction of queries satisfying the frozen theorem gates before final outcomes"),
        ("MET11", "effective_calibration_size", "mandatory_diagnostic", "all", "support and weight concentration"),
        ("MET12", "maximum_normalized_weight", "mandatory_diagnostic", "all weighted methods", "weight dominance"),
        ("MET13", "refusal_rate", "primary", "all", "frequency of abstention"),
        ("MET14", "false_support_rate", "primary", "S5;S8", "interval returned when frozen support conditions fail"),
        ("MET15", "coverage_loss_diagnostic_error", "primary", "controlled;MineDoseBench", "agreement between certified/estimated deficit and undercoverage"),
        ("MET16", "observed_product_coverage", "primary", "S9", "coverage for observed EO product target"),
        ("MET17", "latent_outcome_coverage", "sensitivity", "S9", "coverage for latent truth under explicit measurement model"),
        ("MET18", "runtime", "secondary", "all", "computational practicality"),
        ("MET19", "computational_failure_rate", "mandatory", "all", "non-STAC numerical/method failure frequency"),
        ("MET20", "raw_set_component_count_and_hull_inflation", "mandatory", "M6", "preserve disconnected set structure and quantify presentation inflation"),
        ("MET21", "empirical_semivariogram", "diagnostic_only", "controlled;MineDoseBench;real demonstration", "spatial diagnostic; never substituted for a conformal deficit"),
        ("MET22", "morans_I", "diagnostic_only", "controlled;MineDoseBench;real demonstration", "spatial diagnostic; never substituted for a conformal deficit"),
        ("MET23", "spatial_leakage_optimism", "diagnostic", "MineDoseBench;real demonstration", "random/buffered/LOMO comparison with sign convention frozen by metric direction"),
    ]
    return pd.DataFrame(rows, columns=["metric_id","metric_name","priority","experiment_scope","interpretation"])

def reduction_test_registry() -> pd.DataFrame:
    rows = [
        ("R1", "exact_reference_reduction", "exact target transport and exact conditional spatial law imply zero approximation deficit", "Stage3D", "registered_not_implemented"),
        ("R2", "weighted_conformal_reduction", "exchangeable outcomes with target shift reduce to ordinary weighted conformal", "Stage3C/Stage3D", "registered_not_implemented"),
        ("R3", "no_target_shift_reduction", "target-design ratio equals one and N2 criterion reduces to observational spatial approximation", "Stage3E", "registered_not_implemented"),
        ("R4", "ordinary_conformal_reduction", "no target shift and exchangeability reduce to ordinary conformal", "Stage3C", "registered_not_implemented"),
    ]
    return pd.DataFrame(rows, columns=["reduction_id","reduction_name","required_identity","implementation_stage","status"])


def deferred_feature_registry() -> pd.DataFrame:
    rows = [
        ("rainfall_drought_history", "climate", "deferred_minimum_route", "not used unless acquired and temporally audited before Stage3F freeze"),
        ("terrain_slope_roughness_drainage", "terrain", "deferred_minimum_route", "not used unless acquired and frozen before MineDoseBench production"),
        ("soil_texture_ph_carbon_water_capacity", "soil", "deferred_minimum_route", "not used unless acquired and frozen before MineDoseBench production"),
        ("commodity_and_rehabilitation_target", "mine_context", "deferred_minimum_route", "not used unless a complete audited crosswalk is available"),
    ]
    return pd.DataFrame(rows, columns=["feature_name","feature_group","availability_status","use_rule"])


def target_population_contract() -> dict:
    return {
        "controlled_target_population": "uniform over theorem-eligible final-test slots under the frozen simulation split",
        "minedosebench_target_population": "theorem-eligible blocks in the held-out mine under the frozen LOMO role registry",
        "eligibility_timing": "causal support eligibility uses pretreatment/static information and is frozen before target outcomes",
        "eo_quality_gate": "post-assignment EO quality may restrict the observed-product interpretation or trigger R11 but is not a pretreatment confounder",
        "support_stratification": "mapped_rehabilitation_fraction may describe support strata only and never becomes authentic treatment",
        "target_population_freeze_stage": "operational thresholds and exact block eligibility freeze after the Stage3F pilot and before production outcomes",
    }


def environment_inventory() -> dict:
    def version(name: str):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    return {
        "recorded_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "geopandas": version("geopandas"),
            "PyYAML": version("PyYAML"),
            "shapely": version("shapely"),
            "pyogrio": version("pyogrio"),
            "fiona": version("fiona"),
        },
        "internet_used": False,
        "production_inference_run": False,
    }


def refusal_registry() -> list[dict]:
    values = [
        ("R01_UNSUPPORTED_DOSE", "Requested dose lacks audited positivity over the complete intervention support", "structural_frozen_gate"),
        ("R02_ENDPOINT_NOT_AUDITED", "Endpoint is not a genuine audited treatment atom", "structural_frozen_gate"),
        ("R03_LOW_EFFECTIVE_CALIBRATION_SIZE", "Effective calibration size below frozen threshold", "threshold_freeze_after_pilot"),
        ("R04_WEIGHT_CONCENTRATION", "Maximum normalized weight exceeds frozen threshold", "threshold_freeze_after_pilot"),
        ("R05_GRAPH_COMPONENT_TOO_SMALL", "Graph component cannot support requested calibration", "threshold_and_block_rule_freeze_after_pilot"),
        ("R06_SPATIAL_PRECISION_NOT_PD", "Spatial precision or covariance condition is not proper positive definite", "structural_frozen_gate"),
        ("R07_EMPTY_NUISANCE_CERTIFICATE", "Nuisance confidence set is empty, unbounded, or yields no finite deficit", "structural_frozen_gate"),
        ("R08_CERTIFIED_DEFICIT_VACUOUS", "Final certified lower bound is vacuous or below the frozen operational threshold", "structural_alpha_gate_plus_operational_threshold_after_pilot"),
        ("R09_ORBIT_PROPOSAL_NO_SUPPORT", "Monte Carlo or importance proposal lacks full support or required mutual absolute continuity", "structural_frozen_gate"),
        ("R10_NUMERICAL_NONFINITE", "Non-finite weights, quantiles, certificates, or prediction set", "structural_frozen_gate"),
        ("R11_EO_QUALITY_INSUFFICIENT", "Observed-product target lacks the frozen EO-quality support", "quality_threshold_freeze_after_pilot"),
        ("R12_INTERFERENCE_SCOPE_UNSUPPORTED", "Requested interference estimand exceeds the frozen single-unit or sensitivity scope", "structural_scope_gate"),
        ("R13_EXACT_TREATMENT_LIKELIHOOD_UNAVAILABLE", "Exact G2 branch lacks the required mixed treatment likelihood or theorem-equivalent orbit likelihood", "structural_branch_gate"),
        ("R14_NUISANCE_THEOREM_REGIME_MISMATCH", "Invoked nuisance theorem does not match the actual nuisance-training dependence regime", "structural_theorem_gate"),
        ("R15_VECCHIA_CONDITIONS_UNVERIFIED", "N4/Vecchia route requested without every external condition verified", "structural_branch_gate"),
        ("R16_DESIGN_RULE_NOT_FROZEN", "Target population, graph, block, bandwidth, support, candidate, or proposal rule was not frozen before final evaluation", "structural_freeze_gate"),
        ("R17_TARGET_UNIT_OUTSIDE_SUPPORTED_POPULATION", "Target unit is outside the pre-frozen supported target population", "structural_supported_population_gate"),
    ]
    return [
        {"code": code, "description": description, "gate_status": gate_status}
        for code, description, gate_status in values
    ]


def additional_stress_test_registry() -> pd.DataFrame:
    rows = [
        ("ST1", "mixed_endpoint_atoms", "genuine treatment atoms at 0 and 1 plus continuous interior", "Stage3A preview; Stage3B production", "preview_implemented_production_registered"),
        ("ST2", "repeated_mine_year_temporal_dependence", "same-mine repeated-year dependence with no row-wise temporal leakage", "Stage3B/MineDoseBench", "registered_not_implemented"),
        ("ST3", "spatial_confounding_C2_negative_control", "unobserved spatial latent field drives treatment and potential outcomes but is omitted from fitted adjustment", "Stage3B controlled", "registered_not_implemented"),
    ]
    return pd.DataFrame(rows, columns=["stress_id","stress_name","definition","implementation_stage","status"])


def mandatory_gate_registry() -> pd.DataFrame:
    rows = [
        ("GATE01", "supported_target_population", "R17_TARGET_UNIT_OUTSIDE_SUPPORTED_POPULATION", "target unit belongs to the pre-frozen supported population", "pre_final_outcome"),
        ("GATE02", "positivity_complete_intervention_support", "R01_UNSUPPORTED_DOSE", "positivity holds across the full localized intervention support", "pre_final_outcome"),
        ("GATE03", "audited_endpoint_atom", "R02_ENDPOINT_NOT_AUDITED", "endpoint atom exists and is audited as genuine", "pre_final_outcome"),
        ("GATE04", "frozen_design_rules", "R16_DESIGN_RULE_NOT_FROZEN", "graph, block, bandwidth, support, candidate, and proposal rules are frozen", "pre_final_outcome"),
        ("GATE05", "exact_treatment_likelihood", "R13_EXACT_TREATMENT_LIKELIHOOD_UNAVAILABLE", "exact branch has the required likelihood or theorem-equivalent orbit law", "branch_selection"),
        ("GATE06", "nuisance_theorem_regime_match", "R14_NUISANCE_THEOREM_REGIME_MISMATCH", "training dependence matches the invoked nuisance theorem", "nuisance_certificate"),
        ("GATE07", "proper_spatial_precision", "R06_SPATIAL_PRECISION_NOT_PD", "GMRF or spatial reference satisfies proper positive definiteness", "branch_selection"),
        ("GATE08", "finite_nuisance_confidence_set", "R07_EMPTY_NUISANCE_CERTIFICATE", "confidence set is nonempty, finite, and yields a finite deficit", "nuisance_certificate"),
        ("GATE09", "proposal_support", "R09_ORBIT_PROPOSAL_NO_SUPPORT", "importance/Monte Carlo proposal has theorem-required support and mutual absolute continuity", "computation"),
        ("GATE10", "vecchia_external_conditions", "R15_VECCHIA_CONDITIONS_UNVERIFIED", "all external N4/Vecchia conditions are verified before invoking the route", "branch_selection"),
        ("GATE11", "nonvacuous_final_bound", "R08_CERTIFIED_DEFICIT_VACUOUS", "final certified lower bound exceeds the frozen operational threshold", "post_certificate_pre_return"),
        ("GATE12", "eo_quality_interpretation", "R11_EO_QUALITY_INSUFFICIENT", "EO quality supports the claimed observed-product interpretation", "pre_return"),
        ("GATE13", "finite_numerical_output", "R10_NUMERICAL_NONFINITE", "weights, p-values, quantiles, certificates, and prediction sets are finite", "computation"),
        ("GATE14", "interference_scope", "R12_INTERFERENCE_SCOPE_UNSUPPORTED", "query stays within the frozen single-unit or registered sensitivity scope", "pre_query"),
        ("GATE15", "effective_calibration_size", "R03_LOW_EFFECTIVE_CALIBRATION_SIZE", "ESS exceeds the pilot-frozen operational threshold", "pre_return"),
        ("GATE16", "weight_concentration", "R04_WEIGHT_CONCENTRATION", "maximum normalized weight is below the pilot-frozen threshold", "pre_return"),
        ("GATE17", "graph_component_information", "R05_GRAPH_COMPONENT_TOO_SMALL", "component/block supplies the pilot-frozen minimum independent information", "pre_return"),
    ]
    return pd.DataFrame(rows, columns=["gate_id","gate_name","refusal_code","required_condition","evaluation_timing"])

def reporting_contract_registry() -> pd.DataFrame:
    rows = [
        ("REP01", "requested_dose", "all returned queries"),
        ("REP02", "intervention_bandwidth", "all returned queries"),
        ("REP03", "support_and_eligibility_status", "all returned queries and refusals"),
        ("REP04", "positivity_status", "all returned queries and refusals"),
        ("REP05", "branch_exact_or_sparse", "all returned intervals"),
        ("REP06", "spatial_block_and_boundary_rule", "all returned intervals"),
        ("REP07", "nuisance_certificate_status", "all returned intervals and refusals"),
        ("REP08", "separate_deficit_components", "approximation;misspecification;estimation;computation;certificate failure"),
        ("REP09", "raw_prediction_set_components", "M6 exact/scalable inversion"),
        ("REP10", "convex_hull_inflation", "only when a hull is shown for presentation"),
        ("REP11", "eo_interference_harddose_maup_inflation", "only when separately justified"),
        ("REP12", "effective_calibration_size", "weighted methods"),
        ("REP13", "maximum_normalized_weight", "weighted methods"),
        ("REP14", "final_coverage_lower_bound", "theorem-backed returned intervals"),
        ("REP15", "warnings_and_refusal_code", "all queries"),
        ("REP16", "numerical_diagnostics", "all methods"),
    ]
    return pd.DataFrame(rows, columns=["report_id","returned_item","scope"])


def source_alignment_matrix() -> pd.DataFrame:
    rows = [
        ("ALN01", "primary localized stochastic target; hard dose secondary", "stage3_registry.yaml;target_registry.csv;target_draw_contract.json", "aligned"),
        ("ALN02", "S1-S10 neutral, combined, failure, measurement and MAUP scenarios", "scenario_registry.csv", "aligned"),
        ("ALN03", "mixed endpoint and repeated mine-year temporal stress tests", "additional_stress_test_registry.csv;factor_registry.csv", "aligned"),
        ("ALN04", "spatial-confounding C2 negative control", "additional_stress_test_registry.csv;factor_registry.csv;assumption_registry.csv", "aligned"),
        ("ALN05", "M1-M6 comparison", "method_registry.csv", "aligned"),
        ("ALN06", "mandatory ablations including target-design removal and geographic heuristic", "ablation_registry.csv", "aligned"),
        ("ALN07", "coverage, absolute calibration, matched width, interval score, eligibility, refusal and false support", "metric_registry.csv", "aligned"),
        ("ALN08", "semivariogram and Moran diagnostics not coverage deficits", "metric_registry.csv", "aligned"),
        ("ALN09", "R1-R4 reductions and U1-U18 algebra/proof tests", "reduction_test_registry.csv;proof_test_registry.csv", "aligned"),
        ("ALN10", "all theorem refusal gates and reporting contract", "mandatory_gate_registry.csv;refusal_registry.json;reporting_contract_registry.csv", "aligned"),
        ("ALN11", "exact treatment-likelihood versus scalable target-design separation", "nuisance_model_registry.csv;mandatory_gate_registry.csv", "aligned"),
        ("ALN12", "causal identification separated from conformal validity", "assumption_registry.csv;stage3_registry.yaml", "aligned"),
        ("ALN13", "2025 post-assignment EO quality leakage guard", "feature_registry.csv;data_contract.json", "aligned"),
        ("ALN14", "outcome-blind fixed mine roles and LOMO evaluation", "mine_role_registry.csv;target_population_contract.json", "aligned"),
        ("ALN15", "deterministic seeds and dedicated target-draw stream", "seed_registry.csv", "aligned"),
        ("ALN16", "climate/terrain/soil/commodity scope explicitly deferred, not silently assumed", "deferred_feature_registry.csv", "aligned_minimum_route_with_declared_limitation"),
        ("ALN17", "Stage2A/2B inputs remain immutable and accepted", "input_manifest.json;STAGE3A_VERIFICATION.json", "aligned"),
    ]
    return pd.DataFrame(rows, columns=["alignment_id","source_requirement","registry_evidence","status"])


def feature_registry() -> pd.DataFrame:
    rows: list[tuple] = []
    columns = [
        "feature_name",
        "feature_group",
        "temporal_status",
        "feature_role",
        "allowed_in_synthetic_assignment",
        "allowed_in_fitted_treatment_model",
        "allowed_as_outcome_model_predictor",
        "allowed_in_support_stratification",
        "allowed_in_measurement_error_mechanism",
        "allowed_in_outcome_quality_diagnostics",
        "interpretation",
    ]

    for name in ["X_spatial_1", "X_spatial_2", "X_nonlinear_1", "X_nonlinear_2", "x_coord", "y_coord"]:
        rows.append((
            name, "controlled", "pretreatment", "synthetic_covariate",
            True, True, True, True, False, True,
            "synthetic generator covariate",
        ))

    for name, note in [
        ("pv_2023", "DEA pretreatment PV"),
        ("pv_2024", "DEA pretreatment PV"),
        ("pv_delta_2024_2023", "pretreatment trend"),
        ("ue_2023", "DEA pretreatment quality"),
        ("ue_2024", "DEA pretreatment quality"),
        ("valid_observation_count_2023", "DEA pretreatment availability"),
        ("valid_observation_count_2024", "DEA pretreatment availability"),
        ("valid_pixel_fraction_2023", "DEA pretreatment availability"),
        ("valid_pixel_fraction_2024", "DEA pretreatment availability"),
        ("water_fraction_2023", "DEA pretreatment quality"),
        ("water_fraction_2024", "DEA pretreatment quality"),
        ("unclear_fraction_2023", "DEA pretreatment quality"),
        ("unclear_fraction_2024", "DEA pretreatment quality"),
        ("graph_degree", "frozen graph"),
        ("graph_component_size", "frozen graph"),
        ("centroid_x", "frozen geometry"),
        ("centroid_y", "frozen geometry"),
        ("mine_identity", "frozen mine stratum"),
    ]:
        rows.append((
            name, "MineDoseBench", "pretreatment_or_static", "predictor",
            True, True, True, True, False, True, note,
        ))

    # The mapped fraction is a current regulatory snapshot, not authentic pretreatment dose.
    rows.append((
        "mapped_rehabilitation_fraction",
        "MineDoseBench",
        "frozen_snapshot_design_feature",
        "support_stratification_only",
        False, False, False, True, False, True,
        "noncausal snapshot exposure; never used as authentic annual treatment or pretreatment confounder",
    ))

    # Post-assignment 2025 fields are outcome/measurement-quality objects. They may
    # drive S9 product-error mechanisms and quality/refusal diagnostics, but never
    # treatment assignment, propensity fitting, or the outcome regression predictor.
    rows.append((
        "pv_2025", "MineDoseBench", "post_assignment_outcome", "observed_product_outcome",
        False, False, False, False, False, True,
        "observed DEA product outcome anchor; not a nuisance predictor",
    ))
    for name, note in [
        ("ue_2025", "post-assignment unmixing error"),
        ("valid_observation_count_2025", "post-assignment observation availability"),
        ("valid_pixel_fraction_2025", "post-assignment valid-pixel coverage"),
        ("water_fraction_2025", "post-assignment water quality"),
        ("unclear_fraction_2025", "post-assignment unclear-observation quality"),
        ("high_ue_fraction_20_2025", "post-assignment UE sensitivity at threshold 20"),
        ("high_ue_fraction_25_2025", "post-assignment UE operational screen at threshold 25"),
        ("high_ue_fraction_30_2025", "post-assignment UE sensitivity at threshold 30"),
    ]:
        rows.append((
            name, "MineDoseBench", "post_assignment_quality", "measurement_error_and_quality_driver",
            False, False, False, False, True, True, note,
        ))

    rows.append((
        "ue_proxy", "controlled", "post_assignment_quality", "measurement_error_and_quality_driver",
        False, False, False, False, True, True,
        "synthetic quality proxy reserved for S9 observed-versus-latent measurement-error mechanisms",
    ))
    return pd.DataFrame(rows, columns=columns)


def residual_law_registry() -> pd.DataFrame:
    rows = [
        (
            "RL1",
            "iid_continuous",
            "independent continuous residuals",
            "M1 reduction and benign exchangeable settings",
            "S1;S3;S7",
            "Stage3B",
            "registered_not_implemented",
        ),
        (
            "RL2",
            "proper_gaussian_gmrf",
            "positive proper GMRF with sparse precision",
            "scalable N2/N3 reference and GMRF specialization",
            "S2;S4;S5;S6;S8;S9;S10",
            "Stage3B/Stage3E",
            "preview_implemented",
        ),
        (
            "RL3",
            "monotone_transformed_gmrf",
            "componentwise invertible nonlinear transform of a proper GMRF, with the induced Jacobian retained",
            "non-Gaussian positive graph-factorized G2 theorem stress test",
            "S4;S6",
            "Stage3D",
            "registered_not_implemented",
        ),
        (
            "RL4",
            "misspecified_spatial_reference",
            "data generated under RL2 or RL3 but fitted with an incorrect graph or residual law",
            "structural misspecification and refusal analysis",
            "S6;S8",
            "Stage3E",
            "registered_not_implemented",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "residual_law_id",
            "residual_law_name",
            "definition",
            "scientific_role",
            "scenario_scope",
            "implementation_stage",
            "status",
        ],
    )


def target_draw_contract() -> dict:
    return {
        "primary_target_object": "random localized stochastic potential outcome Y_i(A_star)",
        "interior_draw_rule": "A_star is drawn from the normalized truncated-Gaussian intervention q_h on (0,1)",
        "endpoint_draw_rule": "A_star equals the audited synthetic endpoint atom exactly",
        "coverage_truth_required_in_stage3b": [
            "realized A_star",
            "realized Y_true_at_A_star",
            "realized Y_observed_at_A_star for S9",
            "target-draw seed and intervention-density value",
        ],
        "stage3a_truth_artifact": "analytic conditional mean functional only",
        "stage3a_truth_not_sufficient_for": [
            "finite-sample conformal coverage evaluation",
            "prediction-set membership evaluation",
            "observed-versus-latent S9 coverage",
        ],
        "hard_dose_truth_role": "secondary smoothness/sensitivity and generator diagnostic",
    }

def target_registry() -> pd.DataFrame:
    rows: list[dict] = []
    for alpha in [0.10, 0.05]:
        for dose in PRIMARY_DOSE_GRID:
            endpoint = bool(dose in (0.0, 1.0))
            rows.append(
                {
                    "target_id": f"A{int(round(alpha * 100)):02d}_D{int(round(dose * 100)):03d}",
                    "alpha": alpha,
                    "nominal_coverage": 1 - alpha,
                    "target_dose": dose,
                    "estimand_type": "localized_stochastic_potential_outcome",
                    "intervention_type": "audited_endpoint_atom" if endpoint else "truncated_gaussian_kernel_on_open_unit_interval",
                    "endpoint_atom": endpoint,
                    "endpoint_audit_scope": "synthetic treatment mechanism only; not NSW snapshot treatment" if endpoint else "not_applicable",
                    "bandwidth_primary": np.nan if endpoint else PRIMARY_BANDWIDTH,
                    "bandwidth_sensitivity": "not_applicable" if endpoint else "0.05;0.20",
                    "target_object": "random Y_i(A_star) under the registered localized intervention",
                    "stage3a_preview_artifact": "analytic conditional mean functional; realized target draws deferred to Stage3B",
                    "hard_dose_truth_role": "secondary sensitivity and generator diagnostic",
                }
            )
    return pd.DataFrame(rows)


def assumption_registry() -> pd.DataFrame:
    rows = [
        ("C1", "Consistency", "Observed synthetic outcome equals the potential outcome under assigned synthetic treatment", "satisfied_by_construction_in_controlled_and_MineDoseBench", "not established for the real NSW snapshot"),
        ("C2", "Supported conditional ignorability", "Synthetic assignment is generated from registered pretreatment variables", "satisfied_by_construction_on_supported targets", "not claimed for the real NSW snapshot"),
        ("C3", "Positivity", "Target intervention is restricted to audited support", "checked by support/refusal system", "failure triggers refusal"),
        ("C4", "Primary no-material-interference condition", "Primary generator uses single-unit potential outcomes", "satisfied_by_construction in primary simulations", "interference is a separate sensitivity layer"),
        ("C5", "Outcome-law stability", "Changing the synthetic assignment law does not alter the registered response law conditional on the causal variables", "satisfied_by_construction", "not repaired by conformal calibration"),
        ("F1", "Localized causal identification", "C1-C5 identify the localized stochastic potential-outcome target", "theorem layer; not a conformal-validity claim", "causal and conformal claims remain separate"),
    ]
    return pd.DataFrame(rows, columns=["assumption_id", "assumption_name", "operational_definition", "benchmark_status", "interpretation_boundary"])


def proof_test_registry() -> pd.DataFrame:
    rows = [
        ("U1", "target tilt normalization", "Stage3D", "registered_not_implemented"),
        ("U2", "Jacobian", "Stage3D", "registered_not_implemented"),
        ("U3", "target cancellation", "Stage3D", "registered_not_implemented"),
        ("U4", "duplicate orbit quotient", "Stage3D", "registered_not_implemented"),
        ("U5", "GMRF exponent", "Stage3D", "registered_not_implemented"),
        ("U6", "N1 correlation", "Stage3E", "registered_not_implemented"),
        ("U7", "sharp frontier", "Stage3E", "registered_not_implemented"),
        ("U8", "Gaussian KL sign", "Stage3E", "registered_not_implemented"),
        ("U9", "covariance perturbation positivity", "Stage3E", "registered_not_implemented"),
        ("U10", "boundary second moment", "Stage3E", "registered_not_implemented"),
        ("U11", "normalized-weight invariance", "Stage3C", "registered_not_implemented"),
        ("U12", "sharp total-variation identity", "Stage3E", "registered_not_implemented"),
        ("U13", "N2 exact case", "Stage3E", "registered_not_implemented"),
        ("U14", "good-event arithmetic", "Stage3E", "registered_not_implemented"),
        ("U15", "target-design ratio normalization", "Stage3E", "registered_not_implemented"),
        ("U16", "treatment likelihood normalization including endpoint atoms", "Stage3A", "implemented_preview_numeric_check"),
        ("U17", "score-inversion finite-sample coverage", "Stage3D", "registered_not_implemented"),
        ("U18", "route consistency and tightest valid bound", "Stage3E", "registered_not_implemented"),
    ]
    return pd.DataFrame(rows, columns=["test_id", "test_name", "implementation_stage", "status"])


def seed_registry() -> pd.DataFrame:
    production_counts = {"S1": 300, "S2": 300, "S3": 300, "S4": 500, "S5": 300, "S6": 300, "S7": 300, "S8": 300, "S9": 200, "S10": 200}
    specifications: list[tuple[str, str, int]] = []
    for phase, counts in [("pilot", {key: 20 for key in production_counts}), ("production_provisional", production_counts)]:
        for scenario_id, count in counts.items():
            for replication in range(1, count + 1):
                specifications.append((phase, scenario_id, replication))

    seed_sequence = np.random.SeedSequence(ROOT_SEED)
    children = seed_sequence.spawn(len(specifications))
    rows: list[dict] = []
    for (phase, scenario_id, replication), child in zip(specifications, children):
        values = child.generate_state(6, dtype=np.uint32)
        rows.append(
            {
                "phase": phase,
                "scenario_id": scenario_id,
                "replication": replication,
                "data_seed": int(values[0]),
                "split_seed": int(values[1]),
                "nuisance_seed": int(values[2]),
                "orbit_seed": int(values[3]),
                "measurement_seed": int(values[4]),
                "target_draw_seed": int(values[5]),
                "production_status": "provisional_until_stage3f" if phase.startswith("production") else "frozen_pilot",
            }
        )
    return pd.DataFrame(rows)


def mine_roles(splits: pd.DataFrame) -> pd.DataFrame:
    mines = (
        splits[["fold_id", "held_out_MineID", "held_out_MineN"]]
        .drop_duplicates()
        .sort_values("fold_id")
        .reset_index(drop=True)
    )
    ids = mines["held_out_MineID"].tolist()
    names = dict(zip(mines["held_out_MineID"], mines["held_out_MineN"]))
    rows: list[dict] = []
    n = len(ids)
    for i, test in enumerate(ids):
        roles = {
            test: "test_target",
            ids[(i + 1) % n]: "support_audit",
            ids[(i + 2) % n]: "calibration",
            ids[(i + 3) % n]: "nuisance_training",
            ids[(i + 4) % n]: "nuisance_training",
        }
        for mine_id in ids:
            rows.append({"fold_id": mines.loc[i, "fold_id"], "MineID": mine_id, "MineN": names[mine_id], "role": roles[mine_id]})
    return pd.DataFrame(rows)


def make_registry() -> dict:
    return {
        "registry_version": "1.3.1-stage3a-freeze",
        "created_utc": utc_now(),
        "scientific_scope": {
            "primary_evidence": "known-ground-truth controlled simulation and MineDoseBench",
            "primary_estimand": "localized stochastic potential outcome under a supported mixed intervention",
            "hard_dose_truth": "secondary sensitivity and generator diagnostic",
            "stage3a_localized_truth_scope": "analytic mean functional only; realized stochastic target draws are required in Stage3B",
            "real_nsw_scope": "product-aware EO uncertainty/refusal demonstration only",
            "forbidden_interpretation": "mapped rehabilitation implementation fraction is not annual causal treatment",
        },
        "status": {
            "architecture_frozen": True,
            "pilot_registry_frozen": True,
            "production_numerics": "provisional_until_stage3f",
            "methods_implemented": False,
        },
        "primary_settings": {
            "alpha": 0.10,
            "alpha_sensitivity": 0.05,
            "dose_grid": PRIMARY_DOSE_GRID.tolist(),
            "interior_intervention_kernel": "truncated Gaussian on (0,1)",
            "interior_bandwidth": PRIMARY_BANDWIDTH,
            "bandwidth_sensitivity": [0.05, 0.20],
            "exact_orbit_block_primary": 6,
            "exact_orbit_block_stress": 8,
            "pilot_repetitions_per_scenario": 20,
            "random_forest_all_scenarios": True,
            "xgboost_confirmation": ["S1", "S4", "S5", "S8"],
        },
        "hard_rules": [
            "preserve exact endpoint atoms",
            "use log-scale weights",
            "never form dense precision inverse",
            "do not realization-standardize theorem GMRF residual draws",
            "freeze graph/block IDs/folds",
            "separate causal identification from conformal validity",
            "separate observed-product and latent-outcome coverage",
            "preserve raw disconnected prediction-set components",
            "refuse unsupported targets",
            "do not use 2025 EO quality fields in treatment or nuisance prediction models",
            "generate realized localized target draws before any coverage evaluation",
            "use a dedicated target_draw_seed stream distinct from assignment, orbit, and measurement seeds",
            "include repeated mine-year temporal dependence and the C2-violation negative control",
            "freeze all mandatory ablations, metrics, theorem gates, and reporting fields before implementation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="inputs")
    parser.add_argument("--output", default="outputs_stage3a")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = (PACKAGE_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input).resolve()
    output = (PACKAGE_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    require(output != PACKAGE_ROOT, "Refusing to overwrite the package root")
    require(output != input_root and output not in input_root.parents and input_root not in output.parents, "Input and output paths must be disjoint")
    if output.exists():
        require(args.overwrite, "Output exists; pass --overwrite to recreate Stage 3A only")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    before = {str(path.relative_to(input_root)): sha256_file(path) for path in sorted(input_root.rglob("*")) if path.is_file()}
    blocks, edges, splits, _ = validate_stage2a(input_root)
    by, summary, selected, _, _ = validate_stage2b(input_root, blocks)

    scenarios = scenario_registry()
    methods = method_registry()
    factors = factor_registry()
    nuisance_models = nuisance_model_registry()
    ablations = ablation_registry()
    metrics = metric_registry()
    reductions = reduction_test_registry()
    stress_tests = additional_stress_test_registry()
    mandatory_gates = mandatory_gate_registry()
    reporting_contract = reporting_contract_registry()
    alignment = source_alignment_matrix()
    deferred_features = deferred_feature_registry()
    residual_laws = residual_law_registry()
    refusals = refusal_registry()
    features = feature_registry()
    targets = target_registry()
    assumptions = assumption_registry()
    proof_tests = proof_test_registry()
    seeds = seed_registry()
    roles = mine_roles(splits)
    preview, hard_truth, localized_truth, sim_edges, diagnostics = generate_preview()
    registry = make_registry()

    by_bool = by.assign(block_year_eligible=as_bool(by["block_year_eligible"]))
    all_year_eligible = int(by_bool.groupby("block_id")["block_year_eligible"].all().sum())
    registry["frozen_input_counts"] = {
        "mines": int(blocks.MineID.nunique()),
        "blocks": len(blocks),
        "edges": len(edges),
        "lomo_folds": int(splits.fold_id.nunique()),
        "block_year_rows": len(by),
        "all_year_eligible_blocks": all_year_eligible,
    }

    (output / "stage3_registry.yaml").write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    scenarios.to_csv(output / "scenario_registry.csv", index=False, encoding="utf-8-sig")
    methods.to_csv(output / "method_registry.csv", index=False, encoding="utf-8-sig")
    factors.to_csv(output / "factor_registry.csv", index=False, encoding="utf-8-sig")
    nuisance_models.to_csv(output / "nuisance_model_registry.csv", index=False, encoding="utf-8-sig")
    ablations.to_csv(output / "ablation_registry.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "metric_registry.csv", index=False, encoding="utf-8-sig")
    reductions.to_csv(output / "reduction_test_registry.csv", index=False, encoding="utf-8-sig")
    stress_tests.to_csv(output / "additional_stress_test_registry.csv", index=False, encoding="utf-8-sig")
    mandatory_gates.to_csv(output / "mandatory_gate_registry.csv", index=False, encoding="utf-8-sig")
    reporting_contract.to_csv(output / "reporting_contract_registry.csv", index=False, encoding="utf-8-sig")
    alignment.to_csv(output / "source_alignment_matrix.csv", index=False, encoding="utf-8-sig")
    deferred_features.to_csv(output / "deferred_feature_registry.csv", index=False, encoding="utf-8-sig")
    residual_laws.to_csv(output / "residual_law_registry.csv", index=False, encoding="utf-8-sig")
    seeds.to_csv(output / "seed_registry.csv", index=False, encoding="utf-8-sig")
    roles.to_csv(output / "mine_role_registry.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output / "feature_registry.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(output / "target_registry.csv", index=False, encoding="utf-8-sig")
    assumptions.to_csv(output / "assumption_registry.csv", index=False, encoding="utf-8-sig")
    proof_tests.to_csv(output / "proof_test_registry.csv", index=False, encoding="utf-8-sig")
    write_json(output / "refusal_registry.json", refusals)
    write_json(output / "target_draw_contract.json", target_draw_contract())
    write_json(output / "target_population_contract.json", target_population_contract())
    write_json(output / "stage3a_environment_inventory.json", environment_inventory())
    preview.to_csv(output / "stage3a_generator_preview.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0}, encoding="utf-8")
    hard_truth.to_csv(output / "stage3a_generator_truth.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0}, encoding="utf-8")
    localized_truth.to_csv(output / "stage3a_localized_target_truth.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0}, encoding="utf-8")
    sim_edges.to_csv(output / "stage3a_preview_graph_edges.csv", index=False, encoding="utf-8-sig")
    write_json(output / "stage3a_generator_diagnostics.json", diagnostics)

    contract = {
        "stage2a_block_key": "block_id",
        "stage2b_block_year_key": ["block_id", "year"],
        "mine_key": "MineID",
        "frozen_stage2a_fields": blocks.dtypes.astype(str).to_dict(),
        "stage2b_fields": by.dtypes.astype(str).to_dict(),
        "preview_fields": preview.dtypes.astype(str).to_dict(),
        "hard_truth_fields": hard_truth.dtypes.astype(str).to_dict(),
        "localized_truth_fields": localized_truth.dtypes.astype(str).to_dict(),
        "forbidden_mutations": ["block_id", "MineID", "graph_component_id", "fold_id", "graph edges", "mapped rehabilitation fraction"],
        "temporal_rule": "2023-2024 may be pretreatment for MineDoseBench; 2025 is an outcome/quality anchor and is excluded from treatment assignment and nuisance predictor sets",
        "snapshot_rule": "mapped_rehabilitation_fraction is support-stratification metadata only, not authentic pretreatment treatment or confounder",
        "target_rule": "the primary object is random Y_i(A_star); the Stage 3A localized table stores only its analytic conditional mean, while realized target draws are mandatory in Stage 3B",
        "post_assignment_quality_rule": "2025 UE/availability/water/unclear fields may drive S9 measurement-error mechanisms and EO-quality diagnostics only; they are forbidden in assignment, propensity, and outcome-regression predictors",
        "target_population_rule": target_population_contract(),
        "seed_stream_rule": "each replication has distinct data, split, nuisance, orbit, measurement, and target-draw seed streams",
        "deferred_context_rule": "climate, terrain, soil, and commodity context are absent from the minimum route and cannot be silently used unless acquired, temporally audited, and frozen before production",
    }
    write_json(output / "data_contract.json", contract)

    input_manifest = {
        "created_utc": utc_now(),
        "files": {key: {"sha256": value, "size_bytes": (input_root / key).stat().st_size} for key, value in before.items()},
        "stage2a_manifest_sha256": sha256_file(input_root / "stage2a/stage2_manifest.json"),
        "stage2b_archive_sha256": sha256_file(input_root / "stage2b/GeoDose_Stage2B_OUTPUTS_FINAL.zip"),
    }
    write_json(output / "input_manifest.json", input_manifest)

    endpoint = preview["treatment_category"].value_counts().to_dict()
    report = f"""# GeoDose-CP Stage 3A Prescreen Report

## Decision

Stage 3A registry and generator prescreen completed. This package does **not** implement or run M1–M6 production inference.

## Frozen input audit

- Mines: {blocks.MineID.nunique()}
- Blocks: {len(blocks):,}
- Queen edges: {len(edges):,}
- LOMO folds: {splits.fold_id.nunique()}
- DEA block-year records: {len(by):,}
- All-years eligible blocks: {all_year_eligible:,}
- Selected applied-demonstration mines: {", ".join(selected.sort_values("selection_order")["MineN"].tolist())}

## Controlled generator preview

- Units: {len(preview)} on a 25 × 25 queen graph
- Graph edges: {len(sim_edges):,}
- Genuine synthetic treatment atoms: A=0 ({endpoint.get("atom_0", 0)}), A=1 ({endpoint.get("atom_1", 0)})
- Interior treatments: {endpoint.get("interior", 0)}
- Hard-dose truth rows: {len(hard_truth):,} (secondary truth)
- Localized stochastic target mean rows: {len(localized_truth):,} (analytic mean functional only; realized target draws are deferred to Stage 3B)
- Precision minimum eigenvalue: {min(diagnostics["precision_min_eigenvalues"].values()):.6f}
- Realization-dependent GMRF normalization: no
- Mixed-treatment normalization maximum absolute error: {diagnostics["mixed_treatment_normalization_max_abs_error"]:.3e}
- Dense precision inverse formed: no

## Registry status

- S1–S10 architecture: frozen
- M1–M6 definitions: frozen but not implemented
- Controlled-factor, nuisance-model, complete mandatory ablation, metric, and R1–R4 reduction registries: frozen at architecture level
- Residual-law registry: IID, proper GMRF, non-Gaussian transformed-GMRF, and misspecified-reference routes registered
- Additional stress tests: endpoint atoms, repeated mine-year temporal dependence, and a C2-violation negative control registered
- Mandatory theorem gates and returned-result reporting contract: frozen
- C1–C5/F1 assumptions: explicitly registered
- U1–U18 proof/algebra tests: explicitly registered; U16 preview check implemented
- Pilot seed registry: frozen with a dedicated target-draw seed stream
- Production repetition counts and numerical thresholds: provisional until Stage 3F pilot freeze
- Exact-orbit primary/stress sizes: 6 and 8
- Climate/terrain/soil/commodity covariates: explicitly deferred in the minimum route; no silent use permitted

## Hard interpretation boundary

The mapped rehabilitation implementation fraction remains a noncausal snapshot exposure and is restricted to support stratification. Known-ground-truth causal coverage will be established only through controlled simulation and MineDoseBench. Interior primary targets are random localized stochastic potential outcomes. The Stage 3A table stores only an analytic mean functional; Stage 3B must generate realized A_star and Y(A_star) draws before any coverage claim. Hard-dose curves are secondary sensitivity truth.
"""
    (output / "stage3_prescreen_report.md").write_text(report, encoding="utf-8")

    after = {str(path.relative_to(input_root)): sha256_file(path) for path in sorted(input_root.rglob("*")) if path.is_file()}
    require(before == after, "Frozen input files changed during Stage 3A")

    outputs = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            outputs[path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "script_version": SCRIPT_VERSION,
        "run_utc": utc_now(),
        "input_root": str(input_root),
        "output_root": str(output),
        "frozen_inputs_modified": False,
        "registry_state": "stage3a_pre_pilot",
        "production_experiments_run": False,
        "controlled_preview_units": len(preview),
        "primary_estimand": "localized stochastic potential outcome",
        "outputs": outputs,
    }
    write_json(output / "stage3a_manifest.json", manifest)

    print("STAGE 3A PREPARATION COMPLETE")
    print(f"Frozen blocks: {len(blocks):,}")
    print(f"Frozen block-year rows: {len(by):,}")
    print(f"Registered scenarios: {len(scenarios)}")
    print(f"Registered methods: {len(methods)}")
    print(f"Seed rows: {len(seeds):,}")
    print(f"Generator preview units: {len(preview)}")
    print("Primary target: localized stochastic potential outcome")


if __name__ == "__main__":
    try:
        main()
    except Stage3AError as exc:
        print(f"STAGE 3A FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
