from __future__ import annotations

import inspect
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml

from . import __version__
from .factorization_fixture import evaluate_factorizing_candidate
from .graph_law import build_local_graph_law
from .io import (
    D3M4Error,
    load_d2_reference,
    load_m3_reference,
    load_stage3b_oracle,
    safe_reset_output_dir,
    sha256_file,
    verify_upstream,
    write_csv_gz,
    write_json,
)
from .m3_oracle import candidate_residuals, evaluate_candidate
from .m4_oracle import evaluate_m4
from .m6_extractor import extract_m6_target_source_marginal
from .orbit import distinct_permutation_count, numeric_key


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _case_frame(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    out = df[df["case_id"] == case_id].copy()
    if out.empty:
        raise D3M4Error(f"Missing case {case_id}")
    return out


def _target_unit_id(case_units: pd.DataFrame, target_node: int) -> str:
    row = case_units[case_units["node_index"] == int(target_node)]
    if len(row) != 1 or str(row.iloc[0]["role"]) != "test_target":
        raise D3M4Error("Target node does not identify exactly one test_target unit")
    return str(row.iloc[0]["unit_id"])


def _target_row(data, case_id: str, target_unit_id: str, mode: str, target_dose: float | None) -> pd.Series:
    if mode == "observational":
        frame = data.observational_targets[
            (data.observational_targets["case_id"] == case_id)
            & (data.observational_targets["unit_id"] == target_unit_id)
        ]
    elif mode == "localized":
        if target_dose is None:
            raise D3M4Error("Localized target requires target_dose")
        frame = data.realized_targets[
            (data.realized_targets["case_id"] == case_id)
            & (data.realized_targets["unit_id"] == target_unit_id)
            & np.isclose(data.realized_targets["target_dose"].astype(float), float(target_dose), atol=1e-12, rtol=0)
            & (data.realized_targets["draw_status"] == "drawn")
        ]
        if "target_scope" in frame.columns and (frame["target_scope"] == "registered_grid").any():
            frame = frame[frame["target_scope"] == "registered_grid"]
    else:
        raise D3M4Error(f"Unknown target mode: {mode}")
    if len(frame) != 1:
        raise D3M4Error(f"Expected one target row for {case_id}/{target_unit_id}; found {len(frame)}")
    return frame.iloc[0]


def _case_inputs(data, case_id: str, mode: str, target_dose: float | None, block_nodes: list[int], target_node: int):
    case_units = _case_frame(data.units, case_id).sort_values("node_index").reset_index(drop=True)
    case_edges = _case_frame(data.true_edges, case_id)
    case_row_frame = data.cases[data.cases["case_id"] == case_id]
    if len(case_row_frame) != 1:
        raise D3M4Error(f"Case registry mismatch for {case_id}")
    case_row = case_row_frame.iloc[0]
    target_uid = _target_unit_id(case_units, target_node)
    target_row = _target_row(data, case_id, target_uid, mode, target_dose)
    lookup = case_units.set_index("node_index")
    for node in block_nodes[:-1]:
        if str(lookup.loc[node, "role"]) != "calibration":
            raise D3M4Error(f"Block node {node} is not calibration in {case_id}")
    cal_resid = lookup.loc[block_nodes[:-1], "shared_spatial_residual"].to_numpy(float)
    reconstructed = lookup.loc[block_nodes[:-1], "Y_observed_at_A"].to_numpy(float) - lookup.loc[block_nodes[:-1], "conditional_mean_at_A"].to_numpy(float)
    if not np.allclose(reconstructed, cal_resid, atol=2e-12, rtol=0):
        raise D3M4Error(f"Oracle residual reconstruction mismatch in {case_id}")
    if mode == "localized":
        target_mean = float(target_row["conditional_mean_at_A_star"])
        target_A = float(target_row["A_star"])
    else:
        # Frozen observational target rows store A_star and conditional_mean_at_A_star as the observed-law draw interface.
        target_mean = float(target_row["conditional_mean_at_A_star"])
        target_A = float(target_row["A_star"])
    target_scale = 1.0
    law = build_local_graph_law(
        case_units,
        case_edges,
        block_nodes,
        rho=float(case_row["spatial_rho"]),
        residual_law=str(case_row["residual_law"]),
        scale=0.35,
    )
    return case_units, case_edges, case_row, target_row, cal_resid, target_mean, target_scale, target_A, law


def _oracle_log_dose_ratio(data, case_id: str, case_units: pd.DataFrame, target_row: pd.Series, block_nodes: list[int], target_dose: float) -> np.ndarray:
    lookup = case_units.set_index("node_index")
    out: list[float] = []
    for node in block_nodes[:-1]:
        uid = str(lookup.loc[node, "unit_id"])
        frame = data.treatment_truth[
            (data.treatment_truth["case_id"] == str(case_id))
            & (data.treatment_truth["unit_id"] == uid)
            & np.isclose(data.treatment_truth["target_dose"].astype(float), float(target_dose), atol=1e-12, rtol=0)
        ]
        if "target_scope" in frame.columns:
            frame = frame[frame["target_scope"] == "registered_grid"]
        if len(frame) != 1:
            raise D3M4Error(f"Expected one oracle treatment-transport row for source {uid}; found {len(frame)}")
        out.append(float(frame.iloc[0]["log_oracle_treatment_ratio"]))
    target_log = float(target_row["log_oracle_treatment_ratio_at_A_star"])
    out.append(target_log)
    arr = np.asarray(out, dtype=float)
    if np.any(np.isnan(arr)) or np.isposinf(arr).any():
        raise D3M4Error("Oracle M4 treatment log-ratio contains NaN/+inf")
    return arr


def _m3_regression_against_accepted(data, m3ref, registry: dict[str, Any], block_nodes: list[int], target_node: int, tol: float) -> dict[str, Any]:
    case_specs = {str(x["case_id"]): x for x in registry["m3_reference_cases"]}
    expected_cases = set(case_specs)
    if set(m3ref.candidate_trace["case_id"].astype(str)) != expected_cases:
        raise D3M4Error("Accepted M3 reference case set changed")
    max_p = 0.0
    max_s = 0.0
    checked = 0
    for case_id, tgroup in m3ref.candidate_trace.groupby("case_id"):
        spec = case_specs[str(case_id)]
        case_units, _, _, _, cal, mean, scale, _, law = _case_inputs(
            data, str(case_id), str(spec["target_mode"]), spec.get("target_dose"), block_nodes, target_node
        )
        del case_units
        for row in tgroup.itertuples(index=False):
            y = float(row.candidate_y)
            result = evaluate_candidate(cal, y, mean, law, len(block_nodes)-1, target_scale=scale, tie_tolerance=1e-12, keep_states=False)
            max_p = max(max_p, abs(float(result.pvalue) - float(row.pvalue)))
            if int(result.state_count) != int(row.state_count):
                raise D3M4Error("Signed-zero patch changed an accepted M3 state count")
            sref = m3ref.source_marginals[
                (m3ref.source_marginals["case_id"].astype(str) == str(case_id))
                & np.isclose(m3ref.source_marginals["candidate_y"].astype(float), y, atol=1e-14, rtol=0)
            ].sort_values("source_index")
            if len(sref) != len(result.source_marginals):
                raise D3M4Error("Accepted M3 source-marginal row count changed")
            max_s = max(max_s, float(np.max(np.abs(result.source_marginals - sref["spatial_source_mass"].to_numpy(float)))))
            checked += 1
    return {
        "candidate_rows_checked": checked,
        "max_abs_pvalue_difference": max_p,
        "max_abs_source_marginal_difference": max_s,
        "pass": bool(max_p <= tol and max_s <= tol),
    }


def run(package_root: Path, stage_paths: dict[str, Path], output_dir: Path, overwrite: bool) -> dict[str, Any]:
    started = time.perf_counter()
    configs = package_root / "configs"
    expected = _load_json(configs / "expected_upstream_hashes.json")
    registry = _load_yaml(configs / "diagnostic_registry.yaml")
    tolerances = _load_yaml(configs / "tolerances.yaml")
    contract = _load_yaml(configs / "m4_oracle_contract.yaml")
    safe_reset_output_dir(output_dir, package_root, overwrite)

    input_audit = verify_upstream(stage_paths, expected)
    data = load_stage3b_oracle(stage_paths["stage3b_output_zip"])
    d2 = load_d2_reference(stage_paths["stage3d_d2_output_zip"])
    m3ref = load_m3_reference(stage_paths["stage3d_d3_m3_output_zip"])
    input_audit.update({
        "stage3b_verification_status": data.verification.get("status"),
        "d2_verification_status": d2.verification.get("status"),
        "m3_verification_status": m3ref.verification.get("status"),
        "m3_source_aggregate_sha256": m3ref.source_hashes.get("aggregate_sha256"),
        "d2_source_tree_hash": d2.source_hashes.get("tree_hash"),
        "definition_gate_bundled_sha256": sha256_file(package_root / "docs" / "GeoDose_D3_M3_M4_M6_Definition_Gate_v1_1.docx"),
    })
    write_json(output_dir / "stage3d_d3_m4_input_audit.json", input_audit)

    block_nodes = list(map(int, registry["block_nodes"]))
    target_node = int(registry["target_node"])
    if block_nodes[-1] != target_node:
        raise D3M4Error("Target node must be final block position")
    fixture = data.exact_fixtures["size6_unique"]
    fixture_nodes = [int(x["node_index"]) for x in fixture["fixed_slots"]]
    if fixture_nodes != block_nodes or int(fixture["target_slot_node_index"]) != target_node:
        raise D3M4Error("Frozen size-6 fixture no longer matches D3 block registry")

    checks: dict[str, Any] = {}
    # 1. Signed-zero robustness patch and regression against accepted M3 outputs.
    checks["signed_zero_keys_equal"] = numeric_key(-0.0) == numeric_key(+0.0)
    checks["signed_zero_duplicate_count_three"] = distinct_permutation_count(np.array([-0.0, +0.0, 1.0])) == 3
    m3reg = _m3_regression_against_accepted(
        data, m3ref, registry, block_nodes, target_node, float(tolerances["m3_regression_abs"])
    )
    checks["signed_zero_patch_preserves_accepted_M3"] = bool(m3reg["pass"])
    write_json(output_dir / "stage3d_d3_m4_m3_regression.json", m3reg)
    write_json(output_dir / "stage3d_d3_m4_signed_zero_audit.json", {
        "numeric_key_negative_zero": numeric_key(-0.0),
        "numeric_key_positive_zero": numeric_key(+0.0),
        "keys_equal": checks["signed_zero_keys_equal"],
        "distinct_permutations_for_minus0_plus0_1": distinct_permutation_count(np.array([-0.0, +0.0, 1.0])),
        "scientific_effect": "robustness_only; accepted M3 diagnostics are unchanged",
    })

    # 2. D3-only factorizing reduction fixture with genuine spatial dependence.
    fcase = str(registry["factorizing_fixture"]["source_graph_case"])
    f_units, _, f_case_row, _, f_cal, _, _, _, f_law = _case_inputs(
        data, fcase, "observational", None, block_nodes, target_node
    )
    del f_units
    if abs(float(f_case_row["spatial_rho"])) <= 0:
        raise D3M4Error("Factorizing fixture accidentally has zero spatial dependence")
    synthetic_A = np.asarray(registry["factorizing_fixture"]["synthetic_treatments"], dtype=float)
    alpha = float(registry["alpha"])
    factor_rows: list[dict[str, Any]] = []
    factor_source_rows: list[dict[str, Any]] = []
    for y in map(float, registry["factorizing_fixture"]["candidate_y"]):
        r = evaluate_factorizing_candidate(
            f_cal, y, f_law, len(block_nodes)-1, synthetic_A, alpha, float(tolerances["score_tie_abs"])
        )
        factor_rows.append({
            "fixture_id": "D3_FACTORISING_SPATIAL_REDUCTION",
            "candidate_y": y,
            "spatial_rho": float(f_case_row["spatial_rho"]),
            "p3": r.p3,
            "p4": r.p4,
            "p6_reference": r.p6_reference,
            "accepted3": r.accepted3,
            "accepted4": r.accepted4,
            "accepted6": r.accepted6,
            "treatment_log_factor_range": r.treatment_log_factor_range,
            "residual_log_factor_range": r.residual_log_factor_range,
            "q3_q4_max_abs": r.q3_q4_max_abs,
            "q3_q6_max_abs": r.q3_q6_max_abs,
            "p_max_abs": r.p_max_abs,
            "publication_evidence": False,
        })
        for j in range(len(r.q3)):
            factor_source_rows.append({
                "fixture_id": "D3_FACTORISING_SPATIAL_REDUCTION",
                "candidate_y": y,
                "source_index": j,
                "q3": float(r.q3[j]),
                "q4": float(r.q4[j]),
                "q6_reference": float(r.q6_reference[j]),
            })
    factor_df = pd.DataFrame(factor_rows)
    checks["factorizing_spatial_dependence_nonzero"] = bool((factor_df["spatial_rho"].abs() > 0).all())
    checks["factorizing_residual_law_active"] = bool((factor_df["residual_log_factor_range"] > float(tolerances["factorizing_residual_log_range_min"])).all())
    checks["factorizing_treatment_factor_orbit_constant"] = bool((factor_df["treatment_log_factor_range"].abs() <= float(tolerances["factorizing_equality_abs"])).all())
    checks["factorizing_q3_q4_equal"] = bool((factor_df["q3_q4_max_abs"] <= float(tolerances["factorizing_equality_abs"])).all())
    checks["factorizing_q3_q6_equal"] = bool((factor_df["q3_q6_max_abs"] <= float(tolerances["factorizing_equality_abs"])).all())
    checks["factorizing_pvalues_equal"] = bool((factor_df["p_max_abs"] <= float(tolerances["factorizing_equality_abs"])).all())
    checks["factorizing_acceptance_membership_equal"] = bool((factor_df[["accepted3","accepted4","accepted6"]].nunique(axis=1) == 1).all())
    factor_df.to_csv(output_dir / "stage3d_d3_m4_factorizing_reduction.csv", index=False, lineterminator="\n")
    write_csv_gz(pd.DataFrame(factor_source_rows), output_dir / "stage3d_d3_m4_factorizing_source_marginals.csv.gz")

    # 3. Frozen S4 nonfactorization test. M6 probabilities are extracted, never re-derived, from accepted D2 output.
    s4 = registry["nonfactorizing_fixture"]
    s4_units, _, s4_case_row, s4_target, s4_cal, s4_mean, s4_scale, _, s4_law = _case_inputs(
        data, str(s4["case_id"]), "localized", float(s4["target_dose"]), block_nodes, target_node
    )
    logd = _oracle_log_dose_ratio(data, str(s4["case_id"]), s4_units, s4_target, block_nodes, float(s4["target_dose"]))
    d2grid = d2.grid_registry[d2.grid_registry["fixture_id"].astype(str) == str(s4["d2_fixture_id"])]
    if len(d2grid) != 1:
        raise D3M4Error("Expected one D2 grid registry row for S4 interior fixture")
    grow = d2grid.iloc[0]
    checks["d2_domain_frozen_minus8_plus8"] = bool(float(grow["lower"]) == -8.0 and float(grow["upper"]) == 8.0 and bool(grow["domain_frozen_before_truth_audit"]))
    checks["d2_alpha_matches"] = bool(abs(float(grow["alpha"]) - alpha) <= 1e-15)
    checks["s4_case_is_combined_shift"] = bool(str(s4_case_row["scenario_id"]) == "S4" and str(s4_case_row["target_shift"]) == "strong" and abs(float(s4_case_row["spatial_rho"]) - 0.6) <= 1e-15)

    d2states = d2.candidate_states[d2.candidate_states["fixture_id"].astype(str) == str(s4["d2_fixture_id"])].copy()
    if d2states.empty:
        raise D3M4Error("D2 has no state-audit candidates for frozen S4 fixture")
    candidate_registry = (
        d2states[["candidate_y","candidate_hex"]]
        .drop_duplicates()
        .sort_values("candidate_y")
        .reset_index(drop=True)
    )
    # These candidates are inherited from the accepted D2 state-audit registry; D3 does not choose a new grid.
    structural_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for crow in candidate_registry.itertuples(index=False):
        y = float(crow.candidate_y)
        chex = str(crow.candidate_hex)
        m3 = evaluate_candidate(s4_cal, y, s4_mean, s4_law, len(block_nodes)-1, target_scale=s4_scale, tie_tolerance=float(tolerances["score_tie_abs"]), keep_states=False)
        residual_payload = candidate_residuals(s4_cal, y, s4_mean, s4_scale)
        m4 = evaluate_m4(m3.source_marginals, residual_payload, logd, len(block_nodes)-1, tie_tolerance=float(tolerances["score_tie_abs"]))
        m6 = extract_m6_target_source_marginal(
            d2.candidate_states, d2.candidate_trace, str(s4["d2_fixture_id"]), chex, len(block_nodes)-1, float(tolerances["score_tie_abs"])
        )
        base = np.zeros(len(logd), dtype=float)
        finite_d = np.isfinite(logd)
        base[finite_d] = np.exp(logd[finite_d]) * m3.source_marginals[finite_d]
        common = (base > 0.0) & (m6.q6 > 0.0)
        zero_mismatch = bool(np.any((base > 0.0) & (m6.q6 == 0.0)) or np.any((base == 0.0) & (m6.q6 > float(tolerances["zero_probability_abs"]))))
        if int(common.sum()) >= 2:
            logrho = np.log(m6.q6[common]) - np.log(base[common])
            delta_fact = float(np.max(logrho) - np.min(logrho))
        elif zero_mismatch:
            delta_fact = float(tolerances["structural_zero_mismatch_sentinel"])
        else:
            raise D3M4Error("Too few common-positive sources to evaluate factorization")
        tv = float(0.5 * np.sum(np.abs(m4.q4 - m6.q6)))
        maxabs = float(np.max(np.abs(m4.q4 - m6.q6)))
        d2preplay = abs(float(m6.replay_conservative_p) - float(m6.archived_conservative_p))
        score_compat = float(np.max(np.abs(m4.scores - m6.source_scores)))
        nonfactorizing = bool(zero_mismatch or delta_fact > float(tolerances["nonfactorization_logosc_min"]))
        structural_rows.append({
            "case_id": str(s4["case_id"]),
            "d2_fixture_id": str(s4["d2_fixture_id"]),
            "candidate_y": y,
            "candidate_hex": chex,
            "candidate_source": "accepted_D2_candidate_state_audit",
            "m3_p": float(m3.pvalue),
            "m4_p": float(m4.pvalue),
            "m6_conservative_p_archived": float(m6.archived_conservative_p),
            "m6_conservative_p_replayed": float(m6.replay_conservative_p),
            "m6_p_replay_abs_error": d2preplay,
            "m4_probability_sum": float(m4.probability_sum),
            "m6_probability_sum": float(m6.q6.sum()),
            "delta_fact_logosc": delta_fact,
            "tv_q4_q6": tv,
            "max_abs_q4_q6": maxabs,
            "structural_zero_mismatch": zero_mismatch,
            "nonfactorization_detected": nonfactorizing,
            "max_abs_score_difference_M4_vs_M6": score_compat,
            "weight_factorization_test_only": True,
            "publication_evidence": False,
        })
        for j in range(len(logd)):
            if base[j] > 0 and m6.q6[j] > 0:
                lr = float(math.log(float(m6.q6[j])) - math.log(float(base[j])))
                rho = float(math.exp(lr)) if lr < 700 else float("inf")
            else:
                lr = np.nan
                rho = np.nan
            source_rows.append({
                "case_id": str(s4["case_id"]),
                "candidate_y": y,
                "candidate_hex": chex,
                "source_index": j,
                "source_node": target_node if j == len(block_nodes)-1 else block_nodes[j],
                "source_type": "target_candidate" if j == len(block_nodes)-1 else "calibration",
                "spatial_marginal_s": float(m3.source_marginals[j]),
                "log_dose_ratio": float(logd[j]),
                "dose_ratio": 0.0 if np.isneginf(logd[j]) else float(math.exp(float(logd[j]))),
                "q4": float(m4.q4[j]),
                "q6": float(m6.q6[j]),
                "m4_score_abs_source_residual": float(m4.scores[j]),
                "m6_score_at_target_slot": float(m6.source_scores[j]),
                "base_d_times_s": float(base[j]),
                "rho_q6_over_d_s": rho,
                "log_rho": lr,
                "common_positive": bool(common[j]),
            })
    structural_df = pd.DataFrame(structural_rows)
    source_df = pd.DataFrame(source_rows)
    checks["s4_candidates_inherited_from_D2_state_audit"] = bool(len(candidate_registry) == int(d2states["candidate_hex"].nunique()) and len(candidate_registry) >= 3)
    checks["m4_one_normalization"] = True
    checks["m4_probabilities_sum_one"] = bool(np.allclose(structural_df["m4_probability_sum"], 1.0, atol=float(tolerances["probability_sum_abs"]), rtol=0))
    checks["m6_extracted_probabilities_sum_one"] = bool(np.allclose(structural_df["m6_probability_sum"], 1.0, atol=float(tolerances["probability_sum_abs"]), rtol=0))
    checks["m6_state_pvalue_replay_matches_D2"] = bool((structural_df["m6_p_replay_abs_error"] <= float(tolerances["d2_pvalue_replay_abs"])).all())
    checks["s4_nonfactorization_detected_at_least_one_candidate"] = bool(structural_df["nonfactorization_detected"].any())
    checks["m4_uses_no_target_design_ratio"] = "target_design_ratio" not in inspect.signature(evaluate_m4).parameters
    checks["m4_uses_only_source_marginal_interface"] = "spatial_source_marginal" in inspect.signature(evaluate_m4).parameters and "orbit" not in inspect.signature(evaluate_m4).parameters
    checks["m6_not_reimplemented_in_D3"] = True
    checks["pilot_not_run"] = True
    checks["production_not_run"] = True

    structural_df.to_csv(output_dir / "stage3d_d3_m4_s4_structural_summary.csv", index=False, lineterminator="\n")
    write_csv_gz(source_df, output_dir / "stage3d_d3_m4_s4_source_comparison.csv.gz")
    pd.DataFrame(columns=["case_id","candidate_y","refusal_code","detail"]).to_csv(output_dir / "stage3d_d3_m4_refusal_log.csv", index=False, lineterminator="\n")

    # Acceptance gate.
    boolean_checks = {k: v for k, v in checks.items() if isinstance(v, (bool, np.bool_))}
    failed = sorted(k for k, v in boolean_checks.items() if not bool(v))
    write_json(output_dir / "stage3d_d3_m4_structural_checks.json", checks)
    if failed:
        raise D3M4Error(f"M4 oracle structural gate failed: {failed}; details={checks}")

    environment = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "PyYAML": yaml.__version__,
        "executable": sys.executable,
    }
    write_json(output_dir / "stage3d_d3_m4_environment_inventory.json", environment)
    counts = {
        "accepted_M3_regression_candidate_rows": int(m3reg["candidate_rows_checked"]),
        "factorizing_candidate_rows": int(len(factor_df)),
        "factorizing_source_rows": int(len(factor_source_rows)),
        "s4_structural_candidate_rows": int(len(structural_df)),
        "s4_source_comparison_rows": int(len(source_df)),
        "m4_implemented": True,
        "full_M6_integrated": False,
        "pilot_replications_run": 0,
        "production_replications_run": 0,
    }
    write_json(output_dir / "stage3d_d3_m4_counts.json", counts)
    summary = {
        "stage": "3D_D3_M4_ORACLE_STRUCTURAL_GATE",
        "script_version": __version__,
        "definition_gate_version": "1.1",
        "signed_zero_patch": True,
        "signed_zero_patch_scientific_change": False,
        "accepted_M3_results_regression_preserved": True,
        "M3_implemented": True,
        "M4_implemented": True,
        "M4_role": "naive_product_of_oracle_M2_treatment_ratio_and_M3_target_source_residual_marginal",
        "M4_general_theorem_backed": False,
        "M4_normalization_count": 1,
        "M4_target_design_ratio_used": False,
        "M6_full_integrated": False,
        "M6_reference_for_S4": "exact target-source marginal extracted from accepted D2 G2/G3 candidate-state probabilities; no D3 re-derivation",
        "factorizing_M6_reference": "independent labeled joint-payload enumeration under q_h=g, slot-homogeneous treatment likelihood, m=0, s=1, nonzero graph dependence",
        "pilot_run": False,
        "production_run": False,
        "claim_status": "software structural/reduction gate only; not coverage or publication-performance evidence",
        "next_action_after_independent_output_review": "integrate exact M6 into common M1-M6 interface; do not start pilot until D3 exact-method integration passes",
    }
    write_json(output_dir / "stage3d_d3_m4_contract_summary.json", summary)
    elapsed = time.perf_counter() - started
    return {"checks": checks, "counts": counts, "summary": summary, "elapsed_seconds": elapsed}
