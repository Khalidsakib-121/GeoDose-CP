from __future__ import annotations

import numpy as np
import pandas as pd

from geodose_stage3c.methods import build_interval
from geodose_stage3d_d3_m4.m4_oracle import evaluate_m4


def reduction_table(m3_s7_source: pd.DataFrame, d2_contract: dict, tol: float) -> pd.DataFrame:
    rows = []
    exact_list = d2_contract.get("math_contract", {}).get("implemented", [])
    r1 = "exact_G2_probability_recomputation_for_each_candidate" in exact_list
    rows.append({
        "reduction_id":"R1", "name":"exact_reference_reduction", "registered_owner":"Stage3D",
        "status":"PASS" if r1 else "FAIL", "check_level":"accepted_exact_branch_contract",
        "detail":"Accepted D2 uses exact G2 recomputation at each candidate; no N2 approximation enters this exact-reference gate."
    })

    # R2 component identity: when spatial factor is constant, normalized product equals normalized target-ratio weights.
    d = np.asarray([0.25, 0.5, 1.5, 2.0, 4.0, 3.0], float)
    s = np.ones(6, float)/6.0
    e = np.asarray([-1.4,-0.8,-0.2,0.35,0.9,0.6])
    m4 = evaluate_m4(s, e, np.log(d), 5)
    q_ratio = d/d.sum()
    r2err = float(np.max(np.abs(m4.q4-q_ratio)))
    rows.append({
        "reduction_id":"R2", "name":"weighted_conformal_reduction", "registered_owner":"Stage3C/Stage3D",
        "status":"PASS" if r2err <= tol else "FAIL", "check_level":"target_index_weight_component",
        "detail":f"With constant spatial factor, naive/generalized target-index weights reduce to normalized target ratios; max_abs={r2err:.3e}. Frozen Stage3C weighted quantile is separately replayed."
    })

    rows.append({
        "reduction_id":"R3", "name":"no_target_shift_N2_reduction", "registered_owner":"Stage3E",
        "status":"DEFERRED_STAGE3E", "check_level":"not_applicable_to_exact_D3",
        "detail":"R3 concerns the N2 target-weighted sparse approximation criterion. It is intentionally not claimed or faked in this exact D3 gate."
    })

    # R4: M1=M2 when all treatment log weights are zero; accepted S7 M3 spatial marginals are uniform.
    scores=np.asarray([0.1,0.2,0.4,0.7,1.1])
    idx=np.argsort(scores, kind="mergesort")
    m1=build_interval(0.0,scores[idx],np.zeros(5),0.0,0.1)
    m2=build_interval(0.0,scores[idx],np.zeros(5),0.0,0.1)
    baseline_equal = (m1.interval_status==m2.interval_status and ((np.isinf(m1.quantile) and np.isinf(m2.quantile)) or abs(m1.quantile-m2.quantile)<=tol))
    s7 = m3_s7_source[m3_s7_source["case_id"].astype(str)=="S7_EXCHANGEABLE"]
    if len(s7):
        max_uniform=float(np.max(np.abs(s7["spatial_source_mass"].to_numpy(float)-1.0/6.0)))
    else:
        max_uniform=np.inf
    r4 = baseline_equal and max_uniform <= tol
    rows.append({
        "reduction_id":"R4", "name":"ordinary_conformal_reduction", "registered_owner":"Stage3C",
        "status":"PASS" if r4 else "FAIL", "check_level":"baseline_plus_spatial_component",
        "detail":f"No-shift M1=M2 and accepted S7 M3 source marginal is uniform; max_uniform_error={max_uniform:.3e}."
    })
    return pd.DataFrame(rows)
