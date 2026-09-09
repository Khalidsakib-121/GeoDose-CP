from __future__ import annotations

import numpy as np
import pandas as pd

from geodose_stage3c.graph import build_graph_cache, deterministic_graph_safe_set
from geodose_stage3c.methods import build_interval
from geodose_stage3d_d3_m4.m3_oracle import evaluate_candidate, candidate_residuals
from geodose_stage3d_d3_m4.m4_oracle import evaluate_m4
from geodose_stage3d_d3_m4.runner import _case_inputs, _oracle_log_dose_ratio

from .d2_adapter import AcceptedD2ExactAdapter
from .io import IntegrationError


def _method_metadata(method: str) -> tuple[str,str,str]:
    if method == "M1": return "principal_baseline", "frozen_stage3c_exact_source", "exchangeability_only_when_eligible"
    if method == "M2": return "principal_baseline", "frozen_stage3c_exact_source", "weighted_conformal_under_its_conditions"
    if method == "M3": return "principal_baseline", "accepted_d3_m3_exact_engine", "conditional_response_spatial_comparator"
    if method == "M4": return "principal_baseline", "accepted_d3_m4_exact_engine", "naive_product_not_general_theorem"
    if method == "M5": return "principal_baseline", "frozen_stage3c_exact_source", "graph_safe_fallback_under_sampling_conditions"
    if method == "M6": return "principal_method", "accepted_D2_output_backed_exact_reference", "G1_G2_G3_exact_reference_domain_truncated"
    raise IntegrationError(method)


def build_s4_common_trace(data, d2_trace: pd.DataFrame, alpha: float, tol: float, minimum_candidates: int) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    case_id="S4_RHO060"; fixture_id="G6_GAUSS_INTERIOR"; target_dose=0.9
    block_nodes=[294,318,319,343,344,320]; target_node=320
    case_units, case_edges, case_row, target_row, cal_resid, target_mean, target_scale, target_A, law = _case_inputs(
        data, case_id, "localized", target_dose, block_nodes, target_node
    )
    logd = _oracle_log_dose_ratio(data, case_id, case_units, target_row, block_nodes, target_dose)
    target_pos=len(block_nodes)-1

    # Frozen Stage3C M1/M2/M5 intervals on exactly the same five calibration slots and target center.
    scores=np.abs(cal_resid)
    order=np.argsort(scores, kind="mergesort")
    graph=build_graph_cache(case_units, case_edges, np.asarray([target_node], int))
    safe_nodes=deterministic_graph_safe_set(graph, target_node, np.asarray(block_nodes[:-1], int))
    safe_mask=np.isin(np.asarray(block_nodes[:-1],int),safe_nodes)
    baseline_specs={
        "M1": (np.zeros(len(cal_resid)), 0.0),
        "M2": (logd[:-1], logd[-1]),
        "M5": (np.where(safe_mask, logd[:-1], -np.inf), logd[-1]),
    }
    baseline={}
    for m,(lw,tlw) in baseline_specs.items():
        baseline[m]=build_interval(target_mean, scores[order], np.asarray(lw)[order], float(tlw), alpha)

    adapter=AcceptedD2ExactAdapter(d2_trace, fixture_id, alpha)
    grid=adapter.trace.sort_values("candidate_y", kind="mergesort").reset_index(drop=True)
    if len(grid) < int(minimum_candidates):
        raise IntegrationError(f"Accepted D2 integration grid too small: {len(grid)}")
    hex_values=grid["candidate_hex"].astype(str).map(float.fromhex).to_numpy(float)
    if float(np.max(np.abs(hex_values-grid["candidate_y"].astype(float).to_numpy()))) > 2e-12:
        raise IntegrationError("Accepted D2 candidate decimal/hex registry is inconsistent")

    rows=[]
    summary_acc={m:{"included":0,"returned":0,"refused":0} for m in ["M1","M2","M3","M4","M5","M6"]}
    max_prob_err=0.0
    for d2row in grid.itertuples(index=False):
        h=str(d2row.candidate_hex); y=float.fromhex(h)
        m3=evaluate_candidate(cal_resid,y,target_mean,law,target_pos,target_scale=target_scale,tie_tolerance=1e-12,keep_states=False)
        e=candidate_residuals(cal_resid,y,target_mean,target_scale)
        m4=evaluate_m4(m3.source_marginals,e,logd,target_pos,tie_tolerance=1e-12)
        m6=adapter.evaluate_registered(y,h)
        max_prob_err=max(max_prob_err,abs(m3.probability_sum-1.0),abs(m4.probability_sum-1.0),abs(m6.probability_sum-1.0))

        for method in ["M1","M2","M5"]:
            r=baseline[method]
            returned=r.interval_status!="refused"
            included=bool(returned and r.lower <= y <= r.upper)
            role,source,theorem=_method_metadata(method)
            rows.append({
                "case_id":case_id,"fixture_id":fixture_id,"candidate_y":y,"candidate_hex":h,
                "method":method,"method_role":role,"representation":"interval",
                "p_value":np.nan,"lower":r.lower,"upper":r.upper,"included":included,
                "returned":returned,"refusal_code":r.refusal_code,"nuisance_variant":"OT_TG",
                "theorem_status":theorem,"source_adapter":source,"alpha":alpha,"domain_lower":-8.0,"domain_upper":8.0,
                "domain_truncated":True,"publication_evidence":False
            })
            summary_acc[method]["included"]+=int(included); summary_acc[method]["returned"]+=int(returned); summary_acc[method]["refused"]+=int(not returned)

        for method,p,inc,source_repr in [
            ("M3",m3.pvalue,m3.pvalue>alpha,"candidate_pvalue"),
            ("M4",m4.pvalue,m4.pvalue>alpha,"candidate_pvalue"),
            ("M6",m6.p_value,m6.included,"candidate_pvalue"),
        ]:
            role,source,theorem=_method_metadata(method)
            rows.append({
                "case_id":case_id,"fixture_id":fixture_id,"candidate_y":y,"candidate_hex":h,
                "method":method,"method_role":role,"representation":source_repr,
                "p_value":float(p),"lower":np.nan,"upper":np.nan,"included":bool(inc),
                "returned":True,"refusal_code":"","nuisance_variant":"OT_TG",
                "theorem_status":theorem,"source_adapter":source,"alpha":alpha,"domain_lower":-8.0,"domain_upper":8.0,
                "domain_truncated":True,"publication_evidence":False
            })
            summary_acc[method]["included"]+=int(inc); summary_acc[method]["returned"]+=1

    trace=pd.DataFrame(rows)
    summary=[]
    for method in ["M1","M2","M3","M4","M5","M6"]:
        g=trace[trace.method==method]
        finite_interval = g[(g.representation=="interval") & np.isfinite(g.lower) & np.isfinite(g.upper)]
        summary.append({
            "case_id":case_id,"fixture_id":fixture_id,"method":method,"candidate_count":len(g),
            "included_count":int(g.included.sum()),"included_fraction":float(g.included.mean()),
            "returned_fraction":float(g.returned.mean()),"refused_count":int((~g.returned).sum()),
            "pvalue_min":float(g.p_value.min()) if g.p_value.notna().any() else np.nan,
            "pvalue_max":float(g.p_value.max()) if g.p_value.notna().any() else np.nan,
            "finite_interval_width":float((finite_interval.upper-finite_interval.lower).iloc[0]) if len(finite_interval) else np.nan,
            "publication_evidence":False,
        })
    meta={
        "case_id":case_id,"fixture_id":fixture_id,"candidate_count":len(grid),"block_nodes":block_nodes,
        "target_node":target_node,"target_A_star":float(target_A),"target_mean":float(target_mean),
        "spatial_rho":float(case_row["spatial_rho"]),"safe_nodes_M5":list(map(int,safe_nodes.tolist())),
        "max_probability_sum_abs_error":float(max_prob_err),"M6_adapter":"accepted_D2_output_backed_exact_reference",
        "new_data_M6_engine":False,
    }
    if max_prob_err > tol:
        raise IntegrationError("Common-grid probability normalization failed")
    return trace,pd.DataFrame(summary),meta
