from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Stage3BExternal, case_row, target_row
from .fixture import edge_degrees
from .io import D4Error
from .nuisance import EstimatedTreatmentFit, estimated_log_dose_ratios, oracle_log_dose_ratios


def import_accepted_m3m4(package_root: Path):
    vendor=Path(package_root)/'vendor'
    if str(vendor) not in sys.path: sys.path.insert(0,str(vendor))
    from geodose_stage3d_d3_m4.graph_law import build_local_graph_law
    from geodose_stage3d_d3_m4.m3_oracle import evaluate_candidate, candidate_residuals
    from geodose_stage3d_d3_m4.m4_oracle import evaluate_m4
    return build_local_graph_law,evaluate_candidate,candidate_residuals,evaluate_m4


@dataclass(frozen=True)
class M3M4Result:
    m3_p: float
    m4_p: float
    m3_source_marginal: np.ndarray
    m4_q: np.ndarray
    source_residuals: np.ndarray
    log_dose_ratio: np.ndarray
    state_count: int


def graph_inputs(data: Stage3BExternal, case_id: str, graph_mode: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    units=data.units[data.units.case_id.astype(str)==str(case_id)].copy().sort_values('node_index').reset_index(drop=True)
    if graph_mode=='TG':
        edges=data.true_edges[data.true_edges.case_id.astype(str)==str(case_id)][['source_node','target_node']].copy()
        # Stage3B graph_degree_true is already for the true graph, but recompute as a cross-check and to avoid stale metadata.
    elif graph_mode=='FG':
        edges=data.fitted_edges[data.fitted_edges.case_id.astype(str)==str(case_id)][['source_node','target_node']].copy()
    else:
        raise D4Error(f'Unknown graph mode {graph_mode}')
    n=int(units['node_index'].max())+1
    deg=edge_degrees(edges,n)
    if len(deg)!=len(units): raise D4Error('Graph degree vector length mismatch')
    units['graph_degree_true']=deg.astype(int)
    return units,edges


def calibration_residuals(case_units: pd.DataFrame, block_nodes: tuple[int,...]) -> np.ndarray:
    lookup=case_units.set_index('node_index')
    vals=[]
    for node in block_nodes[:-1]:
        r=lookup.loc[int(node)]
        vals.append(float(r['Y_observed_at_A'])-float(r['conditional_mean_at_A']))
    arr=np.asarray(vals,dtype=float)
    if not np.isfinite(arr).all(): raise D4Error('Nonfinite calibration oracle residuals')
    return arr


def evaluate_m3m4(
    package_root: Path,
    data: Stage3BExternal,
    stage3c_source_root: Path,
    case_id: str,
    target_dose: float,
    block_nodes: tuple[int,...],
    treatment_mode: str,
    graph_mode: str,
    candidate_y: float,
    estimated_fit: EstimatedTreatmentFit|None,
    tie_tolerance: float=1e-12,
) -> M3M4Result:
    build_law,eval_m3,candidate_resids,eval_m4=import_accepted_m3m4(package_root)
    crow=case_row(data,case_id)
    units_for_graph,edges=graph_inputs(data,case_id,graph_mode)
    target_unit=units_for_graph[units_for_graph.node_index.astype(int)==int(block_nodes[-1])]
    if len(target_unit)!=1: raise D4Error('Target unit lookup failed')
    tr=target_row(data,case_id,str(target_unit.iloc[0]['unit_id']),target_dose)
    law=build_law(
        units_for_graph,edges,list(block_nodes),float(crow['spatial_rho']),str(crow['residual_law']),scale=0.35
    )
    cal=calibration_residuals(units_for_graph,block_nodes)
    target_mean=float(tr['conditional_mean_at_A_star'])
    m3=eval_m3(cal,float(candidate_y),target_mean,law,len(block_nodes)-1,target_scale=1.0,tie_tolerance=tie_tolerance,keep_states=False)
    src_res=candidate_resids(cal,float(candidate_y),target_mean,1.0)
    if treatment_mode=='OT':
        logd=oracle_log_dose_ratios(data,case_id,target_dose,block_nodes)
    elif treatment_mode=='ET':
        if estimated_fit is None: raise D4Error('ET requested without estimated treatment fit')
        original_case_units=data.units[data.units.case_id.astype(str)==str(case_id)].copy()
        logd=estimated_log_dose_ratios(stage3c_source_root,estimated_fit,original_case_units,tr,block_nodes,target_dose)
    else:
        raise D4Error(f'Unknown treatment mode {treatment_mode}')
    m4=eval_m4(m3.source_marginals,src_res,logd,len(block_nodes)-1,tie_tolerance=tie_tolerance)
    return M3M4Result(float(m3.pvalue),float(m4.pvalue),m3.source_marginals.copy(),m4.q4.copy(),src_res.copy(),logd.copy(),int(m3.state_count))
