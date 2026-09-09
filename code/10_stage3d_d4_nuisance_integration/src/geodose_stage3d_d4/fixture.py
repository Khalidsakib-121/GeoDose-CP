from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import fields, is_dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .data import Stage3BExternal, case_row, target_row
from .io import D4Error

DEFAULT_BLOCK_NODES=(294,318,319,343,344,320)
DEFAULT_TARGET_NODE=320


def canonical_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if len(edges)==0:
        return pd.DataFrame(columns=['source_node','target_node'])
    a=np.minimum(edges['source_node'].to_numpy(int),edges['target_node'].to_numpy(int))
    b=np.maximum(edges['source_node'].to_numpy(int),edges['target_node'].to_numpy(int))
    out=pd.DataFrame({'source_node':a,'target_node':b})
    return out[out.source_node!=out.target_node].drop_duplicates().sort_values(['source_node','target_node'],kind='mergesort').reset_index(drop=True)


def edge_degrees(edges: pd.DataFrame, n_nodes: int) -> np.ndarray:
    d=np.zeros(int(n_nodes),dtype=int)
    for a,b in canonical_edges(edges).itertuples(index=False):
        d[int(a)]+=1; d[int(b)]+=1
    return d


def block_connected(block_nodes: tuple[int,...]|list[int], edges: pd.DataFrame) -> bool:
    nodes=set(map(int,block_nodes))
    if not nodes: return False
    adj={n:set() for n in nodes}
    for a,b in canonical_edges(edges).itertuples(index=False):
        if a in nodes and b in nodes:
            adj[a].add(b); adj[b].add(a)
    seen=set(); q=[next(iter(nodes))]
    while q:
        x=q.pop()
        if x in seen: continue
        seen.add(x); q.extend(adj[x]-seen)
    return seen==nodes


def deterministic_connected_block(case_units: pd.DataFrame, fitted_edges: pd.DataFrame, target_node: int=DEFAULT_TARGET_NODE, size: int=6) -> tuple[int,...]:
    lookup=case_units.set_index('node_index')
    if target_node not in lookup.index or str(lookup.loc[target_node,'role'])!='test_target':
        raise D4Error(f"Target node {target_node} is not a test_target")
    adj={int(n):set() for n in case_units['node_index'].astype(int)}
    for a,b in canonical_edges(fitted_edges).itertuples(index=False):
        adj.setdefault(int(a),set()).add(int(b)); adj.setdefault(int(b),set()).add(int(a))
    q=deque([int(target_node)]); seen={int(target_node)}; calibration=[]
    while q and len(calibration)<size-1:
        x=q.popleft()
        for y in sorted(adj.get(x,set())):
            if y in seen: continue
            seen.add(y); q.append(y)
            if y in lookup.index and str(lookup.loc[y,'role'])=='calibration':
                calibration.append(int(y))
                if len(calibration)>=size-1: break
    if len(calibration)!=size-1:
        raise D4Error(f"Could not find {size-1} fitted-graph-connected calibration nodes around target {target_node}")
    block=tuple(calibration+[int(target_node)])
    if not block_connected(block,fitted_edges):
        raise D4Error("Deterministic fitted-graph block construction failed connectivity audit")
    return block


def choose_block(data: Stage3BExternal, case_id: str, block_rule: str) -> tuple[int,...]:
    cu=data.units[data.units.case_id.astype(str)==str(case_id)].copy()
    true_e=data.true_edges[data.true_edges.case_id.astype(str)==str(case_id)][['source_node','target_node']]
    fit_e=data.fitted_edges[data.fitted_edges.case_id.astype(str)==str(case_id)][['source_node','target_node']]
    if block_rule=='frozen_stage3b_size6_coordinates':
        block=tuple(DEFAULT_BLOCK_NODES)
        if not block_connected(block,true_e): raise D4Error(f"Frozen size-6 block disconnected in true graph for {case_id}")
        return block
    if block_rule=='deterministic_connected_under_fitted_graph':
        block=deterministic_connected_block(cu,fit_e,DEFAULT_TARGET_NODE,6)
        if not block_connected(block,true_e): raise D4Error(f"Fitted-connected block not connected in true graph for {case_id}")
        return block
    raise D4Error(f"Unknown block rule: {block_rule}")


def build_exact_fixture_dict(data: Stage3BExternal, case_id: str, target_dose: float, block_nodes: tuple[int,...], fixture_edges: pd.DataFrame|None=None) -> dict[str,Any]:
    cu=data.units[data.units.case_id.astype(str)==str(case_id)].copy().set_index('node_index',drop=False)
    if len(block_nodes)!=6 or len(set(block_nodes))!=6 or block_nodes[-1]!=DEFAULT_TARGET_NODE:
        raise D4Error("D4 exact fixture requires five calibration nodes followed by target node 320")
    rows=[]
    for n in block_nodes:
        if n not in cu.index: raise D4Error(f"Node {n} absent in case {case_id}")
        rows.append(cu.loc[n])
    if any(str(r['role'])!='calibration' for r in rows[:-1]) or str(rows[-1]['role'])!='test_target':
        raise D4Error(f"D4 block roles invalid for {case_id}")
    target_unit_id=str(rows[-1]['unit_id'])
    tr=target_row(data,case_id,target_unit_id,float(target_dose))
    slots=[{
        'unit_id':str(r['unit_id']),'node_index':int(r['node_index']),'grid_row':int(r['grid_row']),
        'grid_col':int(r['grid_col']),'role':str(r['role'])
    } for r in rows]
    payloads=[]
    for r in rows[:-1]:
        payloads.append({'A':float(r['A']),'Y':float(r['Y_observed_at_A']),'payload_origin':'factual_calibration_observation'})
    payloads.append({'A':float(tr['A_star']),'Y':float(tr['Y_observed_at_A_star']),'payload_origin':'realized_localized_target_truth_reference'})
    if fixture_edges is None:
        cedges=data.true_edges[data.true_edges.case_id.astype(str)==str(case_id)][['source_node','target_node']].copy()
    else:
        cedges=fixture_edges[['source_node','target_node']].copy()
    node_set=set(map(int,block_nodes))
    internal=cedges[cedges.source_node.isin(node_set)&cedges.target_node.isin(node_set)][['source_node','target_node']]
    boundary=cedges[cedges.source_node.isin(node_set)^cedges.target_node.isin(node_set)][['source_node','target_node']]
    if not block_connected(block_nodes,cedges): raise D4Error(f"Custom exact block disconnected for {case_id}")
    mult={}
    for p in payloads:
        key=f"A={float(p['A']):.17g}|Y={float(p['Y']):.17g}"; mult[key]=mult.get(key,0)+1
    distinct=math.factorial(len(payloads))
    for m in mult.values(): distinct//=math.factorial(m)
    return {
        'fixed_slots':slots,
        'movable_payloads':payloads,
        'payload_definition':['A','Y'],
        'target_slot_position':5,
        'target_slot_node_index':DEFAULT_TARGET_NODE,
        'target_truth_reference':{
            'unit_id':target_unit_id,'target_dose':float(tr['target_dose']),
            'bandwidth':None if pd.isna(tr['bandwidth']) else float(tr['bandwidth']),
            'A_star':float(tr['A_star']),'Y_true_at_A_star':float(tr['Y_true_at_A_star']),
            'Y_observed_at_A_star':float(tr['Y_observed_at_A_star']),
            'G3_rule':'replace target payload Y by candidate y for every inversion state; stored Y is diagnostic truth only'
        },
        'candidate_response_replacement_required':True,
        'calibration_slot_count':5,
        'payload_multiplicities':mult,
        'distinct_permutations':int(distinct),
        'internal_edges':internal.to_dict(orient='records'),
        'boundary_edges':boundary.to_dict(orient='records'),
        'block_connected':True,
    }


def _public_attrs(obj: Any) -> dict[str,Any]:
    if is_dataclass(obj): return {f.name:getattr(obj,f.name) for f in fields(obj)}
    if hasattr(obj,'__dict__'): return {k:v for k,v in vars(obj).items() if not k.startswith('_')}
    if isinstance(obj,dict): return dict(obj)
    raise D4Error(f"Unsupported Stage3B data container type: {type(obj)}")


def discover_data_fields(data_obj: Any) -> dict[str,str]:
    attrs=_public_attrs(data_obj); result={}
    edge_candidates=[]
    for name,val in attrs.items():
        if isinstance(val,pd.DataFrame):
            cols=set(val.columns)
            if {'case_id','unit_id','node_index','role','A','pi_atom_0','beta_alpha'}.issubset(cols): result['units']=name
            if {'case_id','scenario_id','spatial_rho','residual_law'}.issubset(cols): result['cases']=name
            if {'case_id','source_node','target_node'}.issubset(cols): edge_candidates.append(name)
            if {'case_id','unit_id','target_dose','A_star','Y_observed_at_A_star'}.issubset(cols): result['targets']=name
        if isinstance(val,dict) and 'size6_unique' in val: result['fixtures']=name
    for n in edge_candidates:
        low=n.lower()
        if 'fitted' in low: result['fitted_edges']=n
        elif 'true' in low: result['true_edges']=n
    if 'true_edges' not in result:
        # D1/D2 only needs one oracle edge table, and some accepted container
        # implementations may call it simply `edges`/`graph_edges`.  Use that
        # fallback only when it is unambiguous; never guess among multiple tables.
        nonfitted=[n for n in edge_candidates if n!=result.get('fitted_edges')]
        if len(nonfitted)==1: result['true_edges']=nonfitted[0]
    # The accepted D1/D2 oracle container is only required to expose the graph
    # actually consumed by the exact law.  Some accepted loaders may omit a
    # separate fitted-edge attribute because D1/D2 did not need it.  D4 obtains
    # fitted edges independently from the verified Stage3B output and therefore
    # treats a base-container fitted-edge field as optional.
    required={'units','cases','true_edges','fixtures'}
    miss=required-set(result)
    if miss:
        raise D4Error(f"Could not discover D2 Stage3B data fields {sorted(miss)}; available={sorted(attrs)}")
    return result


def clone_data_object(data_obj: Any, overrides: dict[str,Any]) -> Any:
    if is_dataclass(data_obj): return replace(data_obj,**overrides)
    if isinstance(data_obj,dict):
        out=dict(data_obj); out.update(overrides); return out
    out=copy.copy(data_obj)
    for k,v in overrides.items(): setattr(out,k,v)
    return out


def replace_fixture_mapping(mapping: dict[str,Any], fixture: dict[str,Any], case_id: str, target_dose: float) -> dict[str,Any]:
    out=copy.deepcopy(mapping)
    if 'size6_unique' not in out: raise D4Error("D2 Stage3B fixture mapping lacks size6_unique")
    out['size6_unique']=copy.deepcopy(fixture)
    if 'source_case_id' in out: out['source_case_id']=str(case_id)
    if 'source_target_dose' in out: out['source_target_dose']=float(target_dose)
    return out
