from __future__ import annotations
from dataclasses import replace
from typing import Any
import numpy as np
from .io import require

def install_stage3b_production_adapter(core, stage3a:dict[str,Any]):
    """Patch only the two Stage3B extension seams needed by production.

    Frozen Stage3B source itself is never modified. The production seed selector
    uses the already-frozen Stage3A production_provisional rows. 17/35-column
    role/cut definitions are the prospectively frozen F06 extension; all original
    25/13 behavior delegates byte-for-byte to accepted Stage3B functions.
    """
    orig_role=core.role_assignment; orig_cuts=core.role_independence_cuts
    def production_seed_row(stage3a_obj,scenario_id:str,replication:int=1):
        seeds=stage3a_obj['frames']['seed_registry.csv'];source=scenario_id if scenario_id.startswith('S') and scenario_id[1:].isdigit() else 'S4'
        rows=seeds[(seeds.phase.astype(str)=='production_provisional')&(seeds.scenario_id.astype(str)==source)&(seeds.replication.astype(int)==int(replication))]
        require(len(rows)==1,f'MISSING_UNIQUE_PRODUCTION_SEED_ROW:{source}:{replication}')
        row=rows.iloc[0];return {c:int(row[c]) for c in ['data_seed','split_seed','nuisance_seed','orbit_seed','measurement_seed','target_draw_seed']}
    def role_assignment(n_rows:int,n_cols:int,sample_design:str):
        cols=np.tile(np.arange(n_cols),n_rows);roles=np.full(n_rows*n_cols,'nuisance_training',dtype=object)
        if n_cols==17 and sample_design=='f06_small':
            roles[(cols>=7)&(cols<=9)]='support_audit';roles[(cols>=10)&(cols<=12)]='calibration';roles[cols>=13]='test_target';return roles
        if n_cols==35 and sample_design=='f06_large':
            roles[(cols>=14)&(cols<=20)]='support_audit';roles[(cols>=21)&(cols<=27)]='calibration';roles[cols>=28]='test_target';return roles
        return orig_role(n_rows,n_cols,sample_design)
    def role_independence_cuts(n_cols:int,sample_design:str):
        if n_cols==17 and sample_design=='f06_small':return (6,9)
        if n_cols==35 and sample_design=='f06_large':return (13,20)
        return orig_cuts(n_cols,sample_design)
    core.source_seed_row=production_seed_row;core.role_assignment=role_assignment;core.role_independence_cuts=role_independence_cuts
    return {'seed_phase':'production_provisional','original_25_13_role_functions_delegated':True,'f06_extension_widths':[17,35]}

def install_stage3c_production_nuisance_adapter(c3_runner):
    def production_nuisance_seed(stage3a_seeds,scenario_id:str,replication:int):
        source=scenario_id if scenario_id.startswith('S') and scenario_id[1:].isdigit() else 'S4'
        rows=stage3a_seeds[(stage3a_seeds.phase.astype(str)=='production_provisional')&(stage3a_seeds.scenario_id.astype(str)==source)&(stage3a_seeds.replication.astype(int)==int(replication))]
        require(len(rows)==1,f'MISSING_PRODUCTION_NUISANCE_SEED:{source}:{replication}')
        return source,int(rows.iloc[0].nuisance_seed)
    c3_runner.frozen_nuisance_seed=production_nuisance_seed

def f06_case(base,level:str):
    if level=='small':return replace(base,case_id=f'F06_SMALL__{base.case_id}',case_name=f'F06 small 17x17 :: {base.case_name}',case_type='f06_sample_size',common_random_group=f'{base.common_random_group}|F06_SMALL',n_rows=17,n_cols=17,sample_design='f06_small',support_scale='90m_f06_small')
    if level=='large':return replace(base,case_id=f'F06_LARGE__{base.case_id}',case_name=f'F06 large 35x35 :: {base.case_name}',case_type='f06_sample_size',common_random_group=f'{base.common_random_group}|F06_LARGE',n_rows=35,n_cols=35,sample_design='f06_large',support_scale='90m_f06_large')
    if level=='primary':return base
    raise ValueError(level)


def _graph_component_count(n:int,edges)->int:
    adj=[set() for _ in range(int(n))]
    for a,b in edges[['source_node','target_node']].itertuples(index=False,name=None):
        a=int(a);b=int(b);adj[a].add(b);adj[b].add(a)
    seen=set();components=0
    for u in range(int(n)):
        if u in seen:continue
        components+=1;stack=[u];seen.add(u)
        while stack:
            v=stack.pop()
            for w in adj[v]:
                if w not in seen:seen.add(w);stack.append(w)
    return components

def _degree_counts(n:int,edges)->dict[int,int]:
    deg=np.zeros(int(n),dtype=int)
    for a,b in edges[['source_node','target_node']].itertuples(index=False,name=None):
        deg[int(a)]+=1;deg[int(b)]+=1
    vals,cnt=np.unique(deg,return_counts=True)
    return {int(v):int(c) for v,c in zip(vals,cnt)}

def f06_structural_qa(core,contract:dict)->list[dict]:
    rows=[]
    for level,spec in contract['f06_freeze']['levels'].items():
        nr,nc=int(spec['rows']),int(spec['cols']);design={'small':'f06_small','primary':'primary','large':'f06_large'}[level]
        full,_=core.build_grid_graph(nr,nc,True,());cuts=tuple(spec['cut_after_cols']);cut,_=core.build_grid_graph(nr,nc,True,cuts)
        roles=core.role_assignment(nr,nc,design);vc={k:int(np.sum(roles==k)) for k in ['nuisance_training','support_audit','calibration','test_target']}
        expected_cols=spec['role_columns'];expected_counts={k:int(v)*nr for k,v in expected_cols.items()}
        full_components=_graph_component_count(nr*nc,full); cut_components=_graph_component_count(nr*nc,cut); deg=_degree_counts(nr*nc,full)
        # Nonperiodic queen lattice: 4 corners degree 3, boundary non-corners
        # 2*(rows-2)+2*(cols-2) degree 5, all interior degree 8.
        expected_deg={3:4,5:2*max(0,nr-2)+2*max(0,nc-2),8:max(0,nr-2)*max(0,nc-2)}
        checks={
          'n_exact':nr*nc==int(spec['n']),
          'full_queen_edge_count_exact':len(full)==int(spec['full_geographic_queen_edges']),
          'role_cut_edge_count_exact':len(cut)==int(spec['role_cut_dgp_edges']),
          'role_counts_exact':vc==expected_counts,
          'full_graph_connected':full_components==1,
          'role_cut_graph_three_components':cut_components==3,
          'full_queen_degree_structure_exact':deg==expected_deg,
          'no_full_self_edges':not (full.source_node.astype(int)==full.target_node.astype(int)).any(),
          'no_cut_self_edges':not (cut.source_node.astype(int)==cut.target_node.astype(int)).any(),
          'no_full_duplicates':not full.duplicated(['source_node','target_node']).any(),
          'no_cut_duplicates':not cut.duplicated(['source_node','target_node']).any(),
        }
        require(all(checks.values()),f'F06_STRUCTURAL_QA_FAILED:{level}:{checks}')
        rows.append({'level':level,'rows':nr,'cols':nc,'n':nr*nc,'full_queen_edges':len(full),'role_cut_dgp_edges':len(cut),'full_graph_components':full_components,'role_cut_graph_components':cut_components,'full_degree_counts':str(deg),**vc,**checks})
    # Explicitly guard the accepted primary layout against extension drift.
    p=next(r for r in rows if r['level']=='primary');require((p['n'],p['full_queen_edges'],p['role_cut_dgp_edges'],p['nuisance_training'],p['support_audit'],p['calibration'],p['test_target'])==(625,2352,2206,250,125,125,125),'F06_PRIMARY_NOT_ACCEPTED_STAGE3B_GEOMETRY')
    return rows
