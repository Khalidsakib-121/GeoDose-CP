from __future__ import annotations
import hashlib, math, time
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import binom
from .io import require, Stage3FError


def target_node_for_case(case) -> int:
    # Frozen normalized location matching D4 node 320: row fraction 1/2, col fraction 5/6.
    r=int(round(0.5*(int(case.n_rows)-1))); c=int(round((5.0/6.0)*(int(case.n_cols)-1)))
    node=r*int(case.n_cols)+c
    return int(node)



def eligible_target_blocks(units: pd.DataFrame, edges: pd.DataFrame, size: int=6) -> dict[int, tuple[int,...]]:
    """Outcome-blind exact-orbit eligibility among frozen final-test slots.

    A target is eligible only if a connected size-`size` block can be formed with
    that one target plus calibration slots, using the frozen structural graph.
    This operationalizes the Stage3A target-population phrase "theorem-eligible
    final-test slots" without looking at A, Y, support diagnostics, or method
    outputs.
    """
    tests=units.loc[units.role.astype(str)=='test_target','node_index'].astype(int).sort_values().tolist()
    out={}
    for node in tests:
        try: out[int(node)]=choose_connected_block(units,edges,int(node),size=size)
        except Stage3FError: pass
    require(len(out)>0,'NO_THEOREM_ELIGIBLE_FINAL_TEST_SLOTS')
    return out

def select_uniform_eligible_target(core, stage3a: dict[str,Any], case, replication: int, eligible: dict[int,tuple[int,...]]) -> tuple[int,tuple[int,...],dict[str,Any]]:
    """Uniform target-slot selection from the pre-outcome eligible set.

    The selector uses a separately derived stream from the frozen target_draw_seed.
    The label is scenario-level so registered common-random factor variants share
    the same rank/quantile of their eligible target set (S4 rho, S6 graph, S9 EO,
    and approximately aligned S10 supports).
    """
    src=core.source_seed_row(stage3a,case.scenario_id,int(replication)); base=int(src['target_draw_seed'])
    seed=int(core.derived_seed(base,f'STAGE3F_TARGET_SLOT_UNIFORM|{case.scenario_id}'))
    u=float(np.random.default_rng(seed).random())
    nodes=sorted(eligible)
    j=min(int(math.floor(u*len(nodes))),len(nodes)-1); node=int(nodes[j])
    return node,eligible[node],{'target_population_rule':'uniform_over_structurally_theorem_eligible_final_test_slots','eligible_target_count':len(nodes),'target_selection_base_seed':base,'target_selection_seed':seed,'target_selection_uniform':u,'target_selection_rank':j,'target_node':node}

def adjacency_from_edges(n:int, edges:pd.DataFrame) -> list[list[int]]:
    a=[set() for _ in range(n)]
    for r in edges.itertuples(index=False):
        u=int(r.source_node); v=int(r.target_node)
        if u==v: continue
        a[u].add(v); a[v].add(u)
    return [sorted(x) for x in a]

def block_is_connected(block:tuple[int,...] | list[int] | np.ndarray, edges:pd.DataFrame) -> bool:
    """Return whether the induced subgraph on ``block`` is connected.

    This is intentionally independent of the accepted D2 helper so target-population
    eligibility can be audited before any downstream exact-orbit module is invoked.
    """
    nodes=[int(x) for x in block]
    if not nodes: return False
    b=set(nodes); adj={u:set() for u in nodes}
    for r in edges.itertuples(index=False):
        u=int(r.source_node); v=int(r.target_node)
        if u in b and v in b and u!=v:
            adj[u].add(v); adj[v].add(u)
    seen=set(); q=[nodes[0]]
    while q:
        u=q.pop(0)
        if u in seen: continue
        seen.add(u); q.extend(sorted(adj[u]-seen))
    return len(seen)==len(b)


def choose_connected_block(units:pd.DataFrame, edges:pd.DataFrame, target_node:int, size:int=6) -> tuple[int,...]:
    """Grow a deterministic connected induced target+calibration block.

    The target population is defined under the frozen *structural/true* graph.  We
    grow from the target one calibration frontier node at a time.  Every accepted
    node therefore has an edge to the already-selected set, which makes connectivity
    an invariant rather than an assumption.  A final induced-subgraph check fails
    closed if future graph/role changes violate that invariant.
    """
    target_node=int(target_node); size=int(size)
    require(size>=2,'CONNECTED_BLOCK_SIZE_TOO_SMALL')
    n=int(units.node_index.max())+1; adj=adjacency_from_edges(n,edges); roles=units.set_index('node_index').role.astype(str).to_dict()
    require(roles.get(target_node)=='test_target',f'FROZEN_TARGET_NOT_TEST_ROLE:{target_node}')
    selected=[target_node]; selected_set={target_node}
    while len(selected)<size:
        frontier=sorted({v for u in selected for v in adj[u] if v not in selected_set and roles.get(v)=='calibration'})
        require(len(frontier)>0,f'INSUFFICIENT_CONNECTED_CALIBRATION_BLOCK:target={target_node}:found={len(selected)-1}')
        v=int(frontier[0]); selected.append(v); selected_set.add(v)
    cal=selected[1:]
    block=tuple(cal+[target_node])
    require(len(set(block))==size,'CONNECTED_BLOCK_DUPLICATE_NODE')
    require(block_is_connected(block,edges),f'CONNECTED_BLOCK_INDUCED_CONNECTIVITY_FAILURE:target={target_node}')
    return block

def component_calibration_count(units:pd.DataFrame,edges:pd.DataFrame,target_node:int)->int:
    n=int(units.node_index.max())+1; adj=adjacency_from_edges(n,edges); seen=set(); q=[int(target_node)]
    while q:
        u=q.pop();
        if u in seen: continue
        seen.add(u); q.extend(v for v in adj[u] if v not in seen)
    role=units.set_index('node_index').role.astype(str).to_dict()
    return int(sum(role.get(v)=='calibration' for v in seen))

def select_target(case, generated:dict[str,Any], target_node:int) -> tuple[pd.Series,bool]:
    units=generated['units']; uid=str(units.loc[units.node_index.astype(int)==int(target_node),'unit_id'].iloc[0])
    if case.primary_target_mode=='observational':
        d=generated['observational_target_draws']; x=d[d.unit_id.astype(str)==uid]
        require(len(x)==1,f'OBS_TARGET_NOT_UNIQUE:{case.case_id}:{uid}')
        row=x.iloc[0].copy(); row['target_dose']=float(row.A_star); row['bandwidth']=np.nan; row['target_scope']='observational_law'; row['log_oracle_treatment_ratio_at_A_star']=0.0; row['target_supported_by_generator']=True; row['draw_status']='drawn'
        return row,True
    d=generated['target_draws']; x=d[(d.unit_id.astype(str)==uid)&np.isclose(d.target_dose.astype(float),float(case.primary_target_dose))]
    require(len(x)==1,f'LOCAL_TARGET_NOT_UNIQUE:{case.case_id}:{uid}:{case.primary_target_dose}')
    return x.iloc[0].copy(),False

def oracle_calibration_logweights(case,generated:dict[str,Any],calibration:pd.DataFrame,observational:bool)->np.ndarray:
    if observational: return np.zeros(len(calibration),float)
    tt=generated['treatment_transport_truth']; dose=float(case.primary_target_dose)
    x=tt[(tt.role.astype(str)=='calibration')&np.isclose(tt.target_dose.astype(float),dose)].set_index('unit_id')
    require(set(calibration.unit_id.astype(str)).issubset(set(x.index.astype(str))),'CALIBRATION_TREATMENT_TRUTH_MISSING')
    return x.loc[calibration.unit_id.astype(str),'log_oracle_treatment_ratio'].to_numpy(float)

def summarize_logweights(logw:np.ndarray)->dict[str,Any]:
    x=np.asarray(logw,float); finite=np.isfinite(x)
    if not finite.any(): return {'positive_weight_count':0,'ess':0.0,'max_normalized_weight':1.0,'log_weight_range':float('inf')}
    v=x[finite]; m=float(v.max()); w=np.exp(v-m); nw=w/w.sum()
    return {'positive_weight_count':int(finite.sum()),'ess':float(1.0/(nw@nw)),'max_normalized_weight':float(nw.max()),'log_weight_range':float(v.max()-v.min())}

def oracle_target_logweight(target:pd.Series,observational:bool)->float:
    return 0.0 if observational else float(target['log_oracle_treatment_ratio_at_A_star'])

def derived_tie_uniform(core,stage3a,case,replication:int,target_node:int,track:str)->tuple[float,int,int]:
    src=core.source_seed_row(stage3a,case.scenario_id,int(replication)); base=int(src['orbit_seed'])
    seed=int(core.derived_seed(base,f'STAGE3F_TIE_UNIFORM|{case.case_id}|{int(target_node)}|{track}'))
    return float(np.random.default_rng(seed).random()),base,seed

def fit_outcome(c3,stage3a_seeds:pd.DataFrame,case,replication:int,units:pd.DataFrame,kind:str):
    source,base=c3.runner.frozen_nuisance_seed(stage3a_seeds,case.scenario_id,int(replication))
    seed=c3.runner.derived_seed(base,'OUTCOME_RF' if kind=='rf' else 'OUTCOME_XGB')
    training=units[units.role.astype(str)=='nuisance_training'].copy()
    model=c3.models.OutcomeRegressor(kind,seed).fit(training)
    return model,{'source_scenario':source,'frozen_nuisance_seed':int(base),'derived_model_seed':int(seed),'fit_id':model.fit_id,'training_data_hash':model.training_data_hash,'n_train':len(training)}

def primary_graph(case,generated,track:str)->pd.DataFrame:
    # Structural theorem track isolates the true graph. Empirical shared-predictor track exercises registered S6 misspecification.
    if track!='ORACLE_STRUCTURAL_OT_TG' and case.scenario_id=='S6': return generated['fitted_edges'][['source_node','target_node']].copy()
    return generated['true_edges'][['source_node','target_node']].copy()

def make_graph_units(units:pd.DataFrame,edges:pd.DataFrame,residual:np.ndarray)->pd.DataFrame:
    u=units.sort_values('node_index').reset_index(drop=True).copy(); n=len(u); deg=np.zeros(n,int)
    for r in edges.itertuples(index=False): deg[int(r.source_node)]+=1; deg[int(r.target_node)]+=1
    u['graph_degree_true']=deg; u['shared_spatial_residual']=np.asarray(residual,float); return u

def _m3m4_candidate(m34,case,units,edges,block,target,candidate_y,factual_y,mean_factual,mean_target,logd,keep_states=False):
    residual=np.asarray(factual_y,float)-np.asarray(mean_factual,float)
    gu=make_graph_units(units,edges,residual)
    law=m34.graph.build_local_graph_law(gu,edges,list(block),float(case.spatial_rho),('gaussian_gmrf' if case.residual_law=='iid_continuous' else case.residual_law),scale=0.35)
    cal=np.asarray([float(np.asarray(factual_y)[int(n)])-float(np.asarray(mean_factual)[int(n)]) for n in block[:-1]],float)
    m3=m34.m3.evaluate_candidate(cal,float(candidate_y),float(mean_target),law,5,target_scale=1.0,tie_tolerance=1e-12,keep_states=keep_states)
    src=m34.m3.candidate_residuals(cal,float(candidate_y),float(mean_target),1.0)
    m4=m34.m4.evaluate_m4(m3.source_marginals,src,np.asarray(logd,float),5,tie_tolerance=1e-12)
    return m3,m4

def m4_source_logd(case,generated,units,block,target,observational:bool)->np.ndarray:
    if observational: return np.zeros(6,float)
    tt=generated['treatment_transport_truth']; dose=float(case.primary_target_dose); by=tt[(tt.role.astype(str)=='calibration')&np.isclose(tt.target_dose.astype(float),dose)].set_index('unit_id')
    lu=units.set_index('node_index'); vals=[]
    for n in block[:-1]: vals.append(float(by.loc[str(lu.loc[int(n),'unit_id']),'log_oracle_treatment_ratio']))
    vals.append(float(target['log_oracle_treatment_ratio_at_A_star']))
    return np.asarray(vals,float)

def interval_components_from_grid(grid:np.ndarray,accept:np.ndarray,lower:float,upper:float)->list[tuple[float,float]]:
    grid=np.asarray(grid,float); acc=np.asarray(accept,bool); require(len(grid)==len(acc),'GRID_ACCEPT_LENGTH')
    if not acc.any(): return []
    step=float(grid[1]-grid[0]) if len(grid)>1 else upper-lower; idx=np.where(acc)[0]; comps=[]; start=prev=int(idx[0])
    for i in idx[1:]:
        i=int(i)
        if i!=prev+1:
            comps.append((max(lower,float(grid[start]-step/2)),min(upper,float(grid[prev]+step/2)))); start=i
        prev=i
    comps.append((max(lower,float(grid[start]-step/2)),min(upper,float(grid[prev]+step/2))))
    return comps

def component_summary(comps:list[tuple[float,float]])->dict[str,Any]:
    if not comps: return {'component_count':0,'raw_total_width':0.0,'hull_lower':np.nan,'hull_upper':np.nan,'hull_width':0.0,'hull_inflation':0.0}
    raw=float(sum(max(0,b-a) for a,b in comps)); lo=float(min(a for a,b in comps)); hi=float(max(b for a,b in comps)); hull=hi-lo
    return {'component_count':len(comps),'raw_total_width':raw,'hull_lower':lo,'hull_upper':hi,'hull_width':hull,'hull_inflation':float(max(0,hull-raw))}

def _structural_return_mask(frame: pd.DataFrame) -> pd.Series:
    target_supported = frame['target_supported_by_generator'].astype(bool) if 'target_supported_by_generator' in frame else pd.Series(True,index=frame.index)
    pop_ok = frame['target_population_eligible'].astype(bool) if 'target_population_eligible' in frame else pd.Series(True,index=frame.index)
    if 'endpoint_requested' in frame and 'endpoint_audited' in frame:
        endpoint_ok = (~frame['endpoint_requested'].astype(bool)) | frame['endpoint_audited'].astype(bool)
    else:
        endpoint_ok = pd.Series(True,index=frame.index)
    return target_supported & pop_ok & endpoint_ok


def choose_thresholds(support:pd.DataFrame, results:pd.DataFrame, contract:dict[str,Any])->tuple[dict[str,Any],pd.DataFrame]:
    """Freeze operational thresholds from the preregistered controlled pilot.

    Only the three candidate grids frozen in the accepted Stage3C threshold plan are
    searched: minimum ESS, maximum normalized weight, and minimum graph-safe count.
    Stage3F does *not* invent or tune a certificate lower-bound grid.  N3 certificate
    non-vacuity is a fixed structural rule (finite lower bound > 0), applied before
    these operational thresholds.

    Selection uses only theorem-certified ORACLE_STRUCTURAL_OT_TG M6 pilot coverage
    and refusal evidence. Width, NSW, empirical RF/XGB outcomes, and production
    outcomes are excluded.
    """
    cfg=contract['threshold_selection']; grid=contract['threshold_candidate_grid']
    require(set(grid)=={'minimum_ess','maximum_normalized_weight','minimum_graph_safe_count'},'UNREGISTERED_THRESHOLD_CANDIDATE_GRID_PRESENT')
    safe=set(cfg['safe_hard_case_ids']); stress=set(cfg['expected_low_information_case_ids'])
    target_cov=float(cfg.get('coverage_target',1.0-float(contract['alpha'])))
    min_case_n=int(cfg['minimum_returned_per_case_for_casewise_coverage_objective'])
    sel=results[(results.track.astype(str)==str(cfg['selection_track'])) & (results.method.astype(str)==str(cfg['selection_method']))].copy()
    require(len(sel)>0,'THRESHOLD_SELECTION_CERTIFIED_ROUTE_EMPTY')
    require(not sel.duplicated(['case_id','replication']).any(),'THRESHOLD_SELECTION_CERTIFIED_ROUTE_DUPLICATED')
    cols=['case_id','replication','raw_covered','raw_method_returned','computational_failure','target_supported']
    require(all(c in sel.columns for c in cols),'THRESHOLD_SELECTION_RESULTS_COLUMNS_MISSING')
    base=support.merge(sel[cols],on=['case_id','replication'],how='left',validate='one_to_one')
    require(base.raw_covered.notna().all(),'THRESHOLD_SELECTION_MISSING_CERTIFIED_RESULTS')
    structural=_structural_return_mask(base)
    # Fixed N3 structural non-vacuity; it is not a pilot-tuned threshold.
    cert=pd.to_numeric(base.certified_coverage_lower_bound,errors='coerce')
    structural &= np.isfinite(cert) & (cert>0.0)
    rows=[]
    penalty=float(cfg.get('insufficient_case_return_penalty',1.0))
    safe_case_count=len(safe)
    for e in grid['minimum_ess']:
      for w in grid['maximum_normalized_weight']:
       for g in grid['minimum_graph_safe_count']:
         ret=(structural & base.raw_method_returned.astype(bool) & ~base.computational_failure.astype(bool)
              & (base.ess>=float(e)) & (base.max_normalized_weight<=float(w))
              & (base.graph_safe_count>=int(g)))
         tmp=base.assign(_ret=ret)
         safe_mask=tmp.case_id.isin(safe); stress_mask=tmp.case_id.isin(stress)
         safe_ret=float(tmp.loc[safe_mask,'_ret'].mean()); stress_ret=float(tmp.loc[stress_mask,'_ret'].mean())
         s3=float(tmp.loc[tmp.case_id=='S3_SHIFT','_ret'].mean())
         strong=float(tmp.loc[tmp.case_id.isin(['S4_RHO060','S4_RHO080']),'_ret'].mean())
         safe_returned=tmp[safe_mask & tmp._ret]
         safe_cov=float(safe_returned.raw_covered.astype(bool).mean()) if len(safe_returned) else np.nan
         pooled_short=float(max(0.0,target_cov-safe_cov)) if np.isfinite(safe_cov) else penalty
         case_short=[]; penalized_case_short=[]; case_cov_n=0; minimum_case_returned=10**9
         for cid,cg in tmp[safe_mask].groupby('case_id',sort=True):
             rr=cg[cg._ret]; nr=len(rr); minimum_case_returned=min(minimum_case_returned,nr)
             if nr>=min_case_n:
                 case_cov_n+=1; cc=float(rr.raw_covered.astype(bool).mean()); sh=max(0.0,target_cov-cc); case_short.append(sh); penalized_case_short.append(sh)
             else:
                 penalized_case_short.append(penalty)
         worst_short=float(max(case_short)) if case_short else penalty
         worst_penalized=float(max(penalized_case_short)) if penalized_case_short else penalty
         rows.append({'minimum_ess':float(e),'maximum_normalized_weight':float(w),'minimum_graph_safe_count':int(g),'safe_retention':safe_ret,'expected_low_information_return_rate':stress_ret,'s3_retention':s3,'strong_s4_retention':strong,'safe_pooled_selective_coverage':safe_cov,'safe_pooled_undercoverage_shortfall':pooled_short,'safe_case_coverage_groups_used':case_cov_n,'safe_case_count':safe_case_count,'minimum_safe_case_returned':(int(minimum_case_returned) if minimum_case_returned<10**9 else 0),'worst_safe_case_undercoverage_shortfall':worst_short,'worst_safe_case_penalized_undercoverage_shortfall':worst_penalized,'certificate_nonvacuity_rule':'finite_lower_bound_strictly_greater_than_0_not_tuned'})
    audit=pd.DataFrame(rows)
    require(len(audit)>0,'THRESHOLD_CANDIDATE_GRID_EMPTY')
    # Frozen pre-pilot lexicographic criterion.  A safe case with fewer than the
    # predeclared minimum returned units receives the maximal validity penalty, so
    # all-refuse/tiny-selective-set solutions cannot win merely by looking perfect.
    ranked=audit.sort_values(
        ['worst_safe_case_penalized_undercoverage_shortfall','safe_pooled_undercoverage_shortfall','safe_case_coverage_groups_used','expected_low_information_return_rate','safe_retention','minimum_ess','maximum_normalized_weight','minimum_graph_safe_count'],
        ascending=[True,True,False,True,False,False,True,False],kind='mergesort')
    x=ranked.iloc[0]
    chosen={k:(int(x[k]) if k=='minimum_graph_safe_count' else float(x[k])) for k in ['minimum_ess','maximum_normalized_weight','minimum_graph_safe_count']}
    chosen.update({'n3_nonvacuous_coverage_lower_bound_rule':'finite_and_strictly_greater_than_0_not_pilot_tuned','safe_retention':float(x.safe_retention),'expected_low_information_return_rate':float(x.expected_low_information_return_rate),'s3_retention':float(x.s3_retention),'strong_s4_retention':float(x.strong_s4_retention),'safe_pooled_selective_coverage':float(x.safe_pooled_selective_coverage) if np.isfinite(x.safe_pooled_selective_coverage) else None,'safe_pooled_undercoverage_shortfall':float(x.safe_pooled_undercoverage_shortfall),'safe_case_coverage_groups_used':int(x.safe_case_coverage_groups_used),'safe_case_count':int(x.safe_case_count),'minimum_safe_case_returned':int(x.minimum_safe_case_returned),'worst_safe_case_undercoverage_shortfall':float(x.worst_safe_case_undercoverage_shortfall),'worst_safe_case_penalized_undercoverage_shortfall':float(x.worst_safe_case_penalized_undercoverage_shortfall),'selection_used_controlled_pilot_outcomes':True,'selection_used_outcomes':True,'selection_used_width':False,'selection_used_nsw':False,'selection_track':str(cfg['selection_track']),'selection_method':str(cfg['selection_method']),'candidate_grid_source':'accepted_Stage3C_only','selection_rule_status':'STAGE3F_FROZEN_LEXICOGRAPHIC_NO_POSTHOC_RETENTION_CUTOFF','status':'FROZEN_AFTER_PREREGISTERED_20REP_PILOT'})
    return chosen,audit


def operational_gate(method:str,row:pd.Series,thresholds:dict[str,Any])->tuple[bool,str]:
    """Apply frozen structural gates first, then pilot-frozen operational gates."""
    m=str(method)
    pop_ok=bool(row.get('target_population_eligible',True))
    if not pop_ok: return False,'R17_TARGET_UNIT_OUTSIDE_SUPPORTED_POPULATION'
    supported=bool(row.get('target_supported',row.get('target_supported_by_generator',True)))
    if not supported: return False,'R01_UNSUPPORTED_DOSE'
    endpoint_requested=bool(row.get('endpoint_requested',False)); endpoint_audited=bool(row.get('endpoint_audited',True))
    if endpoint_requested and not endpoint_audited: return False,'R02_ENDPOINT_NOT_AUDITED'
    if m=='M1': return True,''
    if m in {'M2','M4','M5','M6'}:
        if float(row.get('ess',0.0))<float(thresholds['minimum_ess']): return False,'R03_LOW_EFFECTIVE_CALIBRATION_SIZE'
        if float(row.get('max_normalized_weight',1.0))>float(thresholds['maximum_normalized_weight']): return False,'R04_WEIGHT_CONCENTRATION'
    if m in {'M3','M4','M5','M6'} and int(row.get('graph_safe_count',row.get('track_graph_safe_calibration_count',0)))<int(thresholds['minimum_graph_safe_count']): return False,'R05_GRAPH_COMPONENT_TOO_SMALL'
    if m=='M6':
        # Fixed N3 mathematical non-vacuity rule.  This is intentionally not a tuned
        # fourth threshold: Stage3C froze only ESS, maximum normalized weight, and
        # graph-safe count candidate grids.
        lb=row.get('certified_coverage_lower_bound',np.nan)
        if pd.isna(lb): lb=row.get('operational_coverage_lower_bound',np.nan)
        if not np.isfinite(float(lb)) or float(lb)<=0.0: return False,'R08_CERTIFIED_DEFICIT_VACUOUS'
    return True,''


def aggregate_metrics(results:pd.DataFrame,alpha:float)->pd.DataFrame:
    rows=[]
    keys=['scenario_id','case_id','track','method']
    for key,g in results.groupby(keys,sort=True,dropna=False):
        n=len(g); raw=float(g.raw_covered.astype(bool).mean()); returned=g.operational_returned.astype(bool); nr=int(returned.sum())
        sel=float(g.loc[returned,'raw_covered'].astype(bool).mean()) if nr else np.nan
        service=float((g.raw_covered.astype(bool)&returned).mean())
        lat=pd.to_numeric(g.get('covered_latent',pd.Series(np.nan,index=g.index)),errors='coerce')
        lat_mask=lat.notna(); raw_lat=float(lat[lat_mask].astype(bool).mean()) if lat_mask.any() else np.nan
        lat_ret=returned & lat_mask; sel_lat=float(lat[lat_ret].astype(bool).mean()) if lat_ret.any() else np.nan
        widths=g.loc[returned & np.isfinite(g.width.astype(float)),'width'].astype(float)
        wis=g.loc[returned & np.isfinite(g.wis.astype(float)),'wis'].astype(float)
        nominal=1.0-alpha
        cert=pd.to_numeric(g.get('certified_coverage_lower_bound',pd.Series(np.nan,index=g.index)),errors='coerce')
        cert_min=float(cert.min()) if cert.notna().any() else np.nan
        empirical_under=max(0.0,nominal-sel) if np.isfinite(sel) else np.nan
        certified_deficit=max(0.0,nominal-cert_min) if np.isfinite(cert_min) else np.nan
        deficit_error=abs(empirical_under-certified_deficit) if np.isfinite(empirical_under) and np.isfinite(certified_deficit) else np.nan
        rows.append(dict(zip(keys,key),requested=n,raw_coverage=raw,raw_coverage_mcse=math.sqrt(raw*(1-raw)/n) if n else np.nan,absolute_raw_calibration_error=abs(raw-nominal),returned=nr,refusal_rate=1-nr/n,selective_coverage=sel,selective_coverage_mcse=(math.sqrt(sel*(1-sel)/nr) if nr and np.isfinite(sel) else np.nan),service_level_covered_and_returned=service,observed_product_coverage=sel,latent_outcome_coverage=sel_lat,raw_latent_outcome_coverage=raw_lat,minimum_certified_coverage_lower_bound=cert_min,empirical_undercoverage=max(0.0,nominal-sel) if np.isfinite(sel) else np.nan,certified_deficit=certified_deficit,coverage_loss_diagnostic_error=deficit_error,mean_width=float(widths.mean()) if len(widths) else np.nan,mean_wis=float(wis.mean()) if len(wis) else np.nan,dose_response_rmse=float(np.sqrt(np.mean(np.square(g.target_mean_error.astype(float))))) if 'target_mean_error' in g else np.nan,theorem_eligibility_rate=float(g.theorem_eligible.mean()),mean_ess=float(g.ess.dropna().mean()) if g.ess.notna().any() else np.nan,mean_max_normalized_weight=float(g.max_normalized_weight.dropna().mean()) if g.max_normalized_weight.notna().any() else np.nan,raw_false_support_rate=float(g.raw_false_support.astype(bool).mean()) if 'raw_false_support' in g else np.nan,false_support_rate=float(g.false_support.astype(bool).mean()) if 'false_support' in g else np.nan,computational_failure_rate=float(g.computational_failure.mean())))
    return pd.DataFrame(rows)

def _dose_bin_labels(values:np.ndarray,dose_grid:list[float])->np.ndarray:
    g=np.asarray(dose_grid,float); require(len(g)>=2 and np.all(np.diff(g)>0),'DOSE_GRID_NOT_STRICTLY_INCREASING')
    v=np.asarray(values,float); mids=(g[:-1]+g[1:])/2.0; idx=np.searchsorted(mids,v,side='right'); idx=np.clip(idx,0,len(g)-1)
    return np.asarray([f'D{g[i]:.2f}' for i in idx],dtype=object)


def dose_bin_metrics(results:pd.DataFrame,alpha:float,dose_grid:list[float])->pd.DataFrame:
    x=results.copy(); x['dose_bin']=_dose_bin_labels(x.A_star.to_numpy(float),dose_grid); rows=[]
    keys=['scenario_id','case_id','track','method','dose_bin']
    for key,g in x.groupby(keys,sort=True,dropna=False):
        ret=g.operational_returned.astype(bool); nret=int(ret.sum()); raw=float(g.raw_covered.astype(bool).mean()); sel=float(g.loc[ret,'raw_covered'].astype(bool).mean()) if nret else np.nan
        rows.append(dict(zip(keys,key),requested=len(g),returned=nret,raw_coverage=raw,selective_coverage=sel,absolute_selective_calibration_error=(abs(sel-(1-alpha)) if np.isfinite(sel) else np.nan),refusal_rate=float(1-ret.mean()),raw_false_support_rate=float(g.raw_false_support.astype(bool).mean()) if 'raw_false_support' in g else np.nan,false_support_rate=float(g.false_support.astype(bool).mean()) if 'false_support' in g else np.nan))
    return pd.DataFrame(rows)


def certificate_summary(support:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for key,g in support.groupby(['scenario_id','case_id'],sort=True):
        rows.append({'scenario_id':key[0],'case_id':key[1],'replications':len(g),'target_supported_rate':float(g.target_supported_by_generator.astype(bool).mean()),'minimum_positive_weight_count':int(g.positive_weight_count.min()),'median_ess':float(g.ess.median()),'minimum_ess':float(g.ess.min()),'maximum_normalized_weight_worst':float(g.max_normalized_weight.max()),'minimum_graph_safe_count':int(g.graph_safe_count.min()),'maximum_n2_delta_kl':float(g.n2_delta_kl.max()),'maximum_n2_tv_bound':float(g.n2_tv_bound.max()),'minimum_certified_coverage_lower_bound':float(g.certified_coverage_lower_bound.min())})
    return pd.DataFrame(rows)


def width_diagnostic_summary(width:pd.DataFrame)->pd.DataFrame:
    if len(width)==0: return pd.DataFrame()
    rows=[]
    for key,g in width.groupby(['scenario_id','case_id','track','method','audit_representation'],sort=True,dropna=False):
        if 'raw_method_returned' in g.columns and g.raw_method_returned.notna().any():
            returned=g.raw_method_returned.fillna(False).astype(bool); used=g[returned]
            refusal_rate=float(1.0-returned.mean())
        else:
            used=g; refusal_rate=0.0
        rows.append(dict(zip(['scenario_id','case_id','track','method','audit_representation'],key),audit_queries=len(g),returned_audit_queries=len(used),refusal_rate=refusal_rate,mean_raw_total_width=(float(used.raw_total_width.mean()) if len(used) else np.nan),mean_hull_width=(float(used.hull_width.mean()) if len(used) else np.nan),mean_hull_inflation=(float(used.hull_inflation.mean()) if len(used) else np.nan),mean_hull_wis_observed=(float(used.hull_wis_observed.replace([np.inf,-np.inf],np.nan).mean()) if len(used) else np.nan),claim_status='DIAGNOSTIC_ONLY_NOT_USED_FOR_THRESHOLD_SELECTION_OR_MATCHED_COVERAGE_CONCLUSION'))
    return pd.DataFrame(rows)


def coverage_sanity(results:pd.DataFrame,contract:dict[str,Any])->pd.DataFrame:
    cfg=contract['coverage_sanity']; g=results[(results.track=='ORACLE_STRUCTURAL_OT_TG')&(results.method=='M6')&(results.theorem_eligible)&(results.operational_returned)].copy()
    groups=[]
    for cid,x in g.groupby('case_id'):
        n=len(x)
        if n<int(cfg['minimum_returned_for_test']): continue
        k=int(x.raw_covered.sum()); p0=float(x.certified_coverage_lower_bound.min())
        pval=float(binom.cdf(k,n,p0))
        groups.append({'case_id':cid,'n_returned':n,'covered':k,'empirical_coverage':k/n,'certified_lower_bound_min':p0,'lower_tail_pvalue':pval})
    out=pd.DataFrame(groups)
    if len(out):
        cutoff=float(cfg['familywise_alpha'])/len(out); out['bonferroni_cutoff']=cutoff; out['significant_undercoverage']=out.lower_tail_pvalue<cutoff
    else:
        out['bonferroni_cutoff']=[]; out['significant_undercoverage']=[]
    return out
