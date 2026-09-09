from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .bridges import Stage3BBridge, Stage3CBridge, M3M4Bridge, D2Bridge, Stage3EBridge
from .io import Stage3FError, require, ensure_empty_output_dir, json_dump, deterministic_gzip_csv, sha256_file, verify_zip, verify_source_root
from .pilot_math import (
    eligible_target_blocks, select_uniform_eligible_target, select_target,
    oracle_calibration_logweights, summarize_logweights, oracle_target_logweight,
    derived_tie_uniform, fit_outcome, primary_graph, _m3m4_candidate,
    m4_source_logd, interval_components_from_grid, component_summary,
    choose_thresholds, operational_gate, aggregate_metrics, coverage_sanity,
    component_calibration_count, dose_bin_metrics, certificate_summary, width_diagnostic_summary,
)
from .provenance import source_inventory, output_manifest
from .version import VERSION, STAGE

METHODS=("M1","M2","M3","M4","M5","M6")


def _canonical_edge_hash(edges: pd.DataFrame) -> str:
    a=np.minimum(edges.source_node.to_numpy(int),edges.target_node.to_numpy(int))
    b=np.maximum(edges.source_node.to_numpy(int),edges.target_node.to_numpy(int))
    x=pd.DataFrame({'a':a,'b':b}); x=x[x.a!=x.b].drop_duplicates().sort_values(['a','b'],kind='mergesort')
    return hashlib.sha256('|'.join(f'{r.a}-{r.b}' for r in x.itertuples(index=False)).encode()).hexdigest()


def _read_verification_from_zip(path: Path, suffix: str) -> dict[str,Any]:
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.replace('\\','/').endswith(suffix)]
        require(len(names)==1,f'VERIFICATION_MEMBER_NOT_UNIQUE:{suffix}:{path}')
        return json.loads(z.read(names[0]).decode('utf-8'))


def _verify_inputs(root: Path, paths: dict[str,Path]) -> dict[str,Any]:
    expected=json.loads((root/'configs/expected_upstream_hashes.json').read_text())
    locks=json.loads((root/'configs/source_locks.json').read_text())
    rows={}
    for key in ['stage3a_output_zip','stage3b_output_zip','stage3d_d4_output_zip','stage3e_output_zip']:
        rows[key]=verify_zip(paths[key],expected[key])
    for key in ['stage3b_source_root','stage3c_source_root','stage3d_d2_source_root','stage3d_d4_source_root','stage3e_source_root']:
        rows[key]=verify_source_root(paths[key],locks[key])
    d4v=_read_verification_from_zip(paths['stage3d_d4_output_zip'],'STAGE3D_D4_VERIFICATION.json')
    e3v=_read_verification_from_zip(paths['stage3e_output_zip'],'STAGE3E_VERIFICATION.json')
    require(d4v.get('status')=='verified_complete' and int(d4v.get('failed_checks',0))==0,'ACCEPTED_D4_VERIFICATION_NOT_COMPLETE')
    require(e3v.get('status')=='verified_complete' and int(e3v.get('failed_checks',0))==0,'ACCEPTED_STAGE3E_VERIFICATION_NOT_COMPLETE')
    rows['accepted_d4_verification']={'status':'verified_complete','checks':d4v.get('checks_passed',d4v.get('passed_checks'))}
    rows['accepted_stage3e_verification']={'status':'verified_complete','checks':e3v.get('checks_passed',e3v.get('passed_checks'))}
    return rows


def _case_pilot_alignment(b3: Stage3BBridge, contract: dict[str,Any]) -> dict[str,Any]:
    ids=list(contract['pilot_case_ids']); require(len(ids)==20 and len(set(ids))==20,'PILOT_CASE_MATRIX_MUST_HAVE_20_UNIQUE_CASE_VARIANTS')
    require(all(cid in b3.cases for cid in ids),'PILOT_CASE_NOT_IN_FROZEN_STAGE3B')
    scenarios={b3.cases[c].scenario_id for c in ids}; require(scenarios=={f'S{i}' for i in range(1,11)},'PILOT_DOES_NOT_COVER_EXACTLY_S1_S10')
    seeds=b3.stage3a['frames']['seed_registry.csv']
    pilot=seeds[seeds.phase.astype(str)=='pilot']
    counts=pilot.groupby('scenario_id').replication.nunique().to_dict()
    require(all(counts.get(f'S{i}',0)==20 for i in range(1,11)),'FROZEN_20REP_SEED_REGISTRY_INCOMPLETE')
    require(int(contract['replications'])==20,'CONTRACT_REPLICATION_COUNT_NOT_20')
    # Stage3A itself names this threshold/target-population freeze stage Stage3F.
    tpc=b3.stage3a['target_population_contract']
    require(str(tpc.get('target_population_freeze_stage','')).lower().find('stage3f')>=0,'STAGE3A_TARGET_POPULATION_EXPECTS_STAGE3F')
    return {'pilot_case_variant_count':20,'scenario_count':10,'replications_per_scenario':20,'frozen_pilot_seed_rows':len(pilot),'target_population_contract':tpc['controlled_target_population'],'target_population_freeze_stage':tpc['target_population_freeze_stage']}



def _threshold_plan_alignment(c3: Stage3CBridge, contract: dict[str,Any]) -> dict[str,Any]:
    plan=c3.threshold_plan; frozen=plan['candidate_grids_frozen_before_pilot']; grid=contract['threshold_candidate_grid']
    require([float(x) for x in frozen['minimum_ess']]==[float(x) for x in grid['minimum_ess']],'STAGE3C_MINIMUM_ESS_GRID_DRIFT')
    require([float(x) for x in frozen['maximum_normalized_weight']]==[float(x) for x in grid['maximum_normalized_weight']],'STAGE3C_MAX_WEIGHT_GRID_DRIFT')
    require([int(x) for x in frozen['minimum_graph_safe_count']]==[int(x) for x in grid['minimum_graph_safe_count']],'STAGE3C_GRAPH_SAFE_GRID_DRIFT')
    require('coverage/refusal criterion' in str(plan['selection_rule']),'STAGE3C_SELECTION_RULE_DRIFT')
    require(set(grid)=={'minimum_ess','maximum_normalized_weight','minimum_graph_safe_count'},'STAGE3F_UNREGISTERED_THRESHOLD_GRID')
    require(contract['threshold_grid_provenance'].get('additional_candidate_grids_added_by_stage3f') is False,'STAGE3F_MUST_NOT_ADD_THRESHOLD_GRID')
    return {'stage3c_threshold_plan_status':plan['status'],'stage3c_selection_rule':plan['selection_rule'],'minimum_ess':frozen['minimum_ess'],'maximum_normalized_weight':frozen['maximum_normalized_weight'],'minimum_graph_safe_count':frozen['minimum_graph_safe_count'],'additional_candidate_grids_added_by_stage3f':False,'n3_nonvacuity_rule':'finite_coverage_lower_bound_strictly_greater_than_0_not_tuned'}


def _oracle_mean(case, units: pd.DataFrame, target: pd.Series) -> tuple[np.ndarray,float]:
    return units.sort_values('node_index').conditional_mean_at_A.to_numpy(float),float(target.conditional_mean_at_A_star)


def _model_mean(model, units: pd.DataFrame, target_node: int, astar: float) -> tuple[np.ndarray,float]:
    u=units.sort_values('node_index').reset_index(drop=True)
    factual=np.asarray(model.predict_factual(u),float)
    row=u[u.node_index.astype(int)==int(target_node)].copy()
    require(len(row)==1,'TARGET_UNIT_NOT_UNIQUE_FOR_MODEL')
    mt=float(model.predict_target(row,np.asarray([float(astar)]))[0])
    return factual,mt


def _interval_one(c3, center: float, scores: np.ndarray, logw: np.ndarray, target_logw: float, alpha: float):
    scores=np.asarray(scores,float); logw=np.asarray(logw,float); require(len(scores)==len(logw),'INTERVAL_SCORE_WEIGHT_LENGTH')
    order=np.argsort(scores,kind='mergesort')
    return c3.methods.build_interval(float(center),scores[order],logw[order],float(target_logw),float(alpha))


def _finite_covered(lo: float, hi: float, y: float) -> bool:
    if np.isnan(lo) or np.isnan(hi) or not np.isfinite(y): return False
    return bool(float(y)>=float(lo) and float(y)<=float(hi))


def _wis(c3,lo:float,hi:float,y:float,alpha:float)->float:
    try: return float(c3.weights.interval_score(float(lo),float(hi),float(y),float(alpha)))
    except Exception: return float('inf')


def _theorem_eligibility(case, method:str, track:str, observational:bool, target_supported:bool, safe_count:int)->tuple[bool,str]:
    if not target_supported: return False,'target_not_supported_by_generator'
    if track!='ORACLE_STRUCTURAL_OT_TG': return False,'empirical_nuisance_route_not_finite_sample_certified'
    iid_res=str(case.residual_law)=='iid_continuous'; iid_cov=str(case.covariate_law)=='iid'
    if method=='M1':
        ok=iid_res and iid_cov and observational and str(case.target_shift)=='none'
        return ok,'exchangeable_observational_reduction' if ok else 'exchangeability_not_established'
    if method=='M2':
        ok=iid_res and iid_cov
        return ok,'oracle_treatment_iid_reduction' if ok else 'spatial_dependence_or_covariate_dependence_present'
    if method=='M3':
        ok=observational
        return ok,'observational_target_spatial_law' if ok else 'treatment_transport_absent'
    if method=='M4':
        ok=observational or (case.scenario_id in {'S1','S3'} and iid_res and iid_cov)
        return ok,'registered_factorizing_reduction_only' if ok else 'naive_product_not_certified'
    if method=='M5':
        ok=iid_res and iid_cov and int(safe_count)>0
        return ok,'graph_safe_iid_oracle_treatment_reduction' if ok else 'graph_safe_finite_sample_reduction_not_established'
    if method=='M6': return True,'stage3e_structural_oracle_N2_N3_certified_route'
    return False,'unknown_method'


def _patch_treatment_units(c3, units:pd.DataFrame, model)->tuple[pd.DataFrame,dict[str,float]]:
    u=units.sort_values('node_index').reset_index(drop=True).copy(); probs,a,b=model.components(u)
    require(probs.shape==(len(u),3) and np.all(np.isfinite(probs)) and np.all(probs>=0),'PROPENSITY_COMPONENT_INVALID')
    err=float(np.max(np.abs(probs.sum(axis=1)-1.0))); require(err<=2e-10,'PROPENSITY_CATEGORY_NORMALIZATION_FAILURE')
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(a>0) and np.all(b>0),'PROPENSITY_BETA_INVALID')
    u['pi_atom_0']=probs[:,0];u['pi_atom_1']=probs[:,1];u['pi_interior']=probs[:,2];u['beta_alpha']=a;u['beta_beta']=b;u['oracle_g_mixed_at_A']=model.density(u)
    return u,{'max_category_sum_abs_error':err,'minimum_beta_alpha':float(np.min(a)),'minimum_beta_beta':float(np.min(b))}


def _estimated_logweights(c3, model, case, units:pd.DataFrame, calibration:pd.DataFrame, target:pd.Series, observational:bool)->tuple[np.ndarray,float]:
    if observational: return np.zeros(len(calibration)),0.0
    dose=float(case.primary_target_dose); bw=None if dose in (0.0,1.0) else float(case.primary_bandwidth)
    audited=bool(units.loc[units.node_index.astype(int)==int(target['_target_node']),'endpoint_audited'].iloc[0])
    q=c3.weights.intervention_density(calibration.A.to_numpy(float),dose,bw,audited); g=model.density(calibration)
    logs=np.full(len(calibration),-np.inf,float); ok=(q>0)&(g>0)&np.isfinite(g); logs[ok]=np.log(q[ok])-np.log(g[ok])
    tu=units[units.node_index.astype(int)==int(target['_target_node'])].copy(); ast=np.asarray([float(target.A_star)])
    qt=float(c3.weights.intervention_density(ast,dose,bw,audited)[0]); gt=float(model.density(tu,a=ast)[0]); tl=float(np.log(qt)-np.log(gt)) if qt>0 and gt>0 and np.isfinite(gt) else -np.inf
    require(not np.isnan(logs).any() and not np.isposinf(logs).any(),'ESTIMATED_LOGWEIGHT_INVALID')
    return logs,tl


def _source_logd_model(c3, model, case, units:pd.DataFrame, block:tuple[int,...], target:pd.Series, observational:bool)->np.ndarray:
    if observational: return np.zeros(6,float)
    lu=units.set_index('node_index',drop=False); rows=[]; av=[]
    for n in block[:-1]: rows.append(lu.loc[[int(n)]].copy()); av.append(float(lu.loc[int(n),'A']))
    rows.append(lu.loc[[int(block[-1])]].copy()); av.append(float(target.A_star))
    base=pd.concat(rows,ignore_index=True); a=np.asarray(av,float); dose=float(case.primary_target_dose); bw=None if dose in (0.,1.) else float(case.primary_bandwidth); audited=bool(lu.loc[int(block[-1]),'endpoint_audited'])
    q=c3.weights.intervention_density(a,dose,bw,audited); g=model.density(base,a=a); out=np.full(6,-np.inf,float); ok=(q>0)&(g>0)&np.isfinite(g);out[ok]=np.log(q[ok])-np.log(g[ok]);return out


def _sparse_cache_key(case,units,edges,m): return (int(case.n_rows),int(case.n_cols),str(case.sample_design),round(float(case.spatial_rho),12),_canonical_edge_hash(edges),int(m))


def _get_sparse(d2:D2Bridge,e3:Stage3EBridge,case,units,edges,m:int,cache:dict):
    key=_sparse_cache_key(case,units,edges,m)
    if key not in cache:
        n=len(units); Q=d2.gp.normalized_precision(n,edges,float(case.spatial_rho),0.35); ids=units.sort_values('node_index').node_index.to_numpy(int)
        Qs,kl,ordering=e3.sparse_precision(Q,units,ids,int(m),mineig=1e-10)
        cache[key]=(Q,Qs,float(kl),ordering)
    return cache[key]


def _evaluate_track(*, b3,c3,m34,d2,e3,case,generated,replication,target_node,block,target,observational,track,model,contract,sparse_cache,width_audit:bool):
    alpha=float(contract['alpha']); units=generated['units'].sort_values('node_index').reset_index(drop=True); true_edges=generated['true_edges'][['source_node','target_node']].copy(); edges=primary_graph(case,generated,track)
    cal=units[units.role.astype(str)=='calibration'].sort_values('node_index').copy(); cal_nodes=cal.node_index.to_numpy(int)
    if track=='ORACLE_STRUCTURAL_OT_TG':
        factual_y=units.Y_true_at_A.to_numpy(float); target_y=float(target.Y_true_at_A_star); target_y_latent=target_y; mean_factual,mean_target=_oracle_mean(case,units,target); response='latent_Y_true'; use_observed=False
    else:
        factual_y=units.Y_observed_at_A.to_numpy(float); target_y=float(target.Y_observed_at_A_star); target_y_latent=float(target.Y_true_at_A_star); mean_factual,mean_target=_model_mean(model,units,target_node,float(target.A_star)); response='observed_EO_product'; use_observed=True
    scores=np.abs(factual_y[cal_nodes]-mean_factual[cal_nodes])
    logw=oracle_calibration_logweights(case,generated,cal,observational); target_logw=oracle_target_logweight(target,observational)
    wsum=summarize_logweights(logw)
    cache=c3.graph.build_graph_cache(units,edges,np.asarray([target_node],int)); safe_nodes=c3.graph.deterministic_graph_safe_set(cache,target_node,cal_nodes); safe_pos={int(n):i for i,n in enumerate(cal_nodes)}; safe_idx=np.asarray([safe_pos[int(n)] for n in safe_nodes],int)
    tie,orbit_base,tie_seed=derived_tie_uniform(b3.core,b3.stage3a,case,replication,target_node,track)
    block_connected=bool(d2.gp.block_is_connected(np.asarray(block,dtype=int),edges))
    # The target population/block is frozen using the structural graph.  A registered
    # fitted-graph diagnostic (S6) can legitimately disconnect that same block.  That
    # is a scientific working-graph refusal, not a computational exception and not a
    # reason to retarget or change the frozen block.
    logd6=m4_source_logd(case,generated,units,block,target,observational)
    m3=m4=m6=ev=None
    if block_connected:
        # M3/M4 share one accepted-vendor spatial-orbit evaluation at realized truth.
        m3,m4=_m3m4_candidate(m34,case,units,edges,block,target,target_y,factual_y,mean_factual,mean_target,logd6,keep_states=False)
    # N2 is still a full-graph diagnostic even when the exact local fitted-graph block
    # is disconnected.  M6 exact candidate evaluation is refused in that case.
    Qfull,Qs,kl,_=_get_sparse(d2,e3,case,units,edges,int(contract['primary_scalable_neighborhood_size']),sparse_cache)
    sparse_boundary=np.asarray([],dtype=int)
    if block_connected:
        prepared=d2.make_fixture(case,units,edges,block,target,observational_target=observational,use_observed_outcomes=use_observed)
        sparse_boundary=e3.precision_boundary(Qs,np.asarray(prepared.fixture.block_nodes,dtype=int))
        prepared=d2.patch_sparse_precision(prepared,Qs,sparse_boundary)
        if model is not None: prepared=d2.patch_outcome_model(prepared,model,units)
        ev=d2.evaluator(prepared,tie); m6=ev.evaluate(float(target_y))
    m3_lat=m4_lat=m6_lat=None
    if block_connected and case.scenario_id=='S9' and track!='ORACLE_STRUCTURAL_OT_TG':
        m3_lat,m4_lat=_m3m4_candidate(m34,case,units,edges,block,target,target_y_latent,factual_y,mean_factual,mean_target,logd6,keep_states=False)
        m6_lat=ev.evaluate(float(target_y_latent))
    graph_count=component_calibration_count(units,edges,target_node)
    rows=[]
    sparse_tv=float(math.sqrt(max(0.0,kl)/2.0)); sparse_lb=float(max(0.0,1-alpha-sparse_tv)); endpoint_requested=bool((not observational) and float(case.primary_target_dose) in (0.0,1.0)); endpoint_audited=bool(units.loc[units.node_index.astype(int)==int(target_node),'endpoint_audited'].iloc[0])
    requested_dose=float(target.A_star if observational else case.primary_target_dose)
    bandwidth_raw=target.get('bandwidth',np.nan); intervention_bandwidth=(float(bandwidth_raw) if pd.notna(bandwidth_raw) else np.nan)
    target_supported=bool(target.get('target_supported_by_generator',True))
    common={'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':int(replication),'track':track,'response_target':response,'target_node':int(target_node),'target_grid_row':int(units.loc[units.node_index==target_node,'grid_row'].iloc[0]),'target_grid_col':int(units.loc[units.node_index==target_node,'grid_col'].iloc[0]),'n_rows':int(case.n_rows),'n_cols':int(case.n_cols),'requested_dose':requested_dose,'target_dose':requested_dose,'intervention_bandwidth':intervention_bandwidth,'A_star':float(target.A_star),'target_supported':target_supported,'positivity_status':('SUPPORTED' if target_supported else 'UNSUPPORTED'),'target_population_eligible':True,'endpoint_requested':endpoint_requested,'endpoint_audited':endpoint_audited,'exact_block_nodes':'|'.join(map(str,block)),'spatial_block_rule':'uniform_structurally_eligible_connected_size6_target_last','sparse_boundary_size':int(len(sparse_boundary)),'spatial_boundary_rule':'accepted_stage3e_precision_boundary_from_sparse_Q_m64','track_graph_edge_hash':_canonical_edge_hash(edges),'track_graph_component_calibration_count':int(graph_count),'track_graph_safe_calibration_count':int(len(safe_nodes)),'track_graph_exact_block_connected':bool(block_connected),'treatment_support_ess':float(wsum['ess']),'treatment_support_max_normalized_weight':float(wsum['max_normalized_weight']),'treatment_positive_weight_count':int(wsum['positive_weight_count']),'n2_delta_kl':float(kl),'delta_sparse':sparse_tv,'delta_mis':(0.0 if track=='ORACLE_STRUCTURAL_OT_TG' else np.nan),'delta_est':(0.0 if track=='ORACLE_STRUCTURAL_OT_TG' else np.nan),'delta_comp':0.0,'certificate_failure_probability_eta':(0.0 if track=='ORACLE_STRUCTURAL_OT_TG' else np.nan),'operational_coverage_lower_bound':sparse_lb,'certified_coverage_lower_bound':(sparse_lb if track=='ORACLE_STRUCTURAL_OT_TG' else np.nan),'certificate_theorem_backed':bool(track=='ORACLE_STRUCTURAL_OT_TG'),'nuisance_certificate_status':('THEOREM_BACKED_ORACLE_STRUCTURAL' if track=='ORACLE_STRUCTURAL_OT_TG' else 'EMPIRICAL_DIAGNOSTIC_NOT_CERTIFIED'),'target_center_estimate':float(mean_target),'oracle_target_mean_truth':float(target.conditional_mean_at_A_star),'target_mean_error':float(mean_target-float(target.conditional_mean_at_A_star)),'tie_uniform':float(tie),'orbit_seed_base':int(orbit_base),'tie_seed':int(tie_seed)}
    # M1/M2/M5 exact interval baselines.
    intervals={
      'M1':_interval_one(c3,mean_target,scores,np.zeros(len(scores)),0.0,alpha),
      'M2':_interval_one(c3,mean_target,scores,logw,target_logw,alpha),
      'M5':_interval_one(c3,mean_target,scores[safe_idx],logw[safe_idx],target_logw,alpha),
    }
    for method in METHODS:
        safe_count=len(safe_nodes); elig,reason=_theorem_eligibility(case,method,track,observational,bool(common['target_supported']),safe_count)
        branch={'M1':'baseline_split_conformal','M2':'dose_weighted_split_conformal','M3':'exact_local_spatial_orbit','M4':'exact_local_naive_product','M5':'graph_safe_split_conformal','M6':'stage3e_sparse_m64_plus_exact_local_orbit'}[method]
        base=dict(common,method=method,method_branch=branch,theorem_eligible=bool(elig),theorem_eligibility_reason=reason,raw_method_returned=True,raw_refusal_code='',computational_failure=False,width=np.nan,wis=np.nan,lower=np.nan,upper=np.nan,pvalue_conservative=np.nan,pvalue_randomized=np.nan,raw_covered=False,covered_latent=np.nan,ess=(float(wsum['ess']) if method in {'M2','M4','M6'} else np.nan),max_normalized_weight=(float(wsum['max_normalized_weight']) if method in {'M2','M4','M6'} else np.nan),positive_weight_count=(int(wsum['positive_weight_count']) if method in {'M2','M4','M6'} else 0),graph_safe_count=(int(len(safe_nodes)) if method in {'M3','M4','M5','M6'} else 0))
        if method in intervals:
            r=intervals[method]; rawret=str(r.interval_status)!='refused'; base.update(raw_method_returned=rawret,raw_refusal_code=str(r.refusal_code),lower=float(r.lower),upper=float(r.upper),width=(float(r.upper-r.lower) if np.isfinite(r.lower) and np.isfinite(r.upper) else float('inf') if rawret else np.nan),raw_covered=(_finite_covered(r.lower,r.upper,target_y) if rawret else False),covered_latent=(_finite_covered(r.lower,r.upper,target_y_latent) if rawret else False),wis=(_wis(c3,r.lower,r.upper,target_y,alpha) if rawret else np.nan),ess=float(r.ess),max_normalized_weight=float(r.max_normalized_weight),positive_weight_count=int(r.positive_weight_count))
        elif method in {'M3','M4','M6'} and not block_connected:
            base.update(raw_method_returned=False,raw_refusal_code='R05_GRAPH_COMPONENT_TOO_SMALL',raw_covered=False,covered_latent=False)
        elif method=='M3':
            base.update(pvalue_conservative=float(m3.pvalue),raw_covered=bool(m3.pvalue>alpha),covered_latent=(bool(m3_lat.pvalue>alpha) if m3_lat is not None else bool(m3.pvalue>alpha)))
        elif method=='M4':
            base.update(pvalue_conservative=float(m4.pvalue),raw_covered=bool(m4.pvalue>alpha),covered_latent=(bool(m4_lat.pvalue>alpha) if m4_lat is not None else bool(m4.pvalue>alpha)))
        else:
            base.update(pvalue_conservative=float(m6.conservative_p),pvalue_randomized=float(m6.randomized_p),raw_covered=bool(m6.conservative_accept),covered_latent=(bool(m6_lat.conservative_accept) if m6_lat is not None else bool(m6.conservative_accept)),m6_distinct_states=int(m6.distinct_states))
        rows.append(base)
    width_rows=[]
    if width_audit:
        lo,hi=map(float,contract['candidate_domain']); grid=np.linspace(lo,hi,int(contract['width_audit_grid_points']))
        # M1/M2/M5 exact interval width audit.
        for method in ['M1','M2','M5']:
            r=intervals[method]; comps=[] if str(r.interval_status)=='refused' else [(float(r.lower),float(r.upper))]
            s=component_summary(comps); width_rows.append({**{k:common[k] for k in ['scenario_id','case_id','replication','track','target_node']},'method':method,'audit_representation':'exact_interval','raw_method_returned':bool(str(r.interval_status)!='refused'),'raw_refusal_code':str(r.refusal_code),'candidate_domain_lower':lo,'candidate_domain_upper':hi,**s,'hull_wis_observed':_wis(c3,s['hull_lower'],s['hull_upper'],target_y,alpha) if s['component_count'] else np.nan,'hull_wis_latent':_wis(c3,s['hull_lower'],s['hull_upper'],target_y_latent,alpha) if s['component_count'] else np.nan})
        a3=[];a4=[];a6=[]
        if block_connected:
            for y in grid:
                z3,z4=_m3m4_candidate(m34,case,units,edges,block,target,float(y),factual_y,mean_factual,mean_target,logd6,keep_states=False); z6=ev.evaluate(float(y)); a3.append(z3.pvalue>alpha);a4.append(z4.pvalue>alpha);a6.append(z6.conservative_accept)
        else:
            a3=[False]*len(grid); a4=[False]*len(grid); a6=[False]*len(grid)
        for method,acc in [('M3',a3),('M4',a4),('M6',a6)]:
            comps=interval_components_from_grid(grid,np.asarray(acc,bool),lo,hi); s=component_summary(comps)
            width_rows.append({**{k:common[k] for k in ['scenario_id','case_id','replication','track','target_node']},'method':method,'audit_representation':'diagnostic_grid_not_coverage_preserving','grid_points':len(grid),'candidate_domain_lower':lo,'candidate_domain_upper':hi,'raw_method_returned':bool(block_connected),'raw_refusal_code':('' if block_connected else 'R05_GRAPH_COMPONENT_TOO_SMALL'),**s,'hull_wis_observed':_wis(c3,s['hull_lower'],s['hull_upper'],target_y,alpha) if s['component_count'] else np.nan,'hull_wis_latent':_wis(c3,s['hull_lower'],s['hull_upper'],target_y_latent,alpha) if s['component_count'] else np.nan})
    return rows,width_rows,{'track_graph_edges':len(edges),'safe_calibration_count':len(safe_nodes),'graph_safe_count':len(safe_nodes),'n2_delta_kl':float(kl),'sparse_precision_nnz':int(Qs.nnz),'sparse_precision_boundary_size':int(len(sparse_boundary)),'exact_block_connected_under_track_graph':bool(block_connected),'exact_block_refusal_code':('' if block_connected else 'R05_GRAPH_COMPONENT_TOO_SMALL')}


def _fit_propensity(c3, seeds, case, replication:int, units:pd.DataFrame, mode:str):
    source,base=c3.runner.frozen_nuisance_seed(seeds,case.scenario_id,int(replication)); label='MIXED_PROPENSITY_ESTIMATED' if mode=='estimated' else 'MIXED_PROPENSITY_MISSPECIFIED'; seed=c3.runner.derived_seed(base,label)
    tr=units[units.role.astype(str)=='nuisance_training'].copy(); require(len(tr)>0,'NO_NUISANCE_TRAINING_FOR_PROPENSITY'); model=c3.models.MixedPropensityModel(mode,seed).fit(tr)
    audit=model.normalization_audit(tr); err=float(np.max(np.abs(audit.total_mixed_mass.to_numpy(float)-1.0)))
    return model,{'source_scenario':source,'frozen_nuisance_seed':int(base),'derived_propensity_seed':int(seed),'mode':mode,'fit_id':model.fit_id,'training_data_hash':model.training_data_hash,'n_train':len(tr),'max_training_mixed_mass_abs_error':err}


def _nuisance_diagnostics(*,b3,c3,m34,d2,e3,case,generated,replication,target_node,block,target,observational,model_rf,contract,sparse_cache):
    if int(replication) not in set(map(int,contract['nuisance_estimation_diagnostic_reps'])): return [],[]
    if str(contract['nuisance_estimation_case_by_scenario'][case.scenario_id])!=case.case_id: return [],[]
    units=generated['units'].sort_values('node_index').reset_index(drop=True); cal=units[units.role.astype(str)=='calibration'].sort_values('node_index').copy(); true_edges=generated['true_edges'][['source_node','target_node']].copy(); alpha=float(contract['alpha'])
    oracle_lw=oracle_calibration_logweights(case,generated,cal,observational); oracle_tlw=oracle_target_logweight(target,observational); y=float(target.Y_observed_at_A_star); meanf,meant=_model_mean(model_rf,units,target_node,float(target.A_star)); scores=np.abs(units.Y_observed_at_A.to_numpy(float)[cal.node_index.to_numpy(int)]-meanf[cal.node_index.to_numpy(int)])
    _,Qs,kl,_=_get_sparse(d2,e3,case,units,true_edges,int(contract['primary_scalable_neighborhood_size']),sparse_cache); tie,*_=derived_tie_uniform(b3.core,b3.stage3a,case,replication,target_node,'RF_PROPENSITY_DIAGNOSTIC')
    rows=[];fits=[]
    for mode in ['estimated','misspecified']:
        pm,fa=_fit_propensity(c3,b3.stage3a['frames']['seed_registry.csv'],case,replication,units,mode); patched,pdiag=_patch_treatment_units(c3,units,pm); lw,tlw=_estimated_logweights(c3,pm,case,units,cal,target,observational); estsum=summarize_logweights(lw); finite=np.isfinite(lw)&np.isfinite(oracle_lw); diff=(lw[finite]-oracle_lw[finite]) if finite.any() else np.asarray([],float)
        safe_cache=c3.graph.build_graph_cache(units,true_edges,np.asarray([target_node])); safe=c3.graph.deterministic_graph_safe_set(safe_cache,target_node,cal.node_index.to_numpy(int)); by={int(n):i for i,n in enumerate(cal.node_index.to_numpy(int))}; si=np.asarray([by[int(n)] for n in safe],int)
        i2=_interval_one(c3,meant,scores,lw,tlw,alpha); i5=_interval_one(c3,meant,scores[si],lw[si],tlw,alpha)
        logd6=_source_logd_model(c3,pm,case,units,block,target,observational)
        # M4 estimated-treatment diagnostic reuses the accepted D4-vendored M3/M4 engine:
        # the spatial source marginal is computed from the shared RF residuals and only d_j
        # is replaced by the fitted treatment ratio.  No private M4 implementation is added.
        edges=true_edges; factual=units.Y_observed_at_A.to_numpy(float)
        _,z4=_m3m4_candidate(m34,case,units,edges,block,target,y,factual,meanf,meant,logd6,keep_states=False)
        prepared=d2.make_fixture(case,patched,edges,block,target,observational_target=observational,use_observed_outcomes=True)
        sparse_boundary=e3.precision_boundary(Qs,np.asarray(prepared.fixture.block_nodes,dtype=int))
        prepared=d2.patch_sparse_precision(prepared,Qs,sparse_boundary); prepared=d2.patch_outcome_model(prepared,model_rf,patched); z=d2.evaluator(prepared,tie).evaluate(y)
        for method,r in [('M2',i2),('M5',i5)]:
            rows.append({'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':int(replication),'propensity_mode':mode,'method':method,'pvalue_conservative':np.nan,'covered_observed':_finite_covered(r.lower,r.upper,y) if r.interval_status!='refused' else False,'raw_method_returned':r.interval_status!='refused','width':float(r.upper-r.lower) if np.isfinite(r.lower) and np.isfinite(r.upper) else np.inf if r.interval_status!='refused' else np.nan,'ess':float(estsum['ess']),'max_normalized_weight':float(estsum['max_normalized_weight']),'theorem_certified':False})
        rows.append({'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':int(replication),'propensity_mode':mode,'method':'M4','pvalue_conservative':float(z4.pvalue),'covered_observed':bool(z4.pvalue>alpha),'raw_method_returned':True,'width':np.nan,'ess':float(estsum['ess']),'max_normalized_weight':float(estsum['max_normalized_weight']),'theorem_certified':False})
        rows.append({'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':int(replication),'propensity_mode':mode,'method':'M6','pvalue_conservative':float(z.conservative_p),'covered_observed':bool(z.conservative_accept),'raw_method_returned':True,'width':np.nan,'ess':float(estsum['ess']),'max_normalized_weight':float(estsum['max_normalized_weight']),'theorem_certified':False})
        fits.append({**fa,**pdiag,'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':int(replication),'oracle_finite_overlap_count':int(finite.sum()),'calibration_logratio_rmse_vs_oracle':float(np.sqrt(np.mean(diff*diff))) if len(diff) else np.nan,'calibration_logratio_max_abs_vs_oracle':float(np.max(np.abs(diff))) if len(diff) else np.nan,'estimated_ess':float(estsum['ess']),'estimated_max_normalized_weight':float(estsum['max_normalized_weight']),'target_logratio_abs_error_vs_oracle':float(abs(tlw-oracle_tlw)) if np.isfinite(tlw) and np.isfinite(oracle_tlw) else np.nan,'m6_n2_delta_kl':float(kl),'finite_sample_theorem_certified':False})
    return rows,fits


def _local_metrics(results:pd.DataFrame,alpha:float)->tuple[pd.DataFrame,pd.DataFrame]:
    x=results.copy(); frac=x.target_grid_row.astype(float)/np.maximum(x.n_rows.astype(float)-1.0,1.0)
    x['target_row_band']=np.where(frac<1.0/3.0,'low_row',np.where(frac<2.0/3.0,'middle_row','high_row'))
    rows=[]
    for key,g in x.groupby(['scenario_id','case_id','track','method','target_row_band'],dropna=False,sort=True):
        ret=g.operational_returned.astype(bool); n=int(ret.sum()); cov=float(g.loc[ret,'raw_covered'].mean()) if n else np.nan
        rows.append(dict(zip(['scenario_id','case_id','track','method','target_row_band'],key),requested=len(g),returned=n,selective_coverage=cov,mcse=(math.sqrt(cov*(1-cov)/n) if n and np.isfinite(cov) else np.nan)))
    loc=pd.DataFrame(rows); summ=[]
    for key,g in loc.groupby(['scenario_id','case_id','track','method'],dropna=False,sort=True):
        vals=g.selective_coverage.dropna().to_numpy(float); summ.append(dict(zip(['scenario_id','case_id','track','method'],key),local_strata_with_returned=len(vals),fifth_percentile_local_coverage=float(np.quantile(vals,.05)) if len(vals) else np.nan,minimum_local_coverage=float(np.min(vals)) if len(vals) else np.nan))
    return loc,pd.DataFrame(summ)


def run(root: Path, paths: dict[str,Path], output_dir: Path, overwrite: bool=False) -> dict[str,Any]:
    root=Path(root); output_dir=Path(output_dir); ensure_empty_output_dir(output_dir,overwrite); t0=time.perf_counter()
    contract=yaml.safe_load((root/'configs/pilot_contract.yaml').read_text()); require(contract['stage']=='Stage3F_Pilot_S1_S10_20Rep_Threshold_Freeze','CONTRACT_STAGE_DRIFT'); require(not contract['claims']['nsw_tuning_allowed'],'NSW_TUNING_MUST_BE_FORBIDDEN')
    provenance=_verify_inputs(root,paths)
    b3=Stage3BBridge(paths['stage3b_source_root'],paths['stage3a_output_zip'],paths['stage3b_output_zip']); alignment=_case_pilot_alignment(b3,contract)
    c3=Stage3CBridge(paths['stage3c_source_root']); threshold_plan_alignment=_threshold_plan_alignment(c3,contract); m34=M3M4Bridge(paths['stage3d_d4_source_root']); d2=D2Bridge(paths['stage3d_d2_source_root'],float(contract['alpha'])); e3=Stage3EBridge(paths['stage3e_source_root'])
    alignment['threshold_plan_alignment']=threshold_plan_alignment
    alignment['metric_reporting_scope']={
      'pilot_fully_computed':['MET01','MET02','MET03','MET04','MET06','MET09','MET10','MET11','MET12','MET13','MET14','MET15','MET16','MET17','MET18','MET19'],
      'pilot_exact_baselines_plus_diagnostic_subset':['MET08','MET20'],
      'deferred_to_production_matched_coverage':['MET07'],
      'deferred_controlled_diagnostics_to_production':['MET21','MET22'],
      'not_applicable_until_minedosebench_or_real_demo':['MET05','MET23'],
      'reporting_contract_note':'M3/M4/M6 full-set width/WIS and component topology are diagnostic-only on the frozen width-audit subset in Stage3F; publication matched-coverage efficiency is intentionally deferred.'
    }
    seeds=b3.stage3a['frames']['seed_registry.csv']; pilot_ids=list(contract['pilot_case_ids']); sparse_cache={}; result_rows=[];support_rows=[];target_rows=[];width_rows=[];fit_rows=[];nuisance_rows=[];graph_rows=[];rep1_rows=[];outcome_fit_rows=[];case_runtime_rows=[]
    for case_pos,case_id in enumerate(pilot_ids,1):
        case=b3.cases[case_id]
        print(f'[PILOT] Case {case_pos}/{len(pilot_ids)}: {case_id} ({case.scenario_id}), 20 frozen replications...',flush=True)
        for rep in range(1,int(contract['replications'])+1):
            tc=time.perf_counter(); gen=b3.generate(case_id,rep); units=gen['units'].sort_values('node_index').reset_index(drop=True); true_edges=gen['true_edges'][['source_node','target_node']].copy()
            # Stage3F S1-S10 pilot intentionally excludes the registered S4 design-shift
            # stress because Stage3E did not certify its nonidentity estimated target-design
            # ratio.  The scalable pilot therefore requires exact identity design transport.
            require(np.max(np.abs(units.oracle_target_design_ratio.to_numpy(float)-1.0))<=1e-15,f'PILOT_NONIDENTITY_TARGET_DESIGN_RATIO_NOT_CERTIFIED:{case_id}:rep={rep}')
            if rep==1: rep1_rows.append(b3.rep1_replay_audit(case_id,gen))
            elig=eligible_target_blocks(units,true_edges,size=6); target_node,block,tsel=select_uniform_eligible_target(b3.core,b3.stage3a,case,rep,elig); target,observational=select_target(case,gen,target_node); target['_target_node']=target_node
            cal=units[units.role.astype(str)=='calibration'].sort_values('node_index').copy(); lw=oracle_calibration_logweights(case,gen,cal,observational); sw=summarize_logweights(lw)
            _,_,kl_true,_=_get_sparse(d2,e3,case,units,true_edges,int(contract['primary_scalable_neighborhood_size']),sparse_cache); lb=max(0.0,1-float(contract['alpha'])-math.sqrt(max(0.,kl_true)/2.0)); gcount=component_calibration_count(units,true_edges,target_node)
            tcache=c3.graph.build_graph_cache(units,true_edges,np.asarray([target_node],int)); gsafe=c3.graph.deterministic_graph_safe_set(tcache,target_node,cal.node_index.to_numpy(int)); endpoint_requested=bool((not observational) and float(case.primary_target_dose) in (0.0,1.0)); endpoint_audited=bool(units.loc[units.node_index.astype(int)==int(target_node),'endpoint_audited'].iloc[0])
            support_rows.append({'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':rep,'target_node':target_node,'target_grid_row':int(units.loc[units.node_index==target_node,'grid_row'].iloc[0]),'target_grid_col':int(units.loc[units.node_index==target_node,'grid_col'].iloc[0]),'n_rows':int(case.n_rows),'n_cols':int(case.n_cols),'requested_dose':float(target.A_star if observational else case.primary_target_dose),'target_dose':float(target.A_star if observational else case.primary_target_dose),'intervention_bandwidth':(float(target.get('bandwidth')) if pd.notna(target.get('bandwidth',np.nan)) else np.nan),'A_star':float(target.A_star),'target_population_eligible':True,'target_supported_by_generator':bool(target.get('target_supported_by_generator',True)),'positivity_status':('SUPPORTED' if bool(target.get('target_supported_by_generator',True)) else 'UNSUPPORTED'),'endpoint_requested':endpoint_requested,'endpoint_audited':endpoint_audited,'calibration_count':len(cal),'positive_weight_count':int(sw['positive_weight_count']),'ess':float(sw['ess']),'max_normalized_weight':float(sw['max_normalized_weight']),'log_weight_range':float(sw['log_weight_range']),'graph_component_calibration_count':int(gcount),'graph_safe_count':int(len(gsafe)),'n2_delta_kl':float(kl_true),'n2_tv_bound':float(math.sqrt(max(0.,kl_true)/2.0)),'certified_coverage_lower_bound':float(lb),'threshold_inputs_use_outcomes':False,'threshold_inputs_use_width':False,'threshold_inputs_use_nsw':False})
            target_rows.append({**tsel,'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':rep,'block_nodes':'|'.join(map(str,block)),'eligible_nodes_sha256':hashlib.sha256('|'.join(map(str,sorted(elig))).encode()).hexdigest(),'selection_uses_A':False,'selection_uses_Y':False,'selection_uses_support_diagnostics':False})
            do_width=(rep in set(map(int,contract['width_audit_reps'])) and str(contract['width_audit_case_by_scenario'][case.scenario_id])==case.case_id)
            # Structural/oracle track.
            rr,ww,gdiag=_evaluate_track(b3=b3,c3=c3,m34=m34,d2=d2,e3=e3,case=case,generated=gen,replication=rep,target_node=target_node,block=block,target=target,observational=observational,track='ORACLE_STRUCTURAL_OT_TG',model=None,contract=contract,sparse_cache=sparse_cache,width_audit=False)
            result_rows.extend(rr); graph_rows.append({'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':rep,'track':'ORACLE_STRUCTURAL_OT_TG','true_graph_edges':len(true_edges),'fitted_graph_edges':len(gen['fitted_edges']),'target_block_connected_under_track_graph':bool(d2.gp.block_is_connected(np.asarray(block,int),true_edges)),**gdiag})
            # RF shared-predictor empirical track is mandatory for all scenarios/cases.
            rf,rfa=fit_outcome(c3,seeds,case,rep,units,'rf'); outcome_fit_rows.append({**rfa,'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':rep,'track':'RF_SHARED_PREDICTOR_EMPIRICAL','predictor':'rf'})
            rr,ww,gdiag=_evaluate_track(b3=b3,c3=c3,m34=m34,d2=d2,e3=e3,case=case,generated=gen,replication=rep,target_node=target_node,block=block,target=target,observational=observational,track='RF_SHARED_PREDICTOR_EMPIRICAL',model=rf,contract=contract,sparse_cache=sparse_cache,width_audit=do_width)
            result_rows.extend(rr);width_rows.extend(ww); pedges=primary_graph(case,gen,'RF_SHARED_PREDICTOR_EMPIRICAL');graph_rows.append({'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':rep,'track':'RF_SHARED_PREDICTOR_EMPIRICAL','true_graph_edges':len(true_edges),'fitted_graph_edges':len(gen['fitted_edges']),'target_block_connected_under_track_graph':bool(d2.gp.block_is_connected(np.asarray(block,int),pedges)),**gdiag})
            nd,nf=_nuisance_diagnostics(b3=b3,c3=c3,m34=m34,d2=d2,e3=e3,case=case,generated=gen,replication=rep,target_node=target_node,block=block,target=target,observational=observational,model_rf=rf,contract=contract,sparse_cache=sparse_cache); nuisance_rows.extend(nd);fit_rows.extend(nf)
            # XGBoost confirmation only for the predeclared scenario scope.
            if case.scenario_id in set(contract['xgb_scenarios']):
                xgb,xga=fit_outcome(c3,seeds,case,rep,units,'xgb'); outcome_fit_rows.append({**xga,'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':rep,'track':'XGB_SHARED_PREDICTOR_EMPIRICAL','predictor':'xgb'})
                rr,ww,gdiag=_evaluate_track(b3=b3,c3=c3,m34=m34,d2=d2,e3=e3,case=case,generated=gen,replication=rep,target_node=target_node,block=block,target=target,observational=observational,track='XGB_SHARED_PREDICTOR_EMPIRICAL',model=xgb,contract=contract,sparse_cache=sparse_cache,width_audit=False)
                result_rows.extend(rr); pedges=primary_graph(case,gen,'XGB_SHARED_PREDICTOR_EMPIRICAL');graph_rows.append({'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':rep,'track':'XGB_SHARED_PREDICTOR_EMPIRICAL','true_graph_edges':len(true_edges),'fitted_graph_edges':len(gen['fitted_edges']),'target_block_connected_under_track_graph':bool(d2.gp.block_is_connected(np.asarray(block,int),pedges)),**gdiag})
            case_runtime_rows.append({'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':rep,'n_units':len(units),'elapsed_seconds':float(time.perf_counter()-tc)})
            if rep in {5,10,15,20}: print(f'  completed replication {rep}/20',flush=True)
        print(f'[PILOT] Completed {case_id}.',flush=True)
    support=pd.DataFrame(support_rows)
    results=pd.DataFrame(result_rows)
    results['raw_false_support']=results.raw_method_returned.astype(bool) & ~results.target_supported.astype(bool)
    thresholds,threshold_audit=choose_thresholds(support,results,contract)
    # Apply thresholds only after all 20-rep outcome-blind support rows are complete.
    support_key=support.set_index(['case_id','replication'])
    returned=[]; reasons=[]; false_support=[]
    for r in results.itertuples(index=False):
        s=support_key.loc[(r.case_id,int(r.replication))].copy(); row=pd.Series(r._asdict())
        # Structural target/support fields are frozen from the pre-outcome support table;
        # method-specific ESS/weight/graph-safe diagnostics come from the assigned track.
        for k in ['target_population_eligible','target_supported_by_generator','endpoint_requested','endpoint_audited']:
            row[k]=s[k]
        gate,reason=operational_gate(r.method,row,thresholds); ok=bool(r.raw_method_returned) and gate; returned.append(ok); reasons.append(str(r.raw_refusal_code) if not r.raw_method_returned else ('' if gate else reason)); false_support.append(bool(ok and not bool(s.target_supported_by_generator)))
    results['operational_returned']=returned;results['operational_refusal_code']=reasons;results['false_support']=false_support
    metrics=aggregate_metrics(results,float(contract['alpha'])); sanity=coverage_sanity(results,contract); loc,locsum=_local_metrics(results,float(contract['alpha'])); dose=dose_bin_metrics(results,float(contract['alpha']),contract['dose_grid']); certsum=certificate_summary(support); widthsum=width_diagnostic_summary(pd.DataFrame(width_rows)); runtime_case=pd.DataFrame(case_runtime_rows)
    if len(sanity): sanity['inferential_status']='DESCRIPTIVE_ONLY_THRESHOLDS_SELECTED_ON_SAME_20REP_PILOT'
    # The 20-rep pilot is a tuning gate, not independent confirmation. Production is the
    # first untouched evaluation of the frozen thresholds, so same-pilot binomial p-values
    # are reported descriptively and never used to declare pilot success.
    primary_fail=float(results.computational_failure.mean()); fs=int(results.false_support.sum()); sanity_bad=int(sanity.significant_undercoverage.sum()) if len(sanity) else 0
    # A threshold tuple is not frozen as production-ready if selective refusal leaves too
    # little evidence in any registered safe case.  This is not an additional threshold
    # grid: it is the predeclared casewise coverage-objective sample requirement used by
    # the selector itself. Production remains the first independent coverage evaluation.
    min_case_n=int(contract['threshold_selection']['minimum_returned_per_case_for_casewise_coverage_objective'])
    threshold_evidence_adequate=bool(
        int(thresholds.get('safe_case_coverage_groups_used',0))==int(thresholds.get('safe_case_count',-1))
        and int(thresholds.get('minimum_safe_case_returned',0))>=min_case_n
        and thresholds.get('safe_pooled_selective_coverage') is not None
    )
    pilot_ready=bool(primary_fail==0.0 and fs==0 and threshold_evidence_adequate and thresholds.get('status')=='FROZEN_AFTER_PREREGISTERED_20REP_PILOT')
    # Sample-size/runtime freeze is descriptive and does not invent unregistered small/large DGPs.
    sample_freeze={'primary_controlled_lattice':'25x25=625 units','S10_180m_lattice':'13x13=169 units','exact_orbit_primary':6,'exact_orbit_stress_stage3a':8,'F06_status':'RUNTIME_EVIDENCE_COLLECTED_BUT_SMALL_LARGE_TOTAL_LATTICE_DGPS_NOT_DEFINED_IN_FROZEN_STAGE3B','production_rule':'before production outcomes, the production package must make one explicit pre-outcome generator-extension freeze for F06 small/large or formally mark F06 omitted; it may use this Windows pilot runtime evidence but must not use production outcomes','reason':'Stage3A registered F06 small/primary/large with exact sizes to freeze after runtime pilot, while Stage3B v1.2 defines primary 25x25 and S10 13x13 support but no generic F06 small/large CaseConfigs. Stage3F does not silently invent a DGP.'}
    claim={'stage':STAGE,'version':VERSION,'pilot_scale_ready':pilot_ready,'production_run':False,'publication_performance_evidence':False,'threshold_selection_used_controlled_pilot_outcomes':True,'threshold_selection_used_outcomes':True,'threshold_selection_used_width':False,'threshold_selection_used_nsw':False,'coverage_sanity_is_independent_confirmation':False,'coverage_sanity_status':'DESCRIPTIVE_ONLY_BECAUSE_THRESHOLDS_USE_SAME_20REP_TUNING_PILOT','threshold_freeze_evidence_adequate':threshold_evidence_adequate,'minimum_returned_per_safe_case_for_freeze':min_case_n,'threshold_selector_detail_status':'STAGE3F_SOFTWARE_FROZEN_BEFORE_PRODUCTION_OUTCOMES','threshold_selector_detail_preregistration_claim':False,'candidate_grids_and_coverage_refusal_principle_frozen_upstream':True,'oracle_structural_m6_theorem_certified':True,'estimated_nuisance_certified':False,'rf_xgb_shared_predictor_certified':False,'width_audit_grid_coverage_preserving':False,'production_blocked_until_independent_review':True,'production_blocked_until_F06_sample_size_freeze':True,'r11_real_eo_rule':contract['r11_eo_quality_rule']}
    # Persist deterministic scientific outputs.
    deterministic_gzip_csv(results,output_dir/'stage3f_pilot_query_results.csv.gz'); deterministic_gzip_csv(support,output_dir/'stage3f_support_certificate_inputs.csv.gz')
    pd.DataFrame(target_rows).to_csv(output_dir/'stage3f_target_block_selection_audit.csv',index=False,lineterminator='\n',float_format='%.17g'); pd.DataFrame(width_rows).to_csv(output_dir/'stage3f_width_diagnostic_audit.csv',index=False,lineterminator='\n',float_format='%.17g'); pd.DataFrame(graph_rows).to_csv(output_dir/'stage3f_graph_diagnostics.csv',index=False,lineterminator='\n',float_format='%.17g'); pd.DataFrame(outcome_fit_rows).to_csv(output_dir/'stage3f_outcome_fit_audit.csv',index=False,lineterminator='\n',float_format='%.17g'); pd.DataFrame(nuisance_rows).to_csv(output_dir/'stage3f_propensity_specification_diagnostics.csv',index=False,lineterminator='\n',float_format='%.17g'); pd.DataFrame(fit_rows).to_csv(output_dir/'stage3f_propensity_fit_audit.csv',index=False,lineterminator='\n',float_format='%.17g'); pd.DataFrame(rep1_rows).to_csv(output_dir/'stage3f_stage3b_rep1_replay_audit.csv',index=False,lineterminator='\n',float_format='%.17g')
    threshold_audit.to_csv(output_dir/'stage3f_threshold_candidate_audit.csv',index=False,lineterminator='\n',float_format='%.17g'); metrics.to_csv(output_dir/'stage3f_pilot_metrics.csv',index=False,lineterminator='\n',float_format='%.17g'); sanity.to_csv(output_dir/'stage3f_certified_coverage_sanity.csv',index=False,lineterminator='\n',float_format='%.17g');loc.to_csv(output_dir/'stage3f_local_coverage_metrics.csv',index=False,lineterminator='\n',float_format='%.17g');locsum.to_csv(output_dir/'stage3f_local_coverage_summary.csv',index=False,lineterminator='\n',float_format='%.17g');dose.to_csv(output_dir/'stage3f_dose_bin_coverage_metrics.csv',index=False,lineterminator='\n',float_format='%.17g');certsum.to_csv(output_dir/'stage3f_certificate_support_summary.csv',index=False,lineterminator='\n',float_format='%.17g');widthsum.to_csv(output_dir/'stage3f_width_diagnostic_summary.csv',index=False,lineterminator='\n',float_format='%.17g');runtime_case.to_csv(output_dir/'stage3f_case_rep_runtime.csv',index=False,lineterminator='\n',float_format='%.17g')
    json_dump(output_dir/'stage3f_operational_thresholds_FROZEN.json',thresholds);json_dump(output_dir/'stage3f_claim_boundary.json',claim);json_dump(output_dir/'stage3f_sample_size_runtime_freeze.json',sample_freeze);json_dump(output_dir/'stage3f_upstream_provenance_audit.json',provenance);json_dump(output_dir/'stage3f_registry_alignment.json',alignment);json_dump(output_dir/'stage3f_source_inventory.json',source_inventory(root))
    runtime={'stage':STAGE,'version':VERSION,'elapsed_seconds':time.perf_counter()-t0,'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':__import__('scipy').__version__,'result_rows':len(results),'support_rows':len(support),'width_audit_rows':len(width_rows),'nuisance_diagnostic_rows':len(nuisance_rows),'sparse_cache_entries':len(sparse_cache),'case_rep_runtime_rows':len(case_runtime_rows),'pilot_scale_ready':pilot_ready};json_dump(output_dir/'stage3f_runtime.json',runtime)
    # Manifest is written last and excludes itself/verification generated downstream.
    man=output_manifest(output_dir,exclude={'stage3f_manifest.json','STAGE3F_VERIFICATION.json'});json_dump(output_dir/'stage3f_manifest.json',man)
    return {'pilot_scale_ready':pilot_ready,'thresholds':thresholds,'results':len(results),'support_rows':len(support),'runtime_seconds':runtime['elapsed_seconds'],'sanity_significant_undercoverage_descriptive_only':sanity_bad,'false_support':fs}
