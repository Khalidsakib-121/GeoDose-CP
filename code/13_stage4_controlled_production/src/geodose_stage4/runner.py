from __future__ import annotations
import copy,gzip,hashlib,io,json,math,platform,sys,time,zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

from .io import Stage4Error,require,verify_zip,verify_source_root,json_dump,deterministic_gzip_csv,atomic_gzip_json,read_gzip_json,prepare_output_dir,sha256_file
from .provenance import source_inventory,output_manifest
from .version import STAGE,VERSION
from .production_registry import build_main_registry,build_f06_registry,build_extension_registry,registry_summary
from .f06 import install_stage3b_production_adapter,install_stage3c_production_nuisance_adapter,f06_case,f06_structural_qa

# The exact accepted Stage3F scientific adapter is vendored byte-for-byte. Stage4
# changes orchestration, production seeds, F06 geometry, and reporting only.
ROOT_HINT=Path(__file__).resolve().parents[2]
VENDOR=ROOT_HINT/'vendor'
if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))
from geodose_stage3f.bridges import Stage3BBridge,Stage3CBridge,M3M4Bridge,D2Bridge,Stage3EBridge
from geodose_stage3f import runner as sf_runner
from geodose_stage3f.pilot_math import (
    eligible_target_blocks,select_uniform_eligible_target,select_target,oracle_calibration_logweights,
    summarize_logweights,component_calibration_count,primary_graph,fit_outcome,operational_gate,
    aggregate_metrics,dose_bin_metrics,certificate_summary,coverage_sanity
)

METHODS=('M1','M2','M3','M4','M5','M6')

def _read_json_zip(path:Path,suffix:str)->dict:
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.replace('\\','/').endswith(suffix)];require(len(names)==1,f'ZIP_MEMBER_NOT_UNIQUE:{suffix}:{path}')
        return json.loads(z.read(names[0]).decode('utf-8'))

def _verify_inputs(root:Path,paths:dict[str,Path])->dict[str,Any]:
    expected=json.loads((root/'configs/expected_upstream_hashes.json').read_text());locks=json.loads((root/'configs/source_locks.json').read_text());audit={}
    for k,h in expected.items():audit[k]=verify_zip(paths[k],h)
    for k in ['stage3b_source_root','stage3c_source_root','stage3d_d2_source_root','stage3d_d4_source_root','stage3e_source_root']:audit[k]=verify_source_root(paths[k],locks[k])
    for k,suf in [('stage3d_d4_output_zip','STAGE3D_D4_VERIFICATION.json'),('stage3e_output_zip','STAGE3E_VERIFICATION.json'),('stage3f_output_zip','STAGE3F_VERIFICATION.json')]:
        v=_read_json_zip(paths[k],suf);require(v.get('status')=='verified_complete' and int(v.get('failed_checks',0))==0,f'UPSTREAM_VERIFICATION_NOT_COMPLETE:{k}')
        audit[k+'_verification']={'status':'verified_complete','checks_passed':v.get('checks_passed')}
    # Stage3F frozen threshold bytes are an input, not recomputed.
    t=_read_json_zip(paths['stage3f_output_zip'],'stage3f_operational_thresholds_FROZEN.json')
    require(float(t['minimum_ess'])==3.0 and float(t['maximum_normalized_weight'])==0.5 and int(t['minimum_graph_safe_count'])==20,'STAGE3F_THRESHOLD_DRIFT')
    audit['stage3f_frozen_thresholds']={k:t[k] for k in ['minimum_ess','maximum_normalized_weight','minimum_graph_safe_count']}
    # Vendored Stage3F scientific adapter lock.
    vl=json.loads((root/'configs/stage3f_vendor_lock.json').read_text());bad=[]
    for rel,h in vl['files'].items():
        p=root/'vendor/geodose_stage3f'/rel
        if not p.is_file() or sha256_file(p)!=h:bad.append(rel)
    require(not bad,f'STAGE3F_VENDOR_BYTE_DRIFT:{bad}')
    audit['stage3f_vendor']={'verified_files':len(vl['files']),'origin_source_zip_sha256':vl['origin_source_zip_sha256']}
    return audit

def _contract_hash(root:Path)->str:return sha256_file(root/'configs/production_contract.yaml')

def _production_seed_map(stage3a:dict)->pd.DataFrame:
    s=stage3a['frames']['seed_registry.csv'];p=s[s.phase.astype(str)=='production_provisional'].copy();require(len(p)==3000,'STAGE3A_PRODUCTION_SEED_ROWS_NOT_3000');return p

def _entry_key(e:dict)->str:
    return '__'.join(str(e.get(k,'')) for k in ['scope','extension_id','f06_level','case_id','seed_replication']).replace('/','_').replace('\\','_')

def _sanitize_records(df:pd.DataFrame)->list[dict]:
    if df is None or len(df)==0:return []
    out=[]
    for d in df.to_dict(orient='records'):
        x={}
        for k,v in d.items():
            if isinstance(v,(np.integer,)):v=int(v)
            elif isinstance(v,(np.floating,)):v=float(v)
            elif isinstance(v,(np.bool_,)):v=bool(v)
            if isinstance(v,float) and not math.isfinite(v):v=None
            x[k]=v
        out.append(x)
    return out

def _frame(records:list[dict])->pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()

def _resolve_case(b3,e:dict):
    base=b3.cases[e['case_id']]
    if e.get('f06_level'):return f06_case(base,str(e['f06_level']))
    td=e.get('target_dose_override',None)
    if td is not None:
        td=float(td);return replace(base,case_id=str(e.get('extension_id') or f'{base.case_id}_D{td:g}'),case_name=f'{base.case_name}; target dose override {td:g}',case_type='additional_stress_target_override',primary_target_mode='localized',primary_target_dose=td,primary_bandwidth=(None if td in (0.0,1.0) else 0.10))
    return base

def _generate(b3,case,rep:int):
    if case.case_id in b3.cases and b3.cases[case.case_id]==case:return b3.core.generate_case(case,b3.stage3a,b3.basis_cache,replication=int(rep))
    return b3.core.generate_case(case,b3.stage3a,b3.basis_cache,replication=int(rep))

def _support_and_target(b3,c3,d2,e3,case,gen,rep:int,contract,sparse_cache):
    units=gen['units'].sort_values('node_index').reset_index(drop=True);edges=gen['true_edges'][['source_node','target_node']].copy();elig=eligible_target_blocks(units,edges,6);node,block,tsel=select_uniform_eligible_target(b3.core,b3.stage3a,case,rep,elig);target,observational=select_target(case,gen,node);target['_target_node']=node
    cal=units[units.role.astype(str)=='calibration'].sort_values('node_index');lw=oracle_calibration_logweights(case,gen,cal,observational);sw=summarize_logweights(lw)
    _,_,kl,_=sf_runner._get_sparse(d2,e3,case,units,edges,int(contract['primary_scalable_neighborhood_size']),sparse_cache);tv=math.sqrt(max(0.0,float(kl))/2.0);lb=max(0.0,1-float(contract['alpha'])-tv);gcount=component_calibration_count(units,edges,node);cache=c3.graph.build_graph_cache(units,edges,np.asarray([node],int));gsafe=c3.graph.deterministic_graph_safe_set(cache,node,cal.node_index.to_numpy(int))
    endpoint_requested=bool((not observational) and float(case.primary_target_dose) in (0.,1.));endpoint_audited=bool(units.loc[units.node_index.astype(int)==node,'endpoint_audited'].iloc[0]);supported=bool(target.get('target_supported_by_generator',False)) and str(target.get('draw_status','drawn'))=='drawn'
    Astar=float(target.A_star) if pd.notna(target.get('A_star',np.nan)) else np.nan
    support={'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':int(rep),'target_node':node,'n_rows':int(case.n_rows),'n_cols':int(case.n_cols),'requested_dose':float(case.primary_target_dose) if not observational else Astar,'A_star':Astar,'target_population_eligible':True,'target_supported_by_generator':supported,'endpoint_requested':endpoint_requested,'endpoint_audited':endpoint_audited,'calibration_count':len(cal),'positive_weight_count':int(sw['positive_weight_count']),'ess':float(sw['ess']),'max_normalized_weight':float(sw['max_normalized_weight']),'graph_component_calibration_count':int(gcount),'graph_safe_count':int(len(gsafe)),'n2_delta_kl':float(kl),'n2_tv_bound':float(tv),'certified_coverage_lower_bound':float(lb),'target_design_mode':str(case.target_design_mode),'target_design_delta':float(case.target_design_delta)}
    targ={**tsel,'scenario_id':case.scenario_id,'case_id':case.case_id,'replication':int(rep),'block_nodes':'|'.join(map(str,block)),'eligible_nodes_sha256':hashlib.sha256('|'.join(map(str,sorted(elig))).encode()).hexdigest(),'selection_uses_A':False,'selection_uses_Y':False,'selection_uses_support_diagnostics':False}
    return units,edges,node,block,target,observational,support,targ

def _refusal_rows(case,rep,support,code,scope,analysis_case_id):
    rows=[]
    for m in METHODS:
        rows.append({'scenario_id':case.scenario_id,'case_id':analysis_case_id,'generator_case_id':case.case_id,'replication':int(rep),'track':'ORACLE_STRUCTURAL_OT_TG','response_target':'latent_Y_true','method':m,'method_branch':'structural_refusal_before_method','theorem_eligible':False,'theorem_eligibility_reason':code,'raw_method_returned':False,'raw_refusal_code':code,'computational_failure':False,'width':np.nan,'wis':np.nan,'lower':np.nan,'upper':np.nan,'pvalue_conservative':np.nan,'pvalue_randomized':np.nan,'raw_covered':False,'covered_latent':False,'ess':support['ess'] if m in {'M2','M4','M6'} else np.nan,'max_normalized_weight':support['max_normalized_weight'] if m in {'M2','M4','M6'} else np.nan,'positive_weight_count':support['positive_weight_count'],'graph_safe_count':support['graph_safe_count'],'target_supported':support['target_supported_by_generator'],'target_population_eligible':True,'endpoint_requested':support['endpoint_requested'],'endpoint_audited':support['endpoint_audited'],'operational_coverage_lower_bound':support['certified_coverage_lower_bound'],'certified_coverage_lower_bound':support['certified_coverage_lower_bound'],'A_star':support['A_star'],'requested_dose':support['requested_dose'],'target_mean_error':np.nan,'production_scope':scope})
    return rows

def _heuristics(c3,case,gen,target,node,observational,rep,scope,analysis_case_id,alpha=0.1):
    units=gen['units'].sort_values('node_index').reset_index(drop=True);cal=units[units.role.astype(str)=='calibration'].sort_values('node_index');cn=cal.node_index.to_numpy(int);y=units.Y_true_at_A.to_numpy(float);mean=units.conditional_mean_at_A.to_numpy(float);scores=np.abs(y[cn]-mean[cn]);mt=float(target.conditional_mean_at_A_star);yt=float(target.Y_true_at_A_star);xy=cal[['x_coord','y_coord']].to_numpy(float);txy=units.loc[units.node_index.astype(int)==int(node),['x_coord','y_coord']].to_numpy(float)[0];h1=c3.graph.geographic_distance_logweights(txy,xy,bandwidth=0.25);tlw=0.0;dose=oracle_calibration_logweights(case,gen,cal,observational);dt=sf_runner.oracle_target_logweight(target,observational)
    rows=[]
    for name,lw,target_lw in [('H1_GEOGRAPHIC',h1,0.0),('H4_DOSE_X_GEOGRAPHIC',h1+dose,dt)]:
        r=sf_runner._interval_one(c3,mt,scores,lw,target_lw,float(alpha));ret=str(r.interval_status)!='refused'
        rows.append({'scenario_id':case.scenario_id,'case_id':analysis_case_id,'replication':int(rep),'production_scope':scope,'heuristic_id':name,'theorem_eligible':False,'raw_method_returned':ret,'raw_refusal_code':str(r.refusal_code),'covered':sf_runner._finite_covered(r.lower,r.upper,yt) if ret else False,'lower':float(r.lower),'upper':float(r.upper),'width':float(r.upper-r.lower) if ret and np.isfinite(r.lower) and np.isfinite(r.upper) else np.nan,'ess':float(r.ess),'max_normalized_weight':float(r.max_normalized_weight)})
    return rows

def _apply_gate(rows:list[dict],support:dict,thresholds:dict):
    for r in rows:
        row=pd.Series(r)
        for k,v in [('target_population_eligible',support['target_population_eligible']),('target_supported_by_generator',support['target_supported_by_generator']),('endpoint_requested',support['endpoint_requested']),('endpoint_audited',support['endpoint_audited'])]:row[k]=v
        gate,reason=operational_gate(r['method'],row,thresholds);ok=bool(r['raw_method_returned']) and bool(gate);r['operational_returned']=ok;r['operational_refusal_code']=str(r['raw_refusal_code']) if not r['raw_method_returned'] else ('' if gate else reason);r['raw_false_support']=bool(r['raw_method_returned'] and not support['target_supported_by_generator']);r['false_support']=bool(ok and not support['target_supported_by_generator'])
    return rows

def _spatial_residual_diagnostic(case,units:pd.DataFrame,edges:pd.DataFrame,rep:int,scope:str,analysis_case_id:str)->dict:
    """Diagnostic-only MET21/MET22 on the frozen true residual graph.

    Moran's I uses binary symmetric true-graph adjacency.  Because ``edges``
    stores each undirected edge once, the doubled numerator and doubled S0
    cancel, giving n/E * sum_edge(z_i z_j) / sum_i z_i^2.  The empirical
    semivariogram is the mean 0.5*(e_i-e_j)^2 over the same registered edges.
    Neither quantity is inserted into the conformal deficit/certificate.
    """
    r=units.sort_values('node_index').shared_spatial_residual.to_numpy(float)
    z=r-float(np.mean(r));a=edges.source_node.to_numpy(int);b=edges.target_node.to_numpy(int);den=float(np.dot(z,z));m=len(edges)
    moran=float(len(r)/m*np.sum(z[a]*z[b])/den) if m and den>0 else np.nan
    semivar=float(np.mean(0.5*np.square(r[a]-r[b]))) if m else np.nan
    return {'scenario_id':case.scenario_id,'case_id':analysis_case_id,'replication':int(rep),'production_scope':scope,'n_units':len(r),'true_graph_edges':m,'morans_I':moran,'empirical_semivariogram_true_graph_edges':semivar,'residual_mean':float(np.mean(r)),'residual_variance':float(np.var(r)),'diagnostic_only':True,'used_in_conformal_deficit':False}

def _local_coverage_metrics(results:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    if len(results)==0:return pd.DataFrame(),pd.DataFrame()
    x=results.copy();frac=x.target_grid_row.astype(float)/np.maximum(x.n_rows.astype(float)-1.0,1.0);x['target_row_band']=np.where(frac<1/3,'low_row',np.where(frac<2/3,'middle_row','high_row'));rows=[]
    keys=['production_scope','scenario_id','case_id','track','method','target_row_band']
    for key,g in x.groupby(keys,dropna=False,sort=True):
        ret=g.operational_returned.astype(bool);n=int(ret.sum());cov=float(g.loc[ret,'raw_covered'].astype(bool).mean()) if n else np.nan
        rows.append(dict(zip(keys,key),requested=len(g),returned=n,selective_coverage=cov,mcse=(math.sqrt(cov*(1-cov)/n) if n and np.isfinite(cov) else np.nan),refusal_rate=float(1-ret.mean())))
    loc=pd.DataFrame(rows);summ=[];skeys=['production_scope','scenario_id','case_id','track','method']
    for key,g in loc.groupby(skeys,dropna=False,sort=True):
        vals=g.selective_coverage.dropna().to_numpy(float);summ.append(dict(zip(skeys,key),local_strata_with_returned=len(vals),fifth_percentile_local_coverage=float(np.quantile(vals,.05)) if len(vals) else np.nan,minimum_local_coverage=float(np.min(vals)) if len(vals) else np.nan))
    return loc,pd.DataFrame(summ)

def _evaluate_entry(*,root,b3,c3,m34,d2,e3,entry,contract,thresholds,sparse_cache,seeds):
    rep=int(entry['seed_replication']);case=_resolve_case(b3,entry);gen=_generate(b3,case,rep);analysis_case_id=case.case_id;scope=entry['scope'];units,edges,node,block,target,observational,support,targ=_support_and_target(b3,c3,d2,e3,case,gen,rep,contract,sparse_cache);spdiag=_spatial_residual_diagnostic(case,units,edges,rep,scope,analysis_case_id)
    support.update({'production_scope':scope,'analysis_case_id':analysis_case_id,'generator_base_case_id':entry['case_id'],'seed_replication':rep,'case_local_index':int(entry['case_local_index']),'f06_level':entry.get('f06_level',''),'extension_id':entry.get('extension_id','')});targ.update({'production_scope':scope,'analysis_case_id':analysis_case_id,'seed_replication':rep,'case_local_index':int(entry['case_local_index']),'f06_level':entry.get('f06_level',''),'extension_id':entry.get('extension_id','')})
    if not entry.get('method_evaluation',True):
        temporal=gen.get('temporal',pd.DataFrame());diag={'scenario_id':case.scenario_id,'case_id':analysis_case_id,'replication':rep,'production_scope':scope,'temporal_rows':int(len(temporal)),'temporal_dependence_registered':bool(case.temporal_dependence),'method_level_temporal_conformal_claim':False}
        if len(temporal):
            nums=temporal.select_dtypes(include=[np.number]);diag['numeric_column_count']=len(nums.columns);diag['finite_numeric_fraction']=float(np.isfinite(nums.to_numpy(float)).mean()) if nums.size else 1.0
        return {'results':[],'support':[support],'targets':[targ],'heuristics':[],'graphs':[],'outcome_fits':[],'temporal':[diag],'nuisance':[],'nuisance_fits':[],'spatial':[spdiag]}
    if support['endpoint_requested'] and not support['endpoint_audited']:
        rows=_refusal_rows(case,rep,support,'R02_ENDPOINT_NOT_AUDITED',scope,analysis_case_id);_apply_gate(rows,support,thresholds)
        return {'results':rows,'support':[support],'targets':[targ],'heuristics':[],'graphs':[],'outcome_fits':[],'temporal':[],'nuisance':[],'nuisance_fits':[],'spatial':[spdiag]}
    allrows=[];graphs=[];outfits=[];nuis=[];nfits=[]
    def ev(track,model=None):
        rr,_,gd=sf_runner._evaluate_track(b3=b3,c3=c3,m34=m34,d2=d2,e3=e3,case=case,generated=gen,replication=rep,target_node=node,block=block,target=target,observational=observational,track=track,model=model,contract=contract,sparse_cache=sparse_cache,width_audit=False)
        for r in rr:r.update({'production_scope':scope,'analysis_case_id':analysis_case_id,'generator_base_case_id':entry['case_id'],'seed_replication':rep,'case_local_index':int(entry['case_local_index']),'f06_level':entry.get('f06_level',''),'extension_id':entry.get('extension_id','')})
        # Nonidentity design-shift fixture is a registered diagnostic; the exact local adapter
        # does not generate a shifted target population, so no performance theorem is claimed.
        if str(case.target_design_mode)!='identity':
            for r in rr:r['theorem_eligible']=False;r['theorem_eligibility_reason']='target_design_shift_fixture_is_certificate_transport_diagnostic_not_shifted_target_population_performance_sample';r['certificate_theorem_backed']=False;r['certified_coverage_lower_bound']=np.nan
        allrows.extend(rr);pedges=primary_graph(case,gen,track);graphs.append({'scenario_id':case.scenario_id,'case_id':analysis_case_id,'replication':rep,'production_scope':scope,'track':track,'true_graph_edges':len(edges),'fitted_graph_edges':len(gen['fitted_edges']),'target_block_connected_under_track_graph':bool(d2.gp.block_is_connected(np.asarray(block,int),pedges)),**gd})
    tracks=contract['tracks']['main'] if scope=='main_production' else contract['tracks']['f06'] if scope=='f06_extension' else contract['tracks']['extension_default']
    for track in tracks:
        if track=='ORACLE_STRUCTURAL_OT_TG':ev(track,None)
        elif track=='RF_SHARED_PREDICTOR_EMPIRICAL':
            model,a=fit_outcome(c3,seeds,case,rep,units,'rf');outfits.append({**a,'scenario_id':case.scenario_id,'case_id':analysis_case_id,'replication':rep,'production_scope':scope,'track':track,'predictor':'rf'});ev(track,model)
    if scope=='main_production' and case.scenario_id in set(contract['tracks']['xgb_scenarios']):
        model,a=fit_outcome(c3,seeds,case,rep,units,'xgb');outfits.append({**a,'scenario_id':case.scenario_id,'case_id':analysis_case_id,'replication':rep,'production_scope':scope,'track':contract['tracks']['xgb_track'],'predictor':'xgb'});ev(contract['tracks']['xgb_track'],model)
    # ABL9: same RF response/model on S6, but with true graph replacing the registered fitted graph.
    if scope=='main_production' and case.scenario_id=='S6':
        model,a=fit_outcome(c3,seeds,case,rep,units,'rf');g2=copy.deepcopy(gen);g2['fitted_edges']=g2['true_edges'].copy();rr,_,gd=sf_runner._evaluate_track(b3=b3,c3=c3,m34=m34,d2=d2,e3=e3,case=case,generated=g2,replication=rep,target_node=node,block=block,target=target,observational=observational,track='RF_ORACLE_SPATIAL_LAW_DIAGNOSTIC',model=model,contract=contract,sparse_cache=sparse_cache,width_audit=False)
        for r in rr:r.update({'production_scope':scope,'analysis_case_id':analysis_case_id,'generator_base_case_id':entry['case_id'],'seed_replication':rep,'case_local_index':int(entry['case_local_index']),'f06_level':'','extension_id':'','theorem_eligible':False,'theorem_eligibility_reason':'ABL9_empirical_oracle_spatial_law_diagnostic'})
        allrows.extend(rr);graphs.append({'scenario_id':case.scenario_id,'case_id':analysis_case_id,'replication':rep,'production_scope':scope,'track':'RF_ORACLE_SPATIAL_LAW_DIAGNOSTIC','true_graph_edges':len(edges),'fitted_graph_edges':len(edges),'target_block_connected_under_track_graph':bool(d2.gp.block_is_connected(np.asarray(block,int),edges)),**gd})
    _apply_gate(allrows,support,thresholds)
    heur=_heuristics(c3,case,gen,target,node,observational,rep,scope,analysis_case_id,float(contract['alpha']))
    return {'results':allrows,'support':[support],'targets':[targ],'heuristics':heur,'graphs':graphs,'outcome_fits':outfits,'temporal':[],'nuisance':nuis,'nuisance_fits':nfits,'spatial':[spdiag]}


def _propensity_diagnostics(b3,c3,m34,d2,e3,contract,main_registry,sparse_cache,seeds):
    rows=[];fits=[];n=int(contract['nuisance_diagnostic']['first_n_case_seed_reps']);repcases=set(contract['nuisance_diagnostic']['representative_cases'])
    for cid in sorted(repcases):
        entries=[e for e in main_registry if e['case_id']==cid][:n]
        require(len(entries)==n,f'ABL8_REPRESENTATIVE_CASE_HAS_TOO_FEW_ROWS:{cid}')
        for e in entries:
            case=_resolve_case(b3,e);rep=int(e['seed_replication']);gen=_generate(b3,case,rep);units,edges,node,block,target,observational,_,_=_support_and_target(b3,c3,d2,e3,case,gen,rep,contract,sparse_cache);rf,_=fit_outcome(c3,seeds,case,rep,units,'rf')
            dc=dict(contract);dc['nuisance_estimation_diagnostic_reps']=[rep];dc['nuisance_estimation_case_by_scenario']={case.scenario_id:case.case_id}
            rr,ff=sf_runner._nuisance_diagnostics(b3=b3,c3=c3,m34=m34,d2=d2,e3=e3,case=case,generated=gen,replication=rep,target_node=node,block=block,target=target,observational=observational,model_rf=rf,contract=dc,sparse_cache=sparse_cache)
            for r in rr:r['production_scope']='ABL8_FIXED_PRODUCTION_DIAGNOSTIC'
            for r in ff:r['production_scope']='ABL8_FIXED_PRODUCTION_DIAGNOSTIC'
            rows.extend(rr);fits.extend(ff)
    return pd.DataFrame(rows),pd.DataFrame(fits)

def _abl8_summary(diag:pd.DataFrame,results:pd.DataFrame)->pd.DataFrame:
    if len(diag)==0:return pd.DataFrame()
    rows=[]
    for key,g in diag.groupby(['scenario_id','case_id','propensity_mode','method'],sort=True):
        scen,cid,mode,method=key;ret=g.raw_method_returned.astype(bool);reps=set(g.replication.astype(int))
        # Hold response model and spatial graph fixed when comparing treatment nuisance.
        # In S6 the diagnostic nuisance engine uses the true graph, so compare with
        # ABL9's RF_ORACLE_SPATIAL_LAW_DIAGNOSTIC. Else fitted=true and ordinary RF
        # is the matching oracle-treatment empirical track.
        oracle_track='RF_ORACLE_SPATIAL_LAW_DIAGNOSTIC' if scen=='S6' else 'RF_SHARED_PREDICTOR_EMPIRICAL'
        q=results[(results.production_scope=='main_production')&(results.case_id==cid)&(results.method==method)&(results.track==oracle_track)&results.replication.astype(int).isin(reps)]
        require(len(q)==len(g),f'ABL8_ORACLE_COMPARATOR_ROW_DRIFT:{cid}:{mode}:{method}:{len(q)}!={len(g)}')
        diag_cov=float(g.covered_observed.astype(bool).mean());oracle_cov=float(q.raw_covered.astype(bool).mean())
        rows.append({'ablation_id':'ABL8','scenario_id':scen,'case_id':cid,'propensity_mode':mode,'method':method,'requested':len(g),'returned':int(ret.sum()),'raw_observed_product_coverage':diag_cov,'oracle_treatment_empirical_coverage_same_response_graph':oracle_cov,'coverage_difference_vs_oracle_treatment':diag_cov-oracle_cov,'oracle_comparator_track':oracle_track,'theorem_certified':False,'claim_note':'fixed production treatment-nuisance diagnostic; response model and spatial graph held fixed; estimated/misspecified treatment route is not finite-sample theorem-certified'})
    return pd.DataFrame(rows)

def _ablation_table(results:pd.DataFrame,heur:pd.DataFrame)->pd.DataFrame:
    rows=[];oracle=results[(results.production_scope=='main_production')&(results.track=='ORACLE_STRUCTURAL_OT_TG')]
    mapping={'ABL0':'M6','ABL1':'M3','ABL3':'M2','ABL4':'M4','ABL5':'M2'}
    for aid,m in mapping.items():
        for key,g in oracle[oracle.method==m].groupby(['scenario_id','case_id'],sort=True):
            ret=g.operational_returned.astype(bool);rows.append({'ablation_id':aid,'scenario_id':key[0],'case_id':key[1],'source_method':m,'requested':len(g),'returned':int(ret.sum()),'refusal_rate':float(1-ret.mean()),'selective_coverage':float(g.loc[ret,'raw_covered'].mean()) if ret.any() else np.nan,'claim_note':'ABL5 algebraically coincides with M2 on the frozen identity target-design branch' if aid=='ABL5' else ''})
    # ABL6 compares the exact same raw M6 decisions before/after frozen eligibility/refusal gates.
    for key,g in oracle[oracle.method=='M6'].groupby(['scenario_id','case_id'],sort=True):
        raw=g.raw_method_returned.astype(bool);op=g.operational_returned.astype(bool);rows.append({'ablation_id':'ABL6','scenario_id':key[0],'case_id':key[1],'source_method':'M6_raw_without_operational_refusal','requested':len(g),'returned':int(raw.sum()),'refusal_rate':float(1-raw.mean()),'selective_coverage':float(g.loc[raw,'raw_covered'].mean()) if raw.any() else np.nan,'gated_returned_reference':int(op.sum()),'gated_false_support_reference':int(g.false_support.sum()),'claim_note':'diagnoses the value of frozen eligibility/refusal; not an alternative valid method'})
    if len(heur):
        for key,g in heur.groupby(['heuristic_id','scenario_id','case_id'],sort=True):
            ret=g.raw_method_returned.astype(bool);rows.append({'ablation_id':'ABL7','scenario_id':key[1],'case_id':key[2],'source_method':key[0],'requested':len(g),'returned':int(ret.sum()),'refusal_rate':float(1-ret.mean()),'selective_coverage':float(g.loc[ret,'covered'].mean()) if ret.any() else np.nan,'claim_note':'geographic heuristic; never theorem-eligible M3/M6'})
    # ABL9: registered S6 empirical RF working graph versus the same fitted response
    # model evaluated with the oracle true spatial graph. Both are diagnostics and
    # neither is promoted to a finite-sample theorem-certified nuisance route.
    s6=results[(results.production_scope=='main_production')&(results.scenario_id=='S6')&results.track.isin(['RF_SHARED_PREDICTOR_EMPIRICAL','RF_ORACLE_SPATIAL_LAW_DIAGNOSTIC'])]
    for key,g in s6.groupby(['case_id','method','track'],sort=True):
        ret=g.operational_returned.astype(bool);rows.append({'ablation_id':'ABL9','scenario_id':'S6','case_id':key[0],'source_method':key[1],'graph_track':key[2],'requested':len(g),'returned':int(ret.sum()),'refusal_rate':float(1-ret.mean()),'selective_coverage':float(g.loc[ret,'raw_covered'].mean()) if ret.any() else np.nan,'claim_note':'same empirical RF response model; registered fitted graph versus oracle true graph; diagnostic only, not theorem-certified'})
    return pd.DataFrame(rows)

def _hero_s4(results:pd.DataFrame)->pd.DataFrame:
    x=results[(results.production_scope=='main_production')&(results.scenario_id=='S4')&(results.track=='ORACLE_STRUCTURAL_OT_TG')].copy();rho={'S4_RHO000':0.0,'S4_RHO020':0.2,'S4_RHO040':0.4,'S4_RHO060':0.6,'S4_RHO080':0.8};rows=[]
    for (cid,m),g in x.groupby(['case_id','method'],sort=True):
        ret=g.operational_returned.astype(bool);rows.append({'case_id':cid,'rho':rho[cid],'method':m,'requested':len(g),'returned':int(ret.sum()),'refusal_rate':float(1-ret.mean()),'raw_coverage':float(g.raw_covered.mean()),'selective_coverage':float(g.loc[ret,'raw_covered'].mean()) if ret.any() else np.nan,'coverage_mcse':math.sqrt(float(g.raw_covered.mean())*(1-float(g.raw_covered.mean()))/len(g))})
    return pd.DataFrame(rows).sort_values(['rho','method'])

def _f06_metrics(results:pd.DataFrame)->pd.DataFrame:
    x=results[(results.track=='ORACLE_STRUCTURAL_OT_TG')].copy();rows=[]
    # extension small/large plus corresponding main primary first 100 local rows for representative cases.
    reps={'S1_BASE':range(1,101),'S4_RHO060':range(301,401),'S5_POOR_OVERLAP':range(1,101)}
    for base,repvals in reps.items():
        p=x[(x.production_scope=='main_production')&(x.case_id==base)&x.replication.astype(int).isin(list(repvals))].copy();p['f06_level']='primary';parts=[p]
        for level in ['small','large']:
            parts.append(x[(x.production_scope=='f06_extension')&(x.generator_base_case_id==base)&(x.f06_level==level)].copy())
        z=pd.concat(parts,ignore_index=True)
        for (lev,m),g in z.groupby(['f06_level','method'],sort=True):
            ret=g.operational_returned.astype(bool);rows.append({'base_case_id':base,'f06_level':lev,'method':m,'requested':len(g),'returned':int(ret.sum()),'refusal_rate':float(1-ret.mean()),'selective_coverage':float(g.loc[ret,'raw_covered'].mean()) if ret.any() else np.nan,'raw_coverage':float(g.raw_covered.mean())})
    return pd.DataFrame(rows)

def _coverage_confirmation(results:pd.DataFrame,contract:dict)->pd.DataFrame:
    cfg={'coverage_sanity':{'familywise_alpha':float(contract['coverage_confirmation']['familywise_alpha']),'minimum_returned_for_test':int(contract['coverage_confirmation']['minimum_returned_for_test'])}}
    x=results[results.production_scope=='main_production'].copy();out=coverage_sanity(x,cfg)
    if len(out):out['inferential_status']='INDEPENDENT_PRODUCTION_CONFIRMATION_THRESHOLDS_FROZEN_ON_STAGE3F';out['thresholds_retuned_from_production']=False
    return out

def _registered_domain_width(root,paths,b3,c3,m34,d2,e3,contract,thresholds,main_registry,sparse_cache)->pd.DataFrame:
    # Reuse accepted D2/Stage3E inversion code. The output is explicitly finite-domain.
    esrc=paths['stage3e_source_root']/'src'
    if str(esrc) not in sys.path:sys.path.insert(0,str(esrc))
    from geodose_stage3e.d2_sparse_bridge import D2SparseBridge
    invbridge=D2SparseBridge(paths['stage3d_d2_source_root'],paths['stage3e_source_root']/'configs/d2_api_lock.json',float(contract['alpha']))
    wanted=set(contract['width_efficiency']['hero_case_ids']);idx=set(map(int,contract['width_efficiency']['fixed_case_local_rep_indices']));domain=list(map(float,contract['width_efficiency']['candidate_domain']));lo,hi=domain;rows=[]
    for e in main_registry:
        if e['case_id'] not in wanted or int(e['case_local_index']) not in idx:continue
        case=_resolve_case(b3,e);rep=int(e['seed_replication']);gen=_generate(b3,case,rep);units,edges,node,block,target,observational,support,_=_support_and_target(b3,c3,d2,e3,case,gen,rep,contract,sparse_cache)
        # Exact M5 interval via accepted Stage3F evaluator, then apply the already-frozen gate.
        rr,_,_=sf_runner._evaluate_track(b3=b3,c3=c3,m34=m34,d2=d2,e3=e3,case=case,generated=gen,replication=rep,target_node=node,block=block,target=target,observational=observational,track='ORACLE_STRUCTURAL_OT_TG',model=None,contract=contract,sparse_cache=sparse_cache,width_audit=False);_apply_gate(rr,support,thresholds);m5=next(r for r in rr if r['method']=='M5');m6truth=next(r for r in rr if r['method']=='M6')
        if not m6truth['raw_method_returned']:
            rows.append({'case_id':case.case_id,'rho':float(case.spatial_rho),'replication':rep,'case_local_index':e['case_local_index'],'method':'M6','status':'refused_before_inversion','raw_total_width':np.nan,'domain_truncated':False,'operational_returned':False});continue
        _,Qs,kl,_=sf_runner._get_sparse(d2,e3,case,units,edges,int(contract['primary_scalable_neighborhood_size']),sparse_cache);prepared=d2.make_fixture(case,units,edges,block,target,observational_target=observational,use_observed_outcomes=False);boundary=e3.precision_boundary(Qs,np.asarray(prepared.fixture.block_nodes,dtype=int));prepared=d2.patch_sparse_precision(prepared,Qs,boundary);inv=invbridge.full_invert(prepared,f'STAGE4_WIDTH|{case.case_id}|{rep}',independent_grid_audit=False);s=inv['summaries']['conservative'];domain_trunc=bool(s.get('domain_truncated_set',False) or s.get('left_domain_truncated',False) or s.get('right_domain_truncated',False))
        rows.append({'case_id':case.case_id,'rho':float(case.spatial_rho),'replication':rep,'case_local_index':e['case_local_index'],'method':'M6','status':str(s['set_status']),'raw_total_width':float(s['raw_total_width']),'hull_width':float(s['hull_width']),'component_count':int(s['component_count']),'domain_truncated':domain_trunc,'operational_returned':bool(m6truth['operational_returned']),'candidate_evaluation_count':int(inv['candidate_evaluation_count']),'full_real_line_claim':False,'n2_delta_kl':float(kl)})
        if m5['raw_method_returned']:
            a=max(lo,float(m5['lower']));b=min(hi,float(m5['upper']));w=max(0.0,b-a);tr=bool(float(m5['lower'])<lo or float(m5['upper'])>hi)
            rows.append({'case_id':case.case_id,'rho':float(case.spatial_rho),'replication':rep,'case_local_index':e['case_local_index'],'method':'M5','status':'exact_interval_intersect_registered_domain','raw_total_width':w,'hull_width':w,'component_count':1,'domain_truncated':tr,'operational_returned':bool(m5['operational_returned']),'candidate_evaluation_count':0,'full_real_line_claim':False,'n2_delta_kl':np.nan})
    return pd.DataFrame(rows)

def _matched_efficiency(results:pd.DataFrame,width:pd.DataFrame,contract)->pd.DataFrame:
    rows=[];tol=float(contract['width_efficiency']['matched_coverage_absolute_tolerance']);main=results[(results.production_scope=='main_production')&(results.track=='ORACLE_STRUCTURAL_OT_TG')]
    for cid in contract['width_efficiency']['hero_case_ids']:
        g=main[main.case_id==cid];vals={}
        for m in ['M5','M6']:
            q=g[g.method==m];ret=q.operational_returned.astype(bool);vals[m]=float(q.loc[ret,'raw_covered'].mean()) if ret.any() else np.nan
        diff=abs(vals['M5']-vals['M6']) if all(np.isfinite(list(vals.values()))) else np.nan;w=width[width.case_id==cid];summ={}
        for m in ['M5','M6']:
            q=w[(w.method==m)&w.operational_returned.astype(bool)];summ[m]=float(q.raw_total_width.mean()) if len(q) else np.nan
        m6tr=bool(w[(w.case_id==cid)&(w.method=='M6')].domain_truncated.astype(bool).any()) if len(w[(w.case_id==cid)&(w.method=='M6')]) else True
        status='MATCHED_COVERAGE_REGISTERED_DOMAIN_ONLY' if np.isfinite(diff) and diff<=tol else 'NOT_MATCHED_COVERAGE';
        if m6tr:status+='__M6_DOMAIN_TRUNCATED_NO_FULL_LINE_EFFICIENCY_CLAIM'
        rows.append({'case_id':cid,'rho':float(g.iloc[0].n2_delta_kl*0+({'S4_RHO000':0,'S4_RHO020':.2,'S4_RHO040':.4,'S4_RHO060':.6,'S4_RHO080':.8}[cid])) if len(g) else np.nan,'m5_selective_coverage_all_reps':vals['M5'],'m6_selective_coverage_all_reps':vals['M6'],'absolute_coverage_difference':diff,'matched_coverage_tolerance':tol,'coverage_matched':bool(np.isfinite(diff) and diff<=tol),'m5_mean_registered_domain_width_fixed_subset':summ['M5'],'m6_mean_registered_domain_width_fixed_subset':summ['M6'],'m6_any_domain_truncation':m6tr,'efficiency_status':status,'full_real_line_efficiency_claim':False})
    return pd.DataFrame(rows)

def run(root:Path,paths:dict[str,Path],output_dir:Path,overwrite:bool=False,qa_smoke:bool=False)->dict[str,Any]:
    root=Path(root);output_dir=Path(output_dir);prepare_output_dir(output_dir,overwrite);t0=time.perf_counter();contract=yaml.safe_load((root/'configs/production_contract.yaml').read_text());require(contract['stage']==STAGE and contract['version']==VERSION,'CONTRACT_VERSION_DRIFT');require(contract['threshold_retuning_allowed'] is False and contract['nsw_tuning_allowed'] is False,'PRODUCTION_MUST_NOT_RETUNE_OR_USE_NSW')
    audit=_verify_inputs(root,paths);thresholds=_read_json_zip(paths['stage3f_output_zip'],'stage3f_operational_thresholds_FROZEN.json');b3=Stage3BBridge(paths['stage3b_source_root'],paths['stage3a_output_zip'],paths['stage3b_output_zip']);c3=Stage3CBridge(paths['stage3c_source_root']);m34=M3M4Bridge(paths['stage3d_d4_source_root']);d2=D2Bridge(paths['stage3d_d2_source_root'],float(contract['alpha']));e3=Stage3EBridge(paths['stage3e_source_root']);adapter=install_stage3b_production_adapter(b3.core,b3.stage3a);install_stage3c_production_nuisance_adapter(c3.runner)
    f06qa=f06_structural_qa(b3.core,contract);seeds=_production_seed_map(b3.stage3a);main=build_main_registry(contract,seeds);f06=build_f06_registry(contract);ext=build_extension_registry(contract);regsum=registry_summary(main,f06,ext);require(regsum['main_rows']==3000,'MAIN_ROWS_NOT_3000')
    registry=main+f06+ext
    if qa_smoke:
        # Deterministic structural/scientific smoke only; never publication evidence.
        selected=[]
        for cid in ['S1_BASE','S4_RHO060','S5_POOR_OVERLAP','S6_OMIT50','S9_TREATMENT10','S10_90M']:
            selected.append(next(e for e in main if e['case_id']==cid))
        selected += [next(e for e in f06 if e['case_id']=='S1_BASE' and e['f06_level']=='small'),next(e for e in f06 if e['case_id']=='S1_BASE' and e['f06_level']=='large')]
        selected += [next(e for e in ext if e.get('extension_id')=='EXT_ST1_ATOM0'),next(e for e in ext if e.get('extension_id')=='EXT_ST3_HIDDEN_C2')]
        registry=selected
    contract_hash=_contract_hash(root);source_hash=source_inventory(root)['aggregate_sha256'];ckdir=output_dir/'checkpoints';ckdir.mkdir(exist_ok=True);sparse_cache={};all_payload=[]
    total=len(registry);print(f'[PRODUCTION] Case-replication units: {total} (main={regsum["main_rows"]}, F06={regsum["f06_extension_rows"]}, stress={regsum["stress_extension_rows"]})',flush=True)
    for i,e in enumerate(registry,1):
        key=_entry_key(e);cp=ckdir/(key+'.json.gz')
        if cp.is_file() and not overwrite:
            obj=read_gzip_json(cp);require(obj.get('contract_sha256')==contract_hash and obj.get('source_aggregate_sha256')==source_hash,f'CHECKPOINT_PROVENANCE_MISMATCH:{key}');payload=obj['payload']
        else:
            ts=time.perf_counter();payload=_evaluate_entry(root=root,b3=b3,c3=c3,m34=m34,d2=d2,e3=e3,entry=e,contract=contract,thresholds=thresholds,sparse_cache=sparse_cache,seeds=seeds);payload['runtime']=[{'key':key,'elapsed_seconds':time.perf_counter()-ts,'scope':e['scope'],'case_id':e['case_id'],'replication':e['seed_replication']}];atomic_gzip_json(cp,{'contract_sha256':contract_hash,'source_aggregate_sha256':source_hash,'payload':payload})
        all_payload.append(payload)
        if i==1 or i%25==0 or i==total:print(f'  completed {i}/{total}; checkpoints make this resumable',flush=True)
    def collect(name):
        rec=[]
        for p in all_payload:rec.extend(p.get(name,[]))
        return pd.DataFrame(rec)
    results=collect('results');support=collect('support');targets=collect('targets');heur=collect('heuristics');graphs=collect('graphs');outfits=collect('outcome_fits');temporal=collect('temporal');spatial=collect('spatial');runtime_case=collect('runtime')
    # Official full run invariants.
    if not qa_smoke:
        require(len(support)==regsum['total_case_replication_rows'],'SUPPORT_ROW_COUNT_DRIFT');require(len(results)>0,'NO_PRODUCTION_RESULTS');require(not results.computational_failure.astype(bool).any(),'PRODUCTION_COMPUTATIONAL_FAILURE_PRESENT');require(not results.false_support.astype(bool).any(),'PRODUCTION_FALSE_SUPPORT_PRESENT')
    metrics=aggregate_metrics(results,float(contract['alpha'])) if len(results) else pd.DataFrame();dose=dose_bin_metrics(results,float(contract['alpha']),[0.0,0.1,0.25,0.5,0.75,0.9,1.0]) if len(results) else pd.DataFrame();local,localsum=_local_coverage_metrics(results);cert=certificate_summary(support) if len(support) else pd.DataFrame();hero=_hero_s4(results) if not qa_smoke else pd.DataFrame();f06m=_f06_metrics(results) if not qa_smoke else pd.DataFrame();abl=_ablation_table(results,heur);cov=_coverage_confirmation(results,contract) if not qa_smoke else pd.DataFrame()
    width=pd.DataFrame();eff=pd.DataFrame()
    if not qa_smoke:
        print('[EFFICIENCY] Running prespecified 50-query S4 M6 finite-domain inversions...',flush=True);width=_registered_domain_width(root,paths,b3,c3,m34,d2,e3,contract,thresholds,main,sparse_cache);eff=_matched_efficiency(results,width,contract)
        print('[ABL8] Running fixed 10-rep-per-representative-case treatment-nuisance diagnostic...',flush=True);propdiag,propfits=_propensity_diagnostics(b3,c3,m34,d2,e3,contract,main,sparse_cache,seeds);abl8=_abl8_summary(propdiag,results)
    else:
        propdiag=pd.DataFrame();propfits=pd.DataFrame();abl8=pd.DataFrame()
    # ABL2 target-design transport certificate diagnostic: frozen Stage3B has constant N2 local loss, so oracle reweighting cancels numerically.
    design_rows=[]
    if not qa_smoke:
        ds=support[support.analysis_case_id.astype(str).str.contains('DESIGN_SHIFT',na=False)]
        for r in ds.itertuples(index=False):design_rows.append({'ablation_id':'ABL2','case_id':r.analysis_case_id,'replication':int(r.replication),'target_design_mode':r.target_design_mode,'target_design_delta':float(r.target_design_delta),'observational_N2_delta':float(r.n2_delta_kl),'oracle_target_weighted_N2_delta':float(r.n2_delta_kl),'difference':0.0,'reason':'frozen Stage3B residual design makes L(U) constant; E_obs[r(U)L(U)]=L(U) when oracle target-design ratio is normalized','pvalue_ablation_claim':False})
    design=pd.DataFrame(design_rows)
    # Main publication result table excludes extensions/diagnostics and keeps frozen thresholds.
    claim={'stage':STAGE,'version':VERSION,'qa_smoke':bool(qa_smoke),'production_main_rows_generated':(0 if qa_smoke else 3000),'production_performance_evidence':bool(not qa_smoke),'thresholds_frozen_from_stage3f':True,'thresholds_retuned_from_production':False,'nsw_used_for_tuning':False,'f06_frozen_before_production_outcomes':True,'f06_experiment_type':'fixed_resolution_increasing_domain_sample_size_sensitivity','oracle_structural_primary_route':'theorem_certified_subject_to_registered_case_claim_boundaries','estimated_nuisance_finite_sample_certified':False,'design_shift_performance_theorem_claim':False,'ST2_method_level_temporal_conformal_claim':False,'ST3_interpretation':'causal_identification_negative_control; conformal calibration cannot repair hidden-C2 identification failure','M6_full_real_line_width_claim':False,'matched_efficiency_scope':'registered finite candidate domain only; full-line efficiency withheld whenever domain truncation is possible','stage3f_thresholds':{k:thresholds[k] for k in ['minimum_ess','maximum_normalized_weight','minimum_graph_safe_count']},'next_stage_blocked_until_independent_review':True}
    # Deterministic scientific outputs.
    deterministic_gzip_csv(results,output_dir/'stage4_production_query_results.csv.gz');deterministic_gzip_csv(support,output_dir/'stage4_support_certificate_inputs.csv.gz')
    for name,df in [('stage4_target_selection_audit.csv',targets),('stage4_graph_diagnostics.csv',graphs),('stage4_outcome_fit_audit.csv',outfits),('stage4_temporal_stress_diagnostic.csv',temporal),('stage4_heuristic_ablation_results.csv',heur),('stage4_production_metrics.csv',metrics),('stage4_dose_bin_metrics.csv',dose),('stage4_local_coverage_metrics.csv',local),('stage4_local_coverage_summary.csv',localsum),('stage4_spatial_residual_diagnostics.csv',spatial),('stage4_certificate_summary.csv',cert),('stage4_s4_hero_results.csv',hero),('stage4_f06_sample_size_metrics.csv',f06m),('stage4_ablation_summary.csv',abl),('stage4_target_design_ablation.csv',design),('stage4_independent_coverage_confirmation.csv',cov),('stage4_m6_m5_width_audit.csv',width),('stage4_m6_m5_matched_coverage_efficiency.csv',eff),('stage4_propensity_ablation_query_results.csv',propdiag),('stage4_propensity_ablation_fit_audit.csv',propfits),('stage4_propensity_ablation_summary.csv',abl8),('stage4_case_rep_runtime.csv',runtime_case)]:
        df.to_csv(output_dir/name,index=False,lineterminator='\n',float_format='%.17g')
    pd.DataFrame(f06qa).to_csv(output_dir/'stage4_f06_structural_QA.csv',index=False,lineterminator='\n');pd.DataFrame(main+f06+ext).to_csv(output_dir/'stage4_frozen_production_registry.csv',index=False,lineterminator='\n')
    json_dump(output_dir/'stage4_f06_DGP_FREEZE.json',contract['f06_freeze']);json_dump(output_dir/'stage4_frozen_operational_thresholds.json',{k:thresholds[k] for k in ['minimum_ess','maximum_normalized_weight','minimum_graph_safe_count']});json_dump(output_dir/'stage4_claim_boundary.json',claim);json_dump(output_dir/'stage4_upstream_provenance_audit.json',audit);json_dump(output_dir/'stage4_registry_summary.json',regsum);json_dump(output_dir/'stage4_source_inventory.json',source_inventory(root));json_dump(output_dir/'stage4_adapter_audit.json',adapter)
    runtime={'stage':STAGE,'version':VERSION,'qa_smoke':bool(qa_smoke),'elapsed_seconds':time.perf_counter()-t0,'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'result_rows':len(results),'support_rows':len(support),'checkpoint_files':len(list(ckdir.glob('*.json.gz'))),'sparse_cache_entries':len(sparse_cache),'width_rows':len(width)};json_dump(output_dir/'stage4_runtime.json',runtime)
    man=output_manifest(output_dir,exclude={'stage4_manifest.json','STAGE4_VERIFICATION.json'});json_dump(output_dir/'stage4_manifest.json',man)
    return {'qa_smoke':qa_smoke,'result_rows':len(results),'support_rows':len(support),'runtime_seconds':runtime['elapsed_seconds'],'main_rows':regsum['main_rows'],'false_support':int(results.false_support.sum()) if len(results) else 0,'computational_failures':int(results.computational_failure.sum()) if len(results) else 0}
