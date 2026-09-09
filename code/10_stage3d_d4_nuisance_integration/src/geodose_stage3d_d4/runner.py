from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
import sklearn
import joblib
import xgboost

from . import __version__
from .data import load_stage3a_seed_registry, load_stage3b, case_row, target_row, frozen_target_unit
from .d2_runtime import D2SourceAdapter
from .fixture import DEFAULT_TARGET_NODE, block_connected, choose_block, edge_degrees
from .io import (
    D4Error, ensure_empty_output_dir, read_csv_from_zip, read_json_from_zip,
    source_hash_inventory_from_output_zip, verify_source_tree, verify_zip,
    write_csv_gz, write_json,
)
from .m3m4 import evaluate_m3m4, graph_inputs
from .nuisance import fit_estimated_treatment
from .provenance import source_inventory


def _json(path: Path) -> dict[str,Any]: return json.loads(path.read_text(encoding='utf-8'))
def _yaml(path: Path) -> dict[str,Any]: return yaml.safe_load(path.read_text(encoding='utf-8'))


def _verification_check_pass(value: Any) -> bool:
    """Normalize the accepted historical verification schemas used by Stage3B onward.

    Older frozen stages recorded check results as strings such as ``"pass"``; newer
    stages use booleans. Unknown shapes fail closed.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"pass", "passed", "true", "ok", "verified", "verified_complete"}
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value) == 1
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value) and float(value) == 1.0)
    return False


def _verification_failure_summary(obj: dict[str, Any]) -> dict[str, Any]:
    """Return a fail-closed, schema-tolerant failure summary for a frozen verifier JSON.

    Accepted upstream stages legitimately use three different ``failed_checks`` schemas:
    * Stage3B: field absent; ``checks`` contains string ``pass`` values.
    * Stage3C: empty/nonempty mapping.
    * D2 and D3 gates: integer failure count.
    This helper supports those frozen schemas without modifying upstream artifacts.
    """
    raw = obj.get('failed_checks', None)
    source = 'failed_checks'
    if raw is None:
        failed_names = obj.get('failed_check_names', None)
        if isinstance(failed_names, (list, tuple, set)):
            return {'count': len(failed_names), 'source': 'failed_check_names', 'raw_type': type(failed_names).__name__}
        checks = obj.get('checks', None)
        if isinstance(checks, dict) and checks:
            failed = [str(k) for k,v in checks.items() if not _verification_check_pass(v)]
            return {'count': len(failed), 'source': 'checks_inferred', 'raw_type': 'missing', 'failed_names': failed}
        raise D4Error('Verification JSON has no interpretable failure indicator/check dictionary')
    if isinstance(raw, (bool, np.bool_)):
        count = int(bool(raw))
    elif isinstance(raw, (int, np.integer)):
        count = int(raw)
    elif isinstance(raw, (float, np.floating)):
        if not np.isfinite(raw) or float(raw) < 0 or not float(raw).is_integer():
            raise D4Error(f'Invalid numeric failed_checks value: {raw!r}')
        count = int(raw)
    elif isinstance(raw, dict):
        count = len(raw)
    elif isinstance(raw, (list, tuple, set)):
        count = len(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if text == '':
            count = 0
        else:
            try:
                count = int(text)
            except ValueError as exc:
                raise D4Error(f'Unrecognized failed_checks string: {raw!r}') from exc
    else:
        raise D4Error(f'Unrecognized failed_checks schema: {type(raw).__name__}')
    if count < 0:
        raise D4Error(f'Negative failed_checks value: {count}')
    return {'count': count, 'source': source, 'raw_type': type(raw).__name__}


def _verify_inputs(package_root: Path, paths: dict[str,Path], expected: dict[str,str]) -> dict[str,Any]:
    audit={'archives':{},'source_trees':{}}
    for key,sha in expected.items():
        if key not in paths: raise D4Error(f'Missing resolved input key {key}')
        audit['archives'][key]=verify_zip(paths[key],sha)
    # Accepted source inventories are authoritative for local source trees.
    s3c_inv=source_hash_inventory_from_output_zip(paths['stage3c_output_zip'],'stage3c_source_hashes.json')
    d2_inv=source_hash_inventory_from_output_zip(paths['stage3d_d2_output_zip'],'stage3d_d2_source_hashes.json')
    s3c=verify_source_tree(paths['stage3c_source_root'],s3c_inv)
    d2=verify_source_tree(paths['stage3d_d2_source_root'],d2_inv)
    if not s3c['pass']: raise D4Error(f'Stage3C local source tree mismatch: {s3c}')
    if not d2['pass']: raise D4Error(f'D2 local source tree mismatch: {d2}')
    audit['source_trees']['stage3c']=s3c; audit['source_trees']['stage3d_d2']=d2
    # Verify every vendored M3/M4 engine byte against the accepted M4 source inventory.
    m4src=read_json_from_zip(paths['stage3d_d3_m4_output_zip'],'stage3d_d3_m4_source_hashes.json')
    m4map={str(x['path']).replace('\\','/'):str(x['sha256']) for x in m4src.get('files',[])}
    vend=[]
    for fp in sorted((package_root/'vendor'/'geodose_stage3d_d3_m4').glob('*.py')):
        rel='src/geodose_stage3d_d3_m4/'+fp.name
        expected_sha=m4map.get(rel)
        from .io import sha256_file
        observed=sha256_file(fp)
        vend.append({'path':rel,'expected':expected_sha,'observed':observed,'pass':expected_sha is not None and expected_sha==observed})
    if not vend or not all(x['pass'] for x in vend): raise D4Error(f'Vendored accepted M3/M4 byte audit failed: {vend}')
    audit['vendor_m3_m4_byte_audit']={'file_count':len(vend),'all_pass':True,'files':vend}
    # Verify upstream stage statuses and D3 claim boundary before new-data work.
    checks={
        'stage3b':read_json_from_zip(paths['stage3b_output_zip'],'STAGE3B_VERIFICATION.json'),
        'stage3c':read_json_from_zip(paths['stage3c_output_zip'],'STAGE3C_REFERENCE_VERIFICATION.json'),
        'd2':read_json_from_zip(paths['stage3d_d2_output_zip'],'STAGE3D_D2_VERIFICATION.json'),
        'm3':read_json_from_zip(paths['stage3d_d3_m3_output_zip'],'STAGE3D_D3_M3_VERIFICATION.json'),
        'm4':read_json_from_zip(paths['stage3d_d3_m4_output_zip'],'STAGE3D_D3_M4_VERIFICATION.json'),
        'd3_integration':read_json_from_zip(paths['stage3d_d3_integration_output_zip'],'STAGE3D_D3_INTEGRATION_VERIFICATION.json'),
    }
    status_summaries={}
    for name,obj in checks.items():
        failure=_verification_failure_summary(obj)
        status_summaries[name]=failure
        if str(obj.get('status'))!='verified_complete' or int(failure['count'])!=0:
            raise D4Error(f"Accepted upstream {name} is not verified_complete: status={obj.get('status')!r}, normalized_failed_checks={failure['count']} ({failure['source']})")
    claim=read_json_from_zip(paths['stage3d_d3_integration_output_zip'],'stage3d_d3_claim_boundary.json')
    if bool(claim.get('M6_new_data_pilot_engine_ready')):
        raise D4Error('D3 claim boundary unexpectedly says M6 new-data engine already ready')
    audit['accepted_statuses']={k:{'status':v.get('status'),'check_count':len(v.get('checks',{})) if isinstance(v.get('checks'),dict) else v.get('check_count'),'failed_checks_raw':v.get('failed_checks'),'failed_checks_normalized':status_summaries[k]} for k,v in checks.items()}
    audit['d3_claim_boundary_before_D4']=claim
    return audit


def _graph_audit(data, case_id: str, block: tuple[int,...], graph_mode: str) -> dict[str,Any]:
    units,edges=graph_inputs(data,case_id,graph_mode)
    connected=block_connected(block,edges)
    n=int(units.node_index.max())+1; deg=edge_degrees(edges,n); bdeg=deg[np.asarray(block,dtype=int)]
    return {'case_id':case_id,'graph_mode':graph_mode,'edge_count':int(len(edges)),'block_connected':bool(connected),'block_nodes':','.join(map(str,block)),'block_degree_min':int(bdeg.min()),'block_degree_max':int(bdeg.max()),'block_degree_sum':int(bdeg.sum())}


def _variant_parts(variant: str) -> tuple[str,str]:
    parts=str(variant).split('_')
    if len(parts)!=2 or parts[0] not in {'OT','ET'} or parts[1] not in {'TG','FG'}:
        raise D4Error(f'Invalid D4 nuisance variant {variant}')
    return parts[0],parts[1]


def _m3_invariance(trace: pd.DataFrame, tol: float) -> pd.DataFrame:
    rows=[]
    m3=trace[trace.method=='M3'].copy()
    for (case_id,dose,graph,y),g in m3.groupby(['case_id','target_dose','graph_mode','candidate_y'],sort=True):
        by={str(r.treatment_mode):float(r.p_value) for r in g.itertuples(index=False)}
        if {'OT','ET'}.issubset(by):
            d=abs(by['OT']-by['ET']); rows.append({'case_id':case_id,'target_dose':dose,'graph_mode':graph,'candidate_y':y,'p_OT':by['OT'],'p_ET':by['ET'],'abs_difference':d,'pass':d<=tol})
    out=pd.DataFrame(rows)
    if out.empty or not bool(out['pass'].all()): raise D4Error('M3 treatment-invariance gate failed')
    return out


def _nuisance_deltas(trace: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for method in ['M4','M6']:
        df=trace[trace.method==method]
        for (case_id,dose,y),g in df.groupby(['case_id','target_dose','candidate_y'],sort=True):
            lookup={(str(r.treatment_mode),str(r.graph_mode)):float(r.p_value) for r in g.itertuples(index=False)}
            if ('OT','TG') in lookup and ('ET','TG') in lookup:
                rows.append({'method':method,'case_id':case_id,'target_dose':dose,'candidate_y':y,'contrast':'treatment_ET_vs_OT_at_TG','abs_p_difference':abs(lookup[('ET','TG')]-lookup[('OT','TG')])})
            if ('OT','TG') in lookup and ('OT','FG') in lookup:
                rows.append({'method':method,'case_id':case_id,'target_dose':dose,'candidate_y':y,'contrast':'graph_FG_vs_TG_at_OT','abs_p_difference':abs(lookup[('OT','FG')]-lookup[('OT','TG')])})
    return pd.DataFrame(rows)


def run(package_root: Path, paths: dict[str,Path], output_dir: Path, overwrite: bool=False) -> dict[str,Any]:
    t0=time.perf_counter(); cfg=package_root/'configs'
    expected=_json(cfg/'expected_upstream_hashes.json'); contract=_yaml(cfg/'d4_contract.yaml'); cases_cfg=_yaml(cfg/'case_registry.yaml'); tol=_yaml(cfg/'tolerances.yaml'); api_lock=_json(cfg/'d2_api_lock.json')
    ensure_empty_output_dir(output_dir,overwrite)

    input_audit=_verify_inputs(package_root,paths,expected)
    write_json(output_dir/'stage3d_d4_input_audit.json',input_audit)

    data=load_stage3b(paths['stage3b_output_zip']); seeds=load_stage3a_seed_registry(paths['stage3a_output_zip'])
    # D4 exact-source wiring intentionally excludes S9 measurement-error cases. For every
    # registered D4 case, observed and latent oracle outcomes must coincide so the accepted
    # D2 residual-law specialization is not silently asked to model extra measurement noise.
    align_rows=[]
    for spec in cases_cfg['cases']:
        cid=str(spec['case_id']); dose=float(spec['target_dose'])
        cu=data.units[data.units.case_id.astype(str)==cid].copy()
        if 'Y_true_at_A' not in cu.columns or 'Y_observed_at_A' not in cu.columns:
            raise D4Error(f'Stage3B exact-wiring case {cid} lacks Y_true_at_A/Y_observed_at_A')
        cal_err=float(np.max(np.abs(cu['Y_true_at_A'].to_numpy(float)-cu['Y_observed_at_A'].to_numpy(float))))
        target_unit=frozen_target_unit(cu,DEFAULT_TARGET_NODE)
        tr=target_row(data,cid,str(target_unit['unit_id']),dose)
        tgt_err=abs(float(tr['Y_true_at_A_star'])-float(tr['Y_observed_at_A_star']))
        ok=cal_err<=1e-14 and tgt_err<=1e-14
        align_rows.append({'case_id':cid,'target_dose':dose,'target_node_index':int(target_unit['node_index']),'target_unit_id':str(target_unit['unit_id']),'max_abs_calibration_observed_minus_true':cal_err,'abs_target_observed_minus_true':tgt_err,'pass':ok})
        if not ok: raise D4Error(f'D4 exact-source gate excludes measurement-error outcomes; {cid}/dose={dose} has calibration_error={cal_err}, target_error={tgt_err}')
    pd.DataFrame(align_rows).drop_duplicates().to_csv(output_dir/'stage3d_d4_outcome_alignment_audit.csv',index=False,lineterminator='\n',float_format='%.17g')
    accepted_d2=read_csv_from_zip(paths['stage3d_d2_output_zip'],'stage3d_d2_candidate_pvalue_trace.csv.gz')

    d2tol={'score_abs_tolerance':1e-12,'score_rel_tolerance':1e-12}
    adapter=D2SourceAdapter(paths['stage3d_d2_source_root'],float(contract['alpha']),d2tol,api_lock)
    replay_cfg=cases_cfg['accepted_replay']
    replay_main,api_meta=adapter.accepted_replay_fixture(
        accepted_d2,str(replay_cfg['d1_evaluation_id']),str(replay_cfg['d2_fixture_id']),
        [float(x) for x in replay_cfg['candidates']],'S4'
    )
    dup_cfg=replay_cfg['duplicate_replay']
    replay_dup,dup_meta=adapter.accepted_replay_fixture(
        accepted_d2,str(dup_cfg['d1_evaluation_id']),str(dup_cfg['d2_fixture_id']),
        [float(x) for x in dup_cfg['candidates']],'S4'
    )
    replay=pd.concat([replay_main,replay_dup],ignore_index=True)
    replay['pass']=(replay['abs_p_error']<=float(tol['accepted_d2_pvalue_replay_abs'])) & (replay['abs_randomized_p_error']<=float(tol['accepted_d2_pvalue_replay_abs'])) & (replay['abs_tie_uniform_error']<=float(tol['accepted_d2_pvalue_replay_abs'])) & replay['accept_match'].astype(bool) & replay['randomized_accept_match'].astype(bool) & replay['distinct_states_match'].astype(bool)
    expected_dup_states=int(dup_cfg['expected_distinct_states'])
    got_dup=replay.loc[replay.fixture_id.astype(str)==str(dup_cfg['d2_fixture_id']),'accepted_distinct_states']
    duplicate_state_pass=(len(got_dup)==len(dup_cfg['candidates']) and bool((got_dup.astype(int)==expected_dup_states).all()))
    if not bool(replay['pass'].all()) or not duplicate_state_pass:
        raise D4Error('Source-level D2 accepted replay/duplicate-quotient regression failed; no new-data computation authorized')
    replay.to_csv(output_dir/'stage3d_d4_d2_source_replay.csv',index=False,lineterminator='\n',float_format='%.17g')
    write_json(output_dir/'stage3d_d4_d2_runtime_api.json',{
        'primary_replay':api_meta,'duplicate_replay':dup_meta,'duplicate_expected_distinct_states':expected_dup_states,
        'duplicate_distinct_state_check_passed':duplicate_state_pass,
    })

    # Expected refusal smoke tests: these are fail-closed software checks, not publication evidence.
    expected_refusal_rows=[]
    for spec in cases_cfg.get('expected_refusals',[]):
        cid=str(spec['case_id']); dose=float(spec['target_dose']); block=tuple(int(x) for x in spec['block_nodes'])
        crow=case_row(data,cid); scenario=str(crow['scenario_id'])
        passed=False; detail=''
        try:
            adapter.prepare_new_case(data,cid,scenario,dose,block,str(spec['treatment_mode']),str(spec['graph_mode']),None)
            detail='unexpectedly accepted'
        except Exception as exc:
            detail=f'{type(exc).__name__}: {exc}'
            token=str(spec.get('expected_reason_contains','')).lower()
            passed=(not token) or (token in detail.lower())
        expected_refusal_rows.append({
            'refusal_id':str(spec['refusal_id']),'case_id':cid,'target_dose':dose,
            'treatment_mode':str(spec['treatment_mode']),'graph_mode':str(spec['graph_mode']),
            'block_nodes':','.join(map(str,block)),'expected_reason_contains':str(spec.get('expected_reason_contains','')),
            'pass':bool(passed),'detail':detail,'purpose':str(spec.get('purpose','')),
        })
    expected_refusals=pd.DataFrame(expected_refusal_rows)
    if expected_refusals.empty or not bool(expected_refusals['pass'].astype(bool).all()):
        raise D4Error(f'Expected refusal gate failed: {expected_refusal_rows}')
    expected_refusals.to_csv(output_dir/'stage3d_d4_expected_refusal_audit.csv',index=False,lineterminator='\n')

    # Fit ET once per Stage3B case using its frozen scenario/replication-1 seed and nuisance-training role only.
    fits={}; fit_rows=[]
    for c in cases_cfg['cases']:
        cid=str(c['case_id'])
        if cid in fits: continue
        crow=case_row(data,cid); cu=data.units[data.units.case_id.astype(str)==cid].copy()
        fit=fit_estimated_treatment(paths['stage3c_source_root'],seeds,cu,str(crow['scenario_id']))
        fits[cid]=fit
        fit_rows.append({'case_id':cid,'scenario_id':str(crow['scenario_id']),'fit_id':fit.fit_id,'frozen_nuisance_seed':fit.frozen_nuisance_seed,'derived_propensity_seed':fit.derived_propensity_seed,'training_data_hash':fit.training_data_hash,'n_train':fit.n_train})
    pd.DataFrame(fit_rows).to_csv(output_dir/'stage3d_d4_estimated_treatment_fit_audit.csv',index=False,lineterminator='\n')

    trace_rows=[]; source_rows=[]; graph_rows=[]; prep_rows=[]; refusal_rows=[]
    candidate_values=[float(x) for x in contract['candidate_test_values']]
    for c in cases_cfg['cases']:
        cid=str(c['case_id']); dose=float(c['target_dose']); block=choose_block(data,cid,str(c['block_rule'])); crow=case_row(data,cid); scenario=str(crow['scenario_id'])
        modes=sorted({_variant_parts(v)[1] for v in c['nuisance_variants']})
        for gm in modes:
            ga=_graph_audit(data,cid,block,gm); graph_rows.append(ga)
            if not ga['block_connected']: raise D4Error(f'D4 selected disconnected {gm} block for {cid}')
        for variant in c['nuisance_variants']:
            tm,gm=_variant_parts(variant); fit=fits[cid] if tm=='ET' else None
            # Build accepted source-level M6 new-data evaluator once per case/variant.
            try:
                m6eval,pmeta=adapter.prepare_new_case(data,cid,scenario,dose,block,tm,gm,fit)
            except Exception as exc:
                refusal_rows.append({'case_id':cid,'target_dose':dose,'variant':variant,'method':'M6','candidate_y':np.nan,'refusal_code':'D4_M6_PREPARE_FAILED','detail':str(exc)})
                raise
            prep_rows.append({'case_id':cid,'scenario_id':scenario,'target_dose':dose,'variant':variant,'treatment_mode':tm,'graph_mode':gm,'block_nodes':','.join(map(str,block)),'metadata_json':json.dumps(pmeta,sort_keys=True,default=str)})
            for y in candidate_values:
                mm=evaluate_m3m4(package_root,data,paths['stage3c_source_root'],cid,dose,block,tm,gm,y,fit,tie_tolerance=1e-12)
                for method,p in [('M3',mm.m3_p),('M4',mm.m4_p)]:
                    trace_rows.append({'case_id':cid,'scenario_id':scenario,'target_dose':dose,'variant':variant,'treatment_mode':tm,'graph_mode':gm,'candidate_y':y,'candidate_hex':float(y).hex(),'method':method,'p_value':float(p),'included':bool(float(p)>float(contract['alpha'])),'source':'accepted_D3_M3' if method=='M3' else 'accepted_D3_M4','state_count':mm.state_count})
                for j,(s,q4,lr,res) in enumerate(zip(mm.m3_source_marginal,mm.m4_q,mm.log_dose_ratio,mm.source_residuals)):
                    source_rows.append({'case_id':cid,'target_dose':dose,'variant':variant,'candidate_y':y,'source_index':j,'source_node':int(block[j]),'m3_spatial_mass':float(s),'log_dose_ratio':float(lr),'m4_mass':float(q4),'source_residual':float(res)})
                try: m6=adapter.evaluate_new_candidate(m6eval,y)
                except Exception as exc:
                    refusal_rows.append({'case_id':cid,'target_dose':dose,'variant':variant,'method':'M6','candidate_y':y,'refusal_code':'D4_M6_CANDIDATE_FAILED','detail':str(exc)})
                    raise
                trace_rows.append({'case_id':cid,'scenario_id':scenario,'target_dose':dose,'variant':variant,'treatment_mode':tm,'graph_mode':gm,'candidate_y':y,'candidate_hex':float(y).hex(),'method':'M6','p_value':float(m6['conservative_p']),'included':bool(m6['conservative_accept']),'source':'accepted_D2_source_CandidateEvaluator_new_case','state_count':np.nan})

    trace=pd.DataFrame(trace_rows); weights=pd.DataFrame(source_rows); graphs=pd.DataFrame(graph_rows); prep=pd.DataFrame(prep_rows)
    write_csv_gz(trace,output_dir/'stage3d_d4_candidate_trace.csv.gz'); write_csv_gz(weights,output_dir/'stage3d_d4_m3_m4_source_weight_audit.csv.gz')
    graphs.to_csv(output_dir/'stage3d_d4_graph_variant_audit.csv',index=False,lineterminator='\n'); prep.to_csv(output_dir/'stage3d_d4_m6_prepare_audit.csv',index=False,lineterminator='\n')

    # Mechanical M4 recomputation from the only permitted interface d_j*s_j(y).
    rec=[]
    for keys,g in weights.groupby(['case_id','target_dose','variant','candidate_y'],sort=True):
        logd=g.log_dose_ratio.to_numpy(float); s=g.m3_spatial_mass.to_numpy(float); finite=np.isfinite(logd)&(s>0)
        if not finite.any(): raise D4Error(f'All M4 product masses zero for {keys}')
        z=np.full(len(g),-np.inf); z[finite]=np.log(s[finite])+logd[finite]; m=float(np.max(z)); raw=np.exp(z-m); q=raw/raw.sum()
        err=float(np.max(np.abs(q-g.m4_mass.to_numpy(float))))
        rec.append({'case_id':keys[0],'target_dose':keys[1],'variant':keys[2],'candidate_y':keys[3],'max_abs_q4_recompute_error':err,'pass':err<=float(tol['m4_recompute_abs'])})
    m4rec=pd.DataFrame(rec)
    if not bool(m4rec['pass'].all()): raise D4Error('M4 d*s one-normalization recomputation failed')
    m4rec.to_csv(output_dir/'stage3d_d4_m4_recompute_audit.csv',index=False,lineterminator='\n',float_format='%.17g')

    inv=_m3_invariance(trace,float(tol['m3_treatment_invariance_abs'])); inv.to_csv(output_dir/'stage3d_d4_m3_treatment_invariance.csv',index=False,lineterminator='\n',float_format='%.17g')
    deltas=_nuisance_deltas(trace); deltas.to_csv(output_dir/'stage3d_d4_nuisance_delta_diagnostics.csv',index=False,lineterminator='\n',float_format='%.17g')

    # Cross-check accepted M4 structural gate and D3 exact integration remained hash-locked inputs.
    m4_struct=read_csv_from_zip(paths['stage3d_d3_m4_output_zip'],'stage3d_d3_m4_s4_structural_summary.csv')
    d3_red=read_csv_from_zip(paths['stage3d_d3_integration_output_zip'],'stage3d_d3_reduction_status.csv')
    regression={
        'accepted_M4_nonfactorization_all':bool(m4_struct['nonfactorization_detected'].astype(bool).all()),
        'accepted_M4_max_delta_fact':float(m4_struct['delta_fact_logosc'].max()),
        'R1_status':str(d3_red.loc[d3_red.reduction_id=='R1','status'].iloc[0]),
        'R2_status':str(d3_red.loc[d3_red.reduction_id=='R2','status'].iloc[0]),
        'R3_status':str(d3_red.loc[d3_red.reduction_id=='R3','status'].iloc[0]),
        'R4_status':str(d3_red.loc[d3_red.reduction_id=='R4','status'].iloc[0]),
    }
    if regression['R3_status']!='DEFERRED_STAGE3E': raise D4Error('R3 must remain deferred to Stage3E')
    write_json(output_dir/'stage3d_d4_upstream_reduction_regression.json',regression)

    pd.DataFrame(refusal_rows,columns=['case_id','target_dose','variant','method','candidate_y','refusal_code','detail']).to_csv(output_dir/'stage3d_d4_refusal_log.csv',index=False,lineterminator='\n')
    case_summary=trace.groupby(['case_id','target_dose','variant','method'],sort=True).agg(candidate_count=('candidate_y','size'),p_min=('p_value','min'),p_max=('p_value','max'),included_count=('included','sum')).reset_index()
    case_summary.to_csv(output_dir/'stage3d_d4_case_summary.csv',index=False,lineterminator='\n',float_format='%.17g')

    claim={
        'stage':'Stage3D_D4','source_level_M6_new_data_candidate_adapter_small_exact_blocks':True,
        'accepted_D2_source_replay_required_and_passed':True,'G1_G2_G3_reimplemented':False,
        'outcome_mean_scale_regime':'oracle_Stage3B_truth_for_D4_wiring_isolation',
        'nuisance_variants_exercised':['OT_TG','ET_TG','OT_FG','ET_FG'],
        'ET_definition':'frozen_Stage3C_MixedPropensityModel_nuisance_training_only',
        'FG_definition':'accepted_Stage3B_fitted_edge_set_with_degrees_recomputed; oracle_rho_and_scale_retained',
        'estimated_nuisance_exact_coverage_claim':False,'full_new_data_prediction_set_inversion_ready':False,
        'pilot_scale_ready':False,'pilot_run':False,'production_run':False,'publication_performance_evidence':False,
        'N2_N3_implemented':False,'R3_status':'DEFERRED_STAGE3E',
        'candidate_smoke_grid':candidate_values,
        'important_reduction_rule':'Do not infer M6=M2 or M6=M3 from scenario labels alone; numerical equality requires the sufficient factorization and score-compatibility conditions already frozen in D3.',
        'next_gate':'Stage3E_N2_N3_scalable_certification_before_20_rep_pilot',
    }
    write_json(output_dir/'stage3d_d4_claim_boundary.json',claim)
    env={'python':platform.python_version(),'implementation':platform.python_implementation(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'pyyaml':yaml.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,'xgboost':xgboost.__version__,'platform':platform.platform(),'script_version':__version__}
    write_json(output_dir/'stage3d_d4_environment_inventory.json',env)
    srcinv=source_inventory(package_root); write_json(output_dir/'stage3d_d4_source_hashes.json',srcinv)
    counts={'new_case_candidate_trace_rows':int(len(trace)),'m6_candidate_rows':int((trace.method=='M6').sum()),'m3_candidate_rows':int((trace.method=='M3').sum()),'m4_candidate_rows':int((trace.method=='M4').sum()),'source_weight_rows':int(len(weights)),'case_registry_entries':int(len(cases_cfg['cases'])),'unique_case_ids':int(trace.case_id.nunique()),'nuisance_variants':sorted(trace.variant.unique().tolist()),'accepted_d2_replay_candidates':int(len(replay)),'expected_refusal_checks':int(len(expected_refusals)),'refusals':int(len(refusal_rows))}
    write_json(output_dir/'stage3d_d4_counts.json',counts)
    runtime={'seconds':time.perf_counter()-t0,'script_version':__version__}; write_json(output_dir/'stage3d_d4_runtime.json',runtime)
    return {'counts':counts,'claim':claim,'regression':regression,'runtime':runtime,'api':{'primary':api_meta,'duplicate':dup_meta}}
