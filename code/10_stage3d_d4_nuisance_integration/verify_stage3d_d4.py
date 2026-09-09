from __future__ import annotations
import gzip, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
for p in [ROOT/'vendor',ROOT/'src']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from geodose_stage3d_d4.io import sha256_file, write_json
from geodose_stage3d_d4.provenance import source_inventory

OUT=ROOT/'outputs_stage3d_d4'
checks=[]
def ck(name, ok, detail=''):
    checks.append({'name':name,'pass':bool(ok),'detail':str(detail)})

def rj(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def rc(name): return pd.read_csv(OUT/name)
def rcgz(name): return pd.read_csv(OUT/name,compression='gzip')

def bool_values(series):
    out=[]
    for v in series.tolist():
        if isinstance(v,(bool,np.bool_)): out.append(bool(v)); continue
        if isinstance(v,(int,np.integer)): out.append(bool(int(v))); continue
        x=str(v).strip().lower()
        if x in {'true','1','yes','y'}: out.append(True)
        elif x in {'false','0','no','n'}: out.append(False)
        else: raise ValueError(f'Cannot parse boolean value {v!r}')
    return np.asarray(out,dtype=bool)

try:
    required=['stage3d_d4_input_audit.json','stage3d_d4_d2_source_replay.csv','stage3d_d4_d2_runtime_api.json','stage3d_d4_expected_refusal_audit.csv','stage3d_d4_outcome_alignment_audit.csv','stage3d_d4_estimated_treatment_fit_audit.csv','stage3d_d4_candidate_trace.csv.gz','stage3d_d4_m3_m4_source_weight_audit.csv.gz','stage3d_d4_graph_variant_audit.csv','stage3d_d4_m6_prepare_audit.csv','stage3d_d4_m4_recompute_audit.csv','stage3d_d4_m3_treatment_invariance.csv','stage3d_d4_nuisance_delta_diagnostics.csv','stage3d_d4_upstream_reduction_regression.json','stage3d_d4_refusal_log.csv','stage3d_d4_case_summary.csv','stage3d_d4_claim_boundary.json','stage3d_d4_environment_inventory.json','stage3d_d4_source_hashes.json','stage3d_d4_counts.json','stage3d_d4_runtime.json','stage3d_d4_manifest.json']
    ck('required_outputs_present',all((OUT/n).is_file() for n in required))
    ia=rj('stage3d_d4_input_audit.json')
    ck('eight_upstream_archives_verified',len(ia['archives'])==8 and all(v['zip_integrity'] for v in ia['archives'].values()))
    ck('stage3c_source_18_of_18',ia['source_trees']['stage3c']['pass'] and ia['source_trees']['stage3c']['matched_file_count']==18 and ia['source_trees']['stage3c']['expected_file_count']==18)
    ck('d2_source_41_of_41',ia['source_trees']['stage3d_d2']['pass'] and ia['source_trees']['stage3d_d2']['matched_file_count']==41 and ia['source_trees']['stage3d_d2']['expected_file_count']==41)

    replay=rc('stage3d_d4_d2_source_replay.csv')
    ck('accepted_d2_source_replay_four_candidates',len(replay)==4,len(replay))
    ck('accepted_duplicate_replay_state_count_360',len(replay[replay.fixture_id.astype(str)=='G6_GAUSS_DUPLICATE'])==1 and int(replay.loc[replay.fixture_id.astype(str)=='G6_GAUSS_DUPLICATE','accepted_distinct_states'].iloc[0])==360)
    ck('accepted_d2_source_replay_conservative_pvalues',float(replay.abs_p_error.max())<=2e-12, replay.abs_p_error.max())
    ck('accepted_d2_source_replay_randomized_pvalues',float(replay.abs_randomized_p_error.max())<=2e-12,replay.abs_randomized_p_error.max())
    ck('accepted_d2_source_replay_tie_uniform',float(replay.abs_tie_uniform_error.max())<=2e-12,replay.abs_tie_uniform_error.max())
    ck('accepted_d2_source_replay_acceptance',bool(bool_values(replay.accept_match).all()) and bool(bool_values(replay.randomized_accept_match).all()))
    ck('accepted_d2_source_replay_state_counts',bool(bool_values(replay.distinct_states_match).all()))

    api=rj('stage3d_d4_d2_runtime_api.json')
    pf=api.get('primary_replay',{}).get('adapter_api',{}).get('preflight',{})
    callables=pf.get('callables',{}) if isinstance(pf,dict) else {}
    required_api={'stage3b_data','load_d1_fixture_config','build_d2_fixture','frozen_orbit_seed','derive_seed','prepare_orbit','CandidateEvaluator','CandidateEvaluator.evaluate','acceptance'}
    ck('accepted_d2_API_preflight_passed',bool(pf.get('pass')) and set(callables)==required_api,sorted(callables))
    ck('accepted_d2_frozen_precision_threshold',abs(float(pf.get('minimum_precision_eigenvalue_required',float('nan')))-1.0e-10)<=1e-25,pf.get('minimum_precision_eigenvalue_required'))
    exact_lock=pf.get('exact_lock',{})
    ck('accepted_d2_exact_API_lock_passed',bool(exact_lock.get('pass')))
    rc_lock=exact_lock.get('randomization_contract',{})
    ck('accepted_d2_randomization_contract_locked',rc_lock.get('scenario_mapping')=='all_D2_fixtures_use_S4' and rc_lock.get('derived_subseed_label')=='D2_G3_TIE_UNIFORM' and rc_lock.get('one_uniform_per_fixture') is True and rc_lock.get('uniform_held_fixed_across_candidates') is True)

    tr=rcgz('stage3d_d4_candidate_trace.csv.gz')
    ck('candidate_trace_252_rows',len(tr)==252,len(tr))
    ck('m3_rows_84',int((tr.method=='M3').sum())==84)
    ck('m4_rows_84',int((tr.method=='M4').sum())==84)
    ck('m6_rows_84',int((tr.method=='M6').sum())==84)
    ck('eight_case_entries_by_case_dose',len(tr[['case_id','target_dose']].drop_duplicates())==8)
    ck('seven_unique_case_ids',tr.case_id.nunique()==7)
    ck('S2_and_S7_wiring_cases_present',{'S2_SPATIAL','S7_EXCHANGEABLE'}.issubset(set(tr.case_id.astype(str))))
    ck('candidate_grid_minus1_0_1',set(np.round(tr.candidate_y.astype(float),12))=={-1.0,0.0,1.0})
    ck('pvalues_finite_bounded',np.isfinite(tr.p_value.astype(float)).all() and (tr.p_value>=-1e-12).all() and (tr.p_value<=1+1e-12).all())
    ck('m3_m4_inclusion_rule',bool((bool_values(tr[tr.method.isin(['M3','M4'])].included)==(tr[tr.method.isin(['M3','M4'])].p_value.astype(float).to_numpy()>0.1)).all()))
    ck('m6_source_is_D2_CandidateEvaluator',set(tr.loc[tr.method=='M6','source'].astype(str))=={'accepted_D2_source_CandidateEvaluator_new_case'})

    w=rcgz('stage3d_d4_m3_m4_source_weight_audit.csv.gz')
    ck('source_weight_504_rows',len(w)==504,len(w))
    sums=w.groupby(['case_id','target_dose','variant','candidate_y'])[['m3_spatial_mass','m4_mass']].sum()
    ck('m3_source_mass_sums_one',float(np.max(np.abs(sums.m3_spatial_mass-1)))<=2e-12)
    ck('m4_source_mass_sums_one',float(np.max(np.abs(sums.m4_mass-1)))<=2e-12)
    ck('source_indices_0_to_5',set(w.source_index.astype(int))==set(range(6)))

    m4=rc('stage3d_d4_m4_recompute_audit.csv')
    ck('m4_recompute_84_groups',len(m4)==84,len(m4))
    ck('m4_recompute_all_pass',bool(bool_values(m4['pass']).all()) and float(m4.max_abs_q4_recompute_error.max())<=2e-12)

    inv=rc('stage3d_d4_m3_treatment_invariance.csv')
    ck('m3_invariance_42_comparisons',len(inv)==42,len(inv))
    ck('m3_invariance_all_pass',bool(bool_values(inv['pass']).all()) and float(inv.abs_difference.max())<=2e-12)

    fits=rc('stage3d_d4_estimated_treatment_fit_audit.csv')
    ck('seven_unique_ET_fits',len(fits)==7 and fits.case_id.nunique()==7,len(fits))
    ck('ET_nuisance_training_nonempty',bool((fits.n_train.astype(int)>0).all()))

    oa=rc('stage3d_d4_outcome_alignment_audit.csv')
    ck('exact_gate_outcome_alignment_rows',len(oa)==8,len(oa))
    ck('exact_gate_target_node_is_frozen_320',set(oa.target_node_index.astype(int))=={320} and oa.target_unit_id.astype(str).str.contains('_R12_C20$').all())
    ck('exact_gate_no_measurement_error',bool(bool_values(oa['pass']).all()) and float(oa.max_abs_calibration_observed_minus_true.max())<=1e-14 and float(oa.abs_target_observed_minus_true.max())<=1e-14)

    graphs=rc('stage3d_d4_graph_variant_audit.csv')
    ck('graph_audit_14_rows',len(graphs)==14,len(graphs))
    ck('all_selected_blocks_connected',bool(bool_values(graphs.block_connected).all()))
    # S6 misspecified graph must actually differ in at least one graph diagnostic.
    s6=graphs[graphs.case_id.astype(str).str.startswith('S6_')]
    pivot=s6.pivot_table(index='case_id',columns='graph_mode',values='edge_count',aggfunc='first')
    ck('S6_fitted_graph_differs_from_true',bool(((pivot['TG']-pivot['FG']).abs()>0).all()),pivot.to_dict())

    prep=rc('stage3d_d4_m6_prepare_audit.csv')
    ck('m6_prepare_28_case_variants',len(prep)==28,len(prep))
    ck('all_four_nuisance_variants_present',set(prep.variant.astype(str))=={'OT_TG','ET_TG','OT_FG','ET_FG'})
    prep_meta=[json.loads(x) for x in prep.metadata_json.astype(str)]
    prep2=prep.copy(); prep2['_meta']=prep_meta
    iid=prep2[prep2.case_id.astype(str).isin(['S3_SHIFT','S7_EXCHANGEABLE'])]
    ck('iid_cases_use_explicit_rho0_gmrf_representation',len(iid)>0 and all(bool(m.get('residual_representation_override_iid_as_rho0_gmrf')) for m in iid['_meta']))
    ep0=prep2[(prep2.case_id.astype(str)=='ST1_MIXED_ATOMS') & (np.isclose(prep2.target_dose.astype(float),0.0))]
    ep1=prep2[(prep2.case_id.astype(str)=='ST1_MIXED_ATOMS') & (np.isclose(prep2.target_dose.astype(float),1.0))]
    ck('endpoint0_uses_accepted_endpoint0_config',len(ep0)==2 and all(m.get('base_evaluation_id')=='G6_GAUSS_ENDPOINT0' for m in ep0['_meta']))
    ck('endpoint1_uses_accepted_endpoint1_config',len(ep1)==2 and all(m.get('base_evaluation_id')=='G6_GAUSS_ENDPOINT1' for m in ep1['_meta']))

    red=rj('stage3d_d4_upstream_reduction_regression.json')
    ck('R1_R2_R4_remain_pass',red['R1_status']=='PASS' and red['R2_status']=='PASS' and red['R4_status']=='PASS')
    ck('R3_remains_deferred',red['R3_status']=='DEFERRED_STAGE3E')
    ck('accepted_M4_nonfactorization_preserved',bool(red['accepted_M4_nonfactorization_all']) and float(red['accepted_M4_max_delta_fact'])>1e-8)

    expected_ref=rc('stage3d_d4_expected_refusal_audit.csv')
    ck('two_expected_fail_closed_refusals',len(expected_ref)==2 and bool(bool_values(expected_ref['pass']).all()),expected_ref.to_dict(orient='records'))
    ck('expected_refusal_ids',set(expected_ref.refusal_id.astype(str))=={'UNSUPPORTED_ENDPOINT','DISCONNECTED_BLOCK'})

    refusals=rc('stage3d_d4_refusal_log.csv')
    ck('no_unexpected_refusals',len(refusals)==0,len(refusals))
    counts=rj('stage3d_d4_counts.json')
    ck('counts_match_trace',counts['new_case_candidate_trace_rows']==252 and counts['m6_candidate_rows']==84 and counts['m3_candidate_rows']==84 and counts['m4_candidate_rows']==84 and counts['source_weight_rows']==504 and counts['accepted_d2_replay_candidates']==4 and counts['expected_refusal_checks']==2 and counts['refusals']==0)

    claim=rj('stage3d_d4_claim_boundary.json')
    ck('source_level_newdata_adapter_true',claim['source_level_M6_new_data_candidate_adapter_small_exact_blocks'] is True)
    ck('no_G1_G2_G3_reimplementation',claim['G1_G2_G3_reimplemented'] is False)
    ck('no_exact_estimated_nuisance_coverage_claim',claim['estimated_nuisance_exact_coverage_claim'] is False)
    ck('not_pilot_scale_ready',claim['pilot_scale_ready'] is False)
    ck('no_pilot_or_production',claim['pilot_run'] is False and claim['production_run'] is False and claim['publication_performance_evidence'] is False)
    ck('N2_N3_not_implemented',claim['N2_N3_implemented'] is False and claim['R3_status']=='DEFERRED_STAGE3E')

    env=rj('stage3d_d4_environment_inventory.json')
    ck('D4_runtime_version_is_v1_1_0',env.get('script_version')=='1.1.0-stage3d-d4',env.get('script_version'))
    src_record=rj('stage3d_d4_source_hashes.json'); src_now=source_inventory(ROOT)
    ck('source_inventory_version_is_v1_1_0',src_record.get('version')=='1.1.0',src_record.get('version'))
    ck('source_inventory_matches_runtime',src_record['aggregate_sha256']==src_now['aggregate_sha256'] and src_record['file_count']==src_now['file_count'])

    manifest=rj('stage3d_d4_manifest.json'); manifest_ok=True; manifest_detail=[]
    for row in manifest['files']:
        fp=OUT/row['name']; ok=fp.is_file() and sha256_file(fp)==row['sha256'] and fp.stat().st_size==row['size']; manifest_ok &= ok
        if not ok: manifest_detail.append(row['name'])
    ck('manifest_files_match',manifest_ok,manifest_detail)

except Exception as exc:
    ck('verifier_exception',False,f'{type(exc).__name__}: {exc}')

failed=[x for x in checks if not x['pass']]
status='verified_complete' if not failed else 'failed'
report={'status':status,'checks':len(checks),'failed_checks':len(failed),'check_results':checks}
write_json(OUT/'STAGE3D_D4_VERIFICATION.json',report)
print(f"STAGE 3D D4 VERIFICATION: {status.upper()} {len(checks)-len(failed)}/{len(checks)}")
if failed:
    for x in failed: print('FAIL',x['name'],x['detail'])
    raise SystemExit(1)
