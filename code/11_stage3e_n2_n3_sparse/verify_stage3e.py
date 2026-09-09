from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import yaml
from geodose_stage3e.io import sha256_file
from geodose_stage3e.provenance import source_inventory
from geodose_stage3e.version import VERSION, STAGE

OUT=ROOT/'outputs_stage3e'
checks=[]
def ck(name: str, cond: bool, detail=''):
    ok=bool(cond); checks.append({'name':name,'pass':ok,'detail':str(detail)})
    if not ok: raise AssertionError(f'{name}: {detail}')
def j(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def c(name): return pd.read_csv(OUT/name)
def finite(x): return np.isfinite(np.asarray(x,dtype=float)).all()
def bools(s):
    if getattr(s,'dtype',None)==bool: return s.to_numpy(dtype=bool)
    return s.astype(str).str.lower().map({'true':True,'false':False}).to_numpy(dtype=bool)

def main():
    required=[
      'stage3e_input_audit.json','stage3e_case_registry.csv','stage3e_vecchia_ordering.csv','stage3e_n2_certificate.csv','stage3e_n2_local_terms.csv.gz',
      'stage3e_neighborhood_monotonicity.csv','stage3e_vecchia_precision_audit.csv','stage3e_r3_reduction.csv','stage3e_exact_sparse_special_case.json',
      'stage3e_target_shift_amplification_test.json','stage3e_target_weighted_n2_formula_test.json','stage3e_target_ratio_certificate_formula_test.json',
      'stage3e_target_ratio_audit.csv','stage3e_treatment_support_diagnostic.csv','stage3e_spatial_confidence_audit.csv','stage3e_spatial_confidence_intervals.csv',
      'stage3e_gmrf_boundary_certificate.csv','stage3e_s3_confidence_coverage_test.json','stage3e_working_graph_misspecification.csv','stage3e_n3_certificate.csv',
      'stage3e_exact_vs_sparse_pvalue.csv','stage3e_newdata_sparse_candidate_trace.csv','stage3e_newdata_sparse_prepare_audit.csv',
      'stage3e_sparse_prediction_set_candidate_trace.csv.gz','stage3e_sparse_prediction_set_components.csv','stage3e_sparse_prediction_set_boundary_audit.csv',
      'stage3e_sparse_prediction_set_summary.csv','stage3e_sparse_prediction_set_crosscheck.csv','stage3e_certificate_refusal_grid.csv','stage3e_n3_arithmetic_tests.json',
      'stage3e_refusal_log.csv','stage3e_pilot_readiness.json','stage3e_claim_boundary.json','stage3e_counts.json','stage3e_runtime.json','stage3e_source_hashes.json','stage3e_manifest.json'
    ]
    ck('required_outputs_present',all((OUT/x).is_file() for x in required),[x for x in required if not (OUT/x).is_file()])

    contract=yaml.safe_load((ROOT/'configs'/'stage3e_contract.yaml').read_text(encoding='utf-8'))
    ck('stage_contract_version',contract['version']==VERSION,(contract['version'],VERSION))
    ck('stage_contract_alpha',float(contract['alpha'])==0.10,contract['alpha'])
    ck('operational_thresholds_deferred',contract['operational_ess_threshold'] is None and contract['operational_max_weight_threshold'] is None and contract['operational_coverage_lower_bound_threshold'] is None)

    # Input/source provenance.
    ia=j('stage3e_input_audit.json')
    expected=jcfg=json.loads((ROOT/'configs'/'expected_upstream_hashes.json').read_text(encoding='utf-8'))
    ck('four_frozen_upstream_archives',set(expected)=={'stage3a_output_zip','stage3b_output_zip','stage3d_d2_output_zip','stage3d_d4_output_zip'})
    for key,exp in expected.items():
        rec=ia[key]; ck(f'{key}_hash_exact',str(rec.get('sha256','')).lower()==str(exp).lower(),rec)
        ck(f'{key}_zip_integrity',bool(rec.get('zip_integrity')),rec)
    ck('D2_source_41_of_41',ia['stage3d_d2_source_root'].get('pass') and int(ia['stage3d_d2_source_root'].get('matched_file_count',-1))==41,ia['stage3d_d2_source_root'])
    ck('D4_source_39_of_39',ia['stage3d_d4_source_root'].get('pass') and int(ia['stage3d_d4_source_root'].get('matched_file_count',-1))==39,ia['stage3d_d4_source_root'])
    api=j('stage3e_d2_api_audit.json')
    ck('accepted_D2_API_file_count',int(api['expected_file_count'])==41)
    ck('accepted_D4_API_file_count',int(api['accepted_D4_source_file_count'])==39)

    src_expected=j('stage3e_source_hashes.json'); src_now=source_inventory(ROOT)
    ck('source_inventory_file_count',int(src_expected['file_count'])==int(src_now['file_count']),(src_expected['file_count'],src_now['file_count']))
    ck('source_inventory_aggregate',src_expected['aggregate_sha256']==src_now['aggregate_sha256'],(src_expected['aggregate_sha256'],src_now['aggregate_sha256']))
    ck('source_inventory_paths',[(x['path'],x['sha256'],x['size']) for x in src_expected['files']]==[(x['path'],x['sha256'],x['size']) for x in src_now['files']])

    # Output manifest independent recomputation.
    man=j('stage3e_manifest.json'); rows=[]
    for r in man['files']:
        fp=OUT/r['name']; rows.append((r['name'],sha256_file(fp),fp.stat().st_size)); ck('manifest_'+r['name'],rows[-1][1]==r['sha256'] and rows[-1][2]==int(r['size']),rows[-1])
    h=hashlib.sha256()
    for n,hs,sz in rows: h.update(f'{n}\0{hs}\0{sz}\n'.encode())
    ck('manifest_aggregate',h.hexdigest()==man['aggregate_sha256'],(h.hexdigest(),man['aggregate_sha256']))
    ck('manifest_file_count',len(rows)==int(man['file_count']))

    cases=c('stage3e_case_registry.csv'); n2=c('stage3e_n2_certificate.csv'); local=pd.read_csv(OUT/'stage3e_n2_local_terms.csv.gz'); mono=c('stage3e_neighborhood_monotonicity.csv'); prec=c('stage3e_vecchia_precision_audit.csv')
    structural=cases.drop_duplicates('case_id')
    mvals=list(map(int,contract['candidate_neighborhood_sizes']))
    ck('case_dose_count_11',len(cases)==11,len(cases)); ck('structural_case_count_10',len(structural)==10,len(structural)); ck('endpoint_case_has_two_doses',len(cases[cases.case_id=='ST1_MIXED_ATOMS'])==2)
    ck('n2_row_count',len(n2)==len(structural)*len(mvals),len(n2)); ck('n2_unique_case_m',not n2.duplicated(['case_id','m']).any())
    ck('local_row_count',len(local)==len(structural)*len(mvals)*625,len(local)); ck('local_terms_nonnegative',(local.local_omitted_information>=-1e-12).all(),local.local_omitted_information.min())
    ck('conditional_variances_positive',(local.full_predecessor_variance>0).all() and (local.sparse_neighbor_variance>0).all())
    sums=local.groupby(['case_id','m']).local_omitted_information.sum().rename('recomputed').reset_index().merge(n2[['case_id','m','observational_omitted_information_L']],on=['case_id','m'])
    ck('N2_sum_local_terms',float(np.max(np.abs(sums.recomputed-sums.observational_omitted_information_L)))<2e-10,float(np.max(np.abs(sums.recomputed-sums.observational_omitted_information_L))))
    recomputed_tv=np.minimum(1.0,np.sqrt(np.maximum(0,n2.target_weighted_delta_N2.to_numpy(dtype=float))/2))
    ck('N2_Pinsker_formula',float(np.max(np.abs(recomputed_tv-n2.delta_sparse_tv_upper.to_numpy(dtype=float))))<2e-12)
    recomputed_lb=np.maximum(0,1-float(contract['alpha'])-recomputed_tv)
    ck('N2_coverage_transfer_formula',float(np.max(np.abs(recomputed_lb-n2.oracle_sparse_coverage_lower_bound.to_numpy(dtype=float))))<2e-12)
    ck('N2_precision_PD',(prec.min_eigenvalue>float(contract['minimum_precision_eigenvalue'])).all(),prec.min_eigenvalue.min())
    ck('N2_no_dense_inverse',not bools(prec.dense_inverse_formed).any())
    ck('N2_neighborhood_monotonicity',len(mono)==len(structural)*(len(mvals)-1) and bools(mono['pass']).all(),mono[~mono['pass'].astype(bool)] if mono['pass'].dtype==bool else '')
    iid=n2[n2.case_id.isin(['S3_SHIFT','S7_EXCHANGEABLE'])]
    ck('IID_N2_zero',float(iid.target_weighted_delta_N2.abs().max())<1e-12,float(iid.target_weighted_delta_N2.abs().max()))

    ex=j('stage3e_exact_sparse_special_case.json'); ck('exact_sparse_special_case',bool(ex['pass']) and float(ex['delta_n2'])<=1e-10,ex)
    r3=c('stage3e_r3_reduction.csv'); ck('R3_identity_cases',len(r3)==9 and (r3.reduction_id=='R3').all())
    ck('R3_exact',bools(r3['pass']).all() and float(r3.abs_difference.max())<=1e-15,r3.abs_difference.max())

    shift=j('stage3e_target_shift_amplification_test.json'); ck('target_shift_identity',float(shift['identity_abs_error'])<1e-15,shift); ck('target_shift_CS',bool(shift['cs_dominates']),shift)
    tw=j('stage3e_target_weighted_n2_formula_test.json'); ck('generic_target_weighting_normalized',abs(float(tw['mean_target_design_ratio'])-1)<1e-15,tw); ck('generic_target_weighting_identity',float(tw['identity_abs_error'])<1e-15,tw); ck('generic_target_weighting_CS',bool(tw['cs_dominates']),tw)
    rf=j('stage3e_target_ratio_certificate_formula_test.json'); ck('bounded_ratio_formula_positive',float(rf['epsilon_r_n'])>0 and float(rf['lambda_r'])>0,rf); ck('bounded_ratio_KL_equals_epsilon',abs(float(rf['target_design_KL_upper_bound'])-float(rf['epsilon_r_n']))<1e-15)

    ratio=c('stage3e_target_ratio_audit.csv'); ck('ratio_row_count_10',len(ratio)==10,len(ratio)); identity=ratio[ratio['mode']=='identity']; nonid=ratio[ratio['mode']!='identity']
    ck('ratio_identity_exact',len(identity)==9 and float(identity.oracle_ratio_identity_max_error.max())<=1e-15)
    ck('ratio_identity_theorem_status',bools(identity.theorem_eligible).all())
    ck('ratio_nonidentity_single',len(nonid)==1 and nonid.iloc[0].case_id=='S4_RHO060_DESIGN_SHIFT')
    ck('ratio_nonidentity_scope_refused',not bool(nonid.iloc[0].theorem_eligible) and 'UNBOUNDED_GAUSSIAN' in str(nonid.iloc[0].theorem_status))
    ck('ratio_empirical_normalization',abs(float(nonid.iloc[0].empirical_observational_normalized_ratio_mean)-1)<2e-15,nonid.iloc[0].empirical_observational_normalized_ratio_mean)
    ck('ratio_prior_odds_factor',bools(ratio.class_prior_odds_factor_used).all() and np.allclose(ratio.prior_odds.to_numpy(dtype=float),2.0))

    supp=c('stage3e_treatment_support_diagnostic.csv'); ck('support_query_count_11',len(supp)==11); ck('support_positive_weights',(supp.positive_weight_rows>0).all()); ck('support_thresholds_not_applied',not bools(supp.operational_threshold_applied).any() and (supp.threshold_status=='DEFERRED_TO_20REP_PILOT').all())

    s3=c('stage3e_spatial_confidence_audit.csv'); sint=c('stage3e_spatial_confidence_intervals.csv'); gm=c('stage3e_gmrf_boundary_certificate.csv')
    ck('S3_case_count_10',len(s3)==10 and len(sint)>=10); ck('S3_training_component_separated',(s3.cross_component_edges==0).all()); ck('S3_training_n_250',(s3.n_training==250).all()); ck('S3_sets_nonempty',bools(s3.eligible).all() and (s3.outer_interval_count>=1).all())
    ck('S3_true_rho_sim_check',bools(s3.true_rho_inside_outer_set_simulation_check).all())
    s3mc=j('stage3e_s3_confidence_coverage_test.json'); ck('S3_internal_coverage_test',bool(s3mc['pass']) and int(s3mc['repetitions'])==int(contract['s3_coverage_test_repetitions']),s3mc)
    ck('GMRF_C_case_count_10',len(gm)==10); ck('GMRF_C_finite_or_refused',all((bool(r.eligible) and math.isfinite(float(r.kl_upper_bound)) and float(r.kl_upper_bound)>=0) or (not bool(r.eligible)) for r in gm.itertuples(index=False)))
    elig=gm[bools(gm.eligible)]; ck('GMRF_C_Pinsker_formula',len(elig)==0 or float(np.max(np.abs(np.minimum(1,np.sqrt(elig.kl_upper_bound.to_numpy(dtype=float)/2))-elig.tv_pinsker_upper_bound.to_numpy(dtype=float))))<2e-12)

    miss=c('stage3e_working_graph_misspecification.csv'); ck('S6_misspec_two_rows',set(miss.case_id)=={'S6_ROOK','S6_OMIT50'} and len(miss)==2); ck('S6_misspec_KL_nonnegative',(miss.oracle_sparse_to_working_sparse_KL_simulation_only>=0).all()); ck('S6_misspec_oracle_only',(miss.theorem_deployment_status=='ORACLE_DIAGNOSTIC_ONLY_NO_REAL_WORLD_MISSPECIFICATION_CERTIFICATE').all())

    n3=c('stage3e_n3_certificate.csv'); ck('N3_rows_24',len(n3)==24,len(n3)); oracle=n3[n3.route=='ORACLE_SPARSE_PRIMARY']; est=n3[n3.route=='ESTIMATED_NUISANCE_STATUS']; work=n3[n3.route=='WORKING_GRAPH_ORACLE_DIAGNOSTIC']
    ck('N3_oracle_rows_11',len(oracle)==11); ck('N3_oracle_theorem_eligible',bools(oracle.theorem_eligible).all())
    olb=np.maximum(0,1-oracle.alpha.to_numpy(float)-oracle.delta_sparse.to_numpy(float)-oracle.delta_mis.to_numpy(float)-oracle.delta_est.to_numpy(float)-oracle.delta_comp.to_numpy(float)-oracle.certificate_failure_probability_eta.to_numpy(float))
    ck('N3_oracle_arithmetic',float(np.max(np.abs(olb-oracle.coverage_lower_bound.to_numpy(float))))<2e-12)
    ck('N3_estimated_not_overclaimed',len(est)==11 and not bools(est.theorem_eligible).any() and est.coverage_lower_bound.isna().all())
    ck('N3_working_graph_oracle_only',len(work)==2 and (work.claim_status=='SIMULATION_ORACLE_DIAGNOSTIC_ONLY').all())
    arith=j('stage3e_n3_arithmetic_tests.json'); g=arith['good_event_example']; ck('N3_good_event_arithmetic',abs(float(g['coverage_lower_bound'])-0.8)<1e-15,g); nd=arith['no_double_counting_example']; ck('N3_no_double_counting',abs(float(nd['selected'])-min(float(nd['pinsker']),float(nd['oscillation'])))<1e-15,nd)

    smoke=c('stage3e_exact_vs_sparse_pvalue.csv'); ck('exact_sparse_smoke_12',len(smoke)==12 and set(smoke.m)=={8,16,32,64}); ck('exact_sparse_pvalues_valid',smoke.exact_conservative_p.between(0,1).all() and smoke.sparse_conservative_p.between(0,1).all()); ck('exact_sparse_abs_difference_recomputed',float(np.max(np.abs(np.abs(smoke.exact_conservative_p-smoke.sparse_conservative_p)-smoke.abs_p_difference)))<2e-15)

    prep=c('stage3e_newdata_sparse_prepare_audit.csv'); trace=c('stage3e_newdata_sparse_candidate_trace.csv')
    ck('newdata_prepare_11',len(prep)==11); ck('newdata_target_node_320',all(str(x).split(',')[-1]=='320' for x in prep.block_nodes),prep.block_nodes.tolist()); ck('newdata_uses_accepted_D4',bools(prep.D4_preparation_reused).all()); ck('S8_scope_alias_transparent',len(prep[(prep.case_id=='S8_SEVERE_TAIL') & (prep.target_scope_original=='scenario_primary_extra') & bools(prep.target_scope_local_D2_alias_applied)])==1)
    ck('newdata_candidate_rows_132',len(trace)==132,len(trace)); ck('newdata_grid_complete',not trace.duplicated(['case_id','target_dose','m','candidate_y']).any() and set(trace.m)=={8,16,32,64} and set(trace.candidate_y)=={-1.0,0.0,1.0})
    ck('newdata_pvalues_valid',trace.conservative_p.between(0,1).all() and trace.randomized_p.between(0,1).all()); ck('newdata_randomized_subset_conservative',not ((bools(trace.randomized_accept)) & (~bools(trace.conservative_accept))).any()); ck('newdata_source_engine_D2',(trace.source_level_exact_engine=='ACCEPTED_D2_G1_G2_G3').all()); ck('newdata_preparation_D4',(trace.newdata_preparation=='ACCEPTED_D4_V1_1_0').all())

    fs=c('stage3e_sparse_prediction_set_summary.csv'); fc=c('stage3e_sparse_prediction_set_crosscheck.csv'); fcomp=c('stage3e_sparse_prediction_set_components.csv'); fb=c('stage3e_sparse_prediction_set_boundary_audit.csv'); ftrace=pd.read_csv(OUT/'stage3e_sparse_prediction_set_candidate_trace.csv.gz')
    ck('full_sparse_inversion_summary_two_modes',len(fs)==2 and set(fs.tie_mode)=={'conservative','randomized'}); ck('full_sparse_inversion_case',set(fs.case_id)=={'S4_RHO060'} and set(fs.m)=={64}); ck('full_sparse_inversion_independent_grid',set(fs.independent_grid_points)=={4096} and set(fs.independent_grid_conservative_false_negatives)=={0} and set(fs.independent_grid_randomized_false_negatives)=={0}); ck('full_sparse_inversion_trace_large',len(ftrace)>5000,len(ftrace)); ck('full_sparse_inversion_pvalues_valid',ftrace.conservative_p.between(0,1).all() and ftrace.randomized_p.between(0,1).all()); ck('full_sparse_randomized_subset',not (bools(ftrace.randomized_accept)&~bools(ftrace.conservative_accept)).any()); ck('full_sparse_components_present',len(fcomp)>=2 and len(fb)>=2); ck('full_sparse_crosscheck_three',len(fc)==3 and float(fc[['conservative_p_abs_diff','randomized_p_abs_diff']].to_numpy(float).max())<=2e-12 and bools(fc.conservative_accept_match).all() and bools(fc.randomized_accept_match).all())
    ck('finite_domain_claim_only',not bools(fs.full_real_line_claim).any() and bools(fs.domain_truncated_set).all() and not bools(fs.unbounded_claim).any())

    cert=c('stage3e_certificate_refusal_grid.csv'); ck('certificate_grid_60',len(cert)==60); rec=np.maximum(0,1-float(contract['alpha'])-cert.delta_sparse_tv_upper.to_numpy(float)); ck('certificate_grid_arithmetic',float(np.max(np.abs(rec-cert.coverage_lower_bound.to_numpy(float))))<2e-12); ck('certificate_nonvacuity_rule',np.array_equal(cert.mathematically_nonvacuous.to_numpy(bool),cert.coverage_lower_bound.to_numpy(float)>0)); ck('certificate_operational_threshold_deferred',not bools(cert.operational_threshold_applied).any() and (cert.operational_threshold_status=='DEFERRED_TO_20REP_PILOT').all())

    ready=j('stage3e_pilot_readiness.json'); ck('pilot_readiness_true',bool(ready['ready_for_preregistered_20rep_pilot']),ready); ck('primary_m64',int(ready['primary_neighborhood_size'])==64); ck('full_inversion_readiness_pass',bool(ready['scalable_full_prediction_set_inversion_smoke_pass'])); ck('production_still_blocked',ready['production_run_allowed'] is False); ck('estimated_nuisance_not_certified',ready['estimated_nuisance_certified'] is False)
    primary=n2[n2.m==64]; ck('readiness_worst_delta_recomputed',abs(float(ready['worst_oracle_sparse_tv_bound_across_stage3e_cases'])-float(primary.delta_sparse_tv_upper.max()))<2e-15); ck('readiness_min_lb_recomputed',abs(float(ready['minimum_oracle_sparse_coverage_lower_bound_across_stage3e_cases'])-float(primary.oracle_sparse_coverage_lower_bound.min()))<2e-15)

    claims=j('stage3e_claim_boundary.json'); ck('claim_N2_oracle_backed',claims['N2_oracle_gaussian_theorem_backed'] is True); ck('claim_N3_oracle_backed',claims['N3_oracle_structural_coverage_bound_theorem_backed'] is True); ck('claim_ratio_shift_not_backed',claims['target_ratio_nonidentity_fit_theorem_backed'] is False); ck('claim_ET_not_T1',claims['Stage3C_estimated_treatment_T1_certified'] is False); ck('claim_no_S3_Vecchia_fusion',claims['S3_GMRF_C_fused_into_Vecchia_estimation_bound'] is False); ck('claim_no_dense_inverse',claims['full_625_node_dense_inverse_formed'] is False); ck('claim_no_MC_orbit',claims['Monte_Carlo_orbit_approximation_used'] is False and float(claims['delta_comp_primary'])==0.0); ck('claim_full_inversion_validated',claims['scalable_full_prediction_set_inversion_validated_on_S4_RHO060_m64'] is True); ck('claim_no_performance_evidence',claims['pilot_performance_evidence_generated'] is False and claims['production_performance_evidence_generated'] is False)

    refusal=c('stage3e_refusal_log.csv'); ck('expected_ratio_scope_warning_present',((refusal.case_id=='S4_RHO060_DESIGN_SHIFT') & refusal.code.astype(str).str.contains('UNBOUNDED_GAUSSIAN')).any())
    counts=j('stage3e_counts.json'); ck('counts_match_n2',int(counts['n2_certificate_rows'])==len(n2)); ck('counts_match_newdata',int(counts['newdata_sparse_candidate_rows'])==len(trace)); ck('counts_match_full_trace',int(counts['sparse_prediction_set_trace_rows'])==len(ftrace)); ck('counts_match_full_summary',int(counts['sparse_prediction_set_summary_rows'])==len(fs))

    runtime=j('stage3e_runtime.json'); ck('runtime_version',runtime['stage']==STAGE and runtime['version']==VERSION and float(runtime['elapsed_seconds'])>0,runtime)

    out={'status':'verified_complete','stage':STAGE,'version':VERSION,'checks':len(checks),'failed_checks':0,'passed_checks':len(checks),'check_results':checks,'claim_boundary':'Stage3E certification/reduction/software validation only; no pilot or production performance evidence'}
    (OUT/'STAGE3E_VERIFICATION.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(f'STAGE 3E N2/N3 SCALABLE CERTIFICATION VERIFIED COMPLETE ({len(checks)}/{len(checks)})')
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        fail={'status':'verification_failed','stage':STAGE,'version':VERSION,'checks_completed':len(checks),'failed_checks':1,'error':f'{type(exc).__name__}: {exc}','check_results':checks}
        try: (OUT/'STAGE3E_VERIFICATION.json').write_text(json.dumps(fail,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        except Exception: pass
        print(f'STAGE3E VERIFICATION FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
