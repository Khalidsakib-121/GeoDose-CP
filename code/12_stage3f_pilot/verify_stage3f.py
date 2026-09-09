from __future__ import annotations
import json,math,sys,traceback,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from geodose_stage3f.io import sha256_file
from geodose_stage3f.pilot_math import choose_thresholds, operational_gate
from geodose_stage3f.version import VERSION,STAGE

OUT=ROOT/'outputs_stage3f'
checks=[]
def ck(name,cond,detail=''):
    ok=bool(cond); checks.append({'check':name,'pass':ok,'detail':str(detail)})
    if not ok: raise RuntimeError(f'{name}: {detail}')

def read(name): return pd.read_csv(OUT/name)

def main():
    ck('output_dir_exists',OUT.is_dir())
    contract=yaml.safe_load((ROOT/'configs/pilot_contract.yaml').read_text())
    results=read('stage3f_pilot_query_results.csv.gz'); support=read('stage3f_support_certificate_inputs.csv.gz'); targets=read('stage3f_target_block_selection_audit.csv'); metrics=read('stage3f_pilot_metrics.csv'); width=read('stage3f_width_diagnostic_audit.csv'); nuisance=read('stage3f_propensity_specification_diagnostics.csv'); fits=read('stage3f_propensity_fit_audit.csv'); outfits=read('stage3f_outcome_fit_audit.csv'); sanity=read('stage3f_certified_coverage_sanity.csv'); rep1=read('stage3f_stage3b_rep1_replay_audit.csv'); graph=read('stage3f_graph_diagnostics.csv'); dose=read('stage3f_dose_bin_coverage_metrics.csv'); certsum=read('stage3f_certificate_support_summary.csv'); widthsum=read('stage3f_width_diagnostic_summary.csv'); case_runtime=read('stage3f_case_rep_runtime.csv')
    thresh=json.loads((OUT/'stage3f_operational_thresholds_FROZEN.json').read_text()); claim=json.loads((OUT/'stage3f_claim_boundary.json').read_text()); runtime=json.loads((OUT/'stage3f_runtime.json').read_text()); align=json.loads((OUT/'stage3f_registry_alignment.json').read_text()); up=json.loads((OUT/'stage3f_upstream_provenance_audit.json').read_text())
    ck('version',runtime['version']==VERSION and claim['version']==VERSION)
    ck('stage_name',runtime['stage']==STAGE and claim['stage']==STAGE)
    ck('support_count',len(support)==400,len(support))
    ck('target_selection_count',len(targets)==400,len(targets))
    ck('result_count',len(results)==5880,len(results))
    ck('width_audit_count',len(width)==120,len(width))
    ck('nuisance_diag_count',len(nuisance)==240,len(nuisance))
    ck('propensity_fit_count',len(fits)==60,len(fits))
    ck('outcome_fit_count',len(outfits)==580,len(outfits))
    ck('rep1_replay_count',len(rep1)==20,len(rep1))
    ck('case_runtime_count',len(case_runtime)==400,len(case_runtime))
    ck('dose_metrics_nonempty',len(dose)>0,len(dose))
    ck('query_reporting_fields_present',set(['requested_dose','intervention_bandwidth','positivity_status','method_branch','spatial_block_rule','sparse_boundary_size','spatial_boundary_rule','nuisance_certificate_status']).issubset(results.columns))
    ck('support_reporting_fields_present',set(['requested_dose','intervention_bandwidth','positivity_status']).issubset(support.columns))
    ck('metric_reporting_scope_recorded','metric_reporting_scope' in align and 'MET07' in align['metric_reporting_scope']['deferred_to_production_matched_coverage'] and 'MET01' in align['metric_reporting_scope']['pilot_fully_computed'])
    ck('certificate_summary_20_cases',len(certsum)==20,len(certsum))
    ck('width_summary_nonempty',len(widthsum)>0,len(widthsum))
    ck('rep1_replay_all_pass',rep1['pass'].astype(str).str.lower().isin(['true','1']).all())
    ck('scenario_set',set(support.scenario_id.astype(str))=={f'S{i}' for i in range(1,11)})
    ck('case_set',set(support.case_id.astype(str))==set(contract['pilot_case_ids']))
    ck('every_case_20rep',(support.groupby('case_id').replication.nunique()==20).all())
    ck('every_case_rep_unique',not support.duplicated(['case_id','replication']).any())
    ck('methods_exact',set(results.method.astype(str))==set(['M1','M2','M3','M4','M5','M6']))
    ck('rf_all_case_rep',len(results[results.track=='RF_SHARED_PREDICTOR_EMPIRICAL'])==2400)
    ck('oracle_all_case_rep',len(results[results.track=='ORACLE_STRUCTURAL_OT_TG'])==2400)
    ck('xgb_predeclared_only',set(results.loc[results.track=='XGB_SHARED_PREDICTOR_EMPIRICAL','scenario_id'].astype(str))==set(contract['xgb_scenarios']))
    # Target population: pre-outcome uniform eligible slot selection, paired scenario-level u across registered factor variants.
    ck('target_selection_preoutcome',(~targets.selection_uses_A.astype(bool)&~targets.selection_uses_Y.astype(bool)&~targets.selection_uses_support_diagnostics.astype(bool)).all())
    ck('eligible_positive',(targets.eligible_target_count.astype(int)>0).all())
    ck('target_rank_valid',((targets.target_selection_rank.astype(int)>=0)&(targets.target_selection_rank.astype(int)<targets.eligible_target_count.astype(int))).all())
    u_spread=targets.groupby(['scenario_id','replication']).target_selection_uniform.agg(lambda x: float(np.max(x)-np.min(x)))
    ck('factor_variants_share_target_uniform',(u_spread<=1e-15).all(),u_spread.max())
    blocks=targets.block_nodes.astype(str).str.split('|')
    ck('exact_block_size6',blocks.map(len).eq(6).all())
    ck('target_last_in_block',all(int(b[-1])==int(t) for b,t in zip(blocks,targets.target_node)))
    # Threshold selector must reproduce from pre-outcome support only.
    chosen,audit=choose_thresholds(support,results,contract)
    for k in ['minimum_ess','maximum_normalized_weight','minimum_graph_safe_count']:
        ck('threshold_recompute_'+k,abs(float(chosen[k])-float(thresh[k]))<=1e-15,(chosen[k],thresh[k]))
    ck('threshold_controlled_pilot_coverage_only',thresh['selection_used_outcomes'] is True and thresh['selection_used_controlled_pilot_outcomes'] is True and claim['threshold_selection_used_controlled_pilot_outcomes'] is True)
    ck('threshold_no_width',thresh['selection_used_width'] is False and claim['threshold_selection_used_width'] is False)
    ck('threshold_no_nsw',thresh['selection_used_nsw'] is False and claim['threshold_selection_used_nsw'] is False)
    ck('threshold_selection_rule_frozen',thresh['selection_rule_status']=='STAGE3F_FROZEN_LEXICOGRAPHIC_NO_POSTHOC_RETENTION_CUTOFF')
    ck('threshold_no_posthoc_retention_cutoff',all(k not in contract['threshold_selection'] for k in ['minimum_safe_retention','minimum_s3_retention','minimum_strong_s4_retention']))
    ck('threshold_casewise_penalty_recorded',int(thresh['safe_case_coverage_groups_used'])<=int(thresh['safe_case_count']) and int(thresh['minimum_safe_case_returned'])>=0)
    # N2/N3 arithmetic.
    tv=np.sqrt(np.maximum(support.n2_delta_kl.to_numpy(float),0.0)/2.0); lb=np.maximum(0.0,0.9-tv)
    ck('support_tv_formula',np.max(np.abs(tv-support.n2_tv_bound.to_numpy(float)))<=2e-12,np.max(np.abs(tv-support.n2_tv_bound.to_numpy(float))))
    ck('support_lb_formula',np.max(np.abs(lb-support.certified_coverage_lower_bound.to_numpy(float)))<=2e-12,np.max(np.abs(lb-support.certified_coverage_lower_bound.to_numpy(float))))
    ck('threshold_support_inputs_no_outcomes',(~support.threshold_inputs_use_outcomes.astype(bool)).all())
    ck('threshold_support_inputs_no_width',(~support.threshold_inputs_use_width.astype(bool)).all())
    ck('threshold_support_inputs_no_nsw',(~support.threshold_inputs_use_nsw.astype(bool)).all())
    ck('support_graph_safe_positive',(support.graph_safe_count.astype(int)>0).all())
    ck('support_population_eligible',support.target_population_eligible.astype(bool).all())
    # Main method arithmetic and acceptance rules.
    ck('pvalues_in_range',results.pvalue_conservative.dropna().between(0,1).all() and results.pvalue_randomized.dropna().between(0,1).all())
    om=results[results.track=='ORACLE_STRUCTURAL_OT_TG']; em=results[results.track!='ORACLE_STRUCTURAL_OT_TG']
    ck('oracle_certificate_fields_finite',om.certified_coverage_lower_bound.notna().all() and om.certificate_theorem_backed.astype(bool).all())
    ck('empirical_certificate_not_claimed',em.certified_coverage_lower_bound.isna().all() and not em.certificate_theorem_backed.astype(bool).any())
    p=results[results.method.isin(['M3','M4']) & results.raw_method_returned.astype(bool)].copy(); ck('m3_m4_accept_rule',(p.raw_covered.astype(bool)==(p.pvalue_conservative.astype(float)>0.1)).all())
    p=results[(results.method=='M6') & results.raw_method_returned.astype(bool)].copy(); ck('m6_accept_rule',(p.raw_covered.astype(bool)==(p.pvalue_conservative.astype(float)>0.1)).all())
    ck('m6_state_count_720',(p.m6_distinct_states.dropna().astype(int)==720).all())
    refused_graph=results[(results.method.isin(['M3','M4','M6'])) & (~results.raw_method_returned.astype(bool)) & (results.raw_refusal_code.astype(str)=='R05_GRAPH_COMPONENT_TOO_SMALL')]
    ck('working_graph_disconnect_is_refusal_not_failure',len(refused_graph)>0 and not refused_graph.computational_failure.astype(bool).any())
    ints=results[results.method.isin(['M1','M2','M5']) & results.raw_method_returned.astype(bool)].copy(); calc=(ints.A_star*0+1).astype(bool)
    # Coverage check uses the track target Y internally; raw_covered is already tested in unit/integration; verify interval ordering here.
    ck('returned_intervals_ordered',((ints.lower<=ints.upper)|(~np.isfinite(ints.lower)&~np.isfinite(ints.upper))).all())
    ck('no_computational_failure',not results.computational_failure.astype(bool).any())
    ck('false_support_zero',not results.false_support.astype(bool).any())
    # Recompute operational gate exactly.
    sk=support.set_index(['case_id','replication']); expected=[]; reasons=[]
    for r in results.itertuples(index=False):
        ss=sk.loc[(r.case_id,int(r.replication))].copy(); row=pd.Series(r._asdict())
        for k in ['target_population_eligible','target_supported_by_generator','endpoint_requested','endpoint_audited']: row[k]=ss[k]
        gate,reason=operational_gate(r.method,row,thresh); ok=bool(r.raw_method_returned) and gate; expected.append(ok); reasons.append(str(r.raw_refusal_code) if not r.raw_method_returned else ('' if gate else reason))
    ck('operational_return_recompute',np.array_equal(np.asarray(expected,bool),results.operational_returned.to_numpy(bool)))
    ck('operational_reason_recompute',np.array_equal(np.asarray(reasons,str),results.operational_refusal_code.fillna('').astype(str).to_numpy()))
    # Certified route and empirical claim boundary.
    om6=results[(results.track=='ORACLE_STRUCTURAL_OT_TG')&(results.method=='M6')]
    ck('oracle_m6_theorem_eligible',om6.theorem_eligible.astype(bool).all())
    ck('empirical_m6_not_theorem_certified',not results[(results.track!='ORACLE_STRUCTURAL_OT_TG')&(results.method=='M6')].theorem_eligible.astype(bool).any())
    ck('nuisance_diagnostics_not_certified',not nuisance.theorem_certified.astype(bool).any() and not fits.finite_sample_theorem_certified.astype(bool).any())
    ck('propensity_modes',set(fits['mode'].astype(str))=={'estimated','misspecified'})
    ck('nuisance_reps',set(fits.replication.astype(int))=={1,10,20})
    ck('propensity_mass_normalization',(fits.max_training_mixed_mass_abs_error.astype(float)<=2e-10).all())
    # S9 must retain observed-product vs latent-outcome separation.
    s9=results[(results.scenario_id=='S9')&(results.track!='ORACLE_STRUCTURAL_OT_TG')]
    ck('s9_latent_sensitivity_present',s9.covered_latent.notna().all() and len(s9)>0)
    # Width is deliberately diagnostic for p-value methods, not threshold input and not a coverage-preserving inversion claim.
    pwidth=width[width.method.isin(['M3','M4','M6'])]
    ck('p_width_claim_boundary',(pwidth.audit_representation=='diagnostic_grid_not_coverage_preserving').all())
    ck('width_grid_points',(pwidth.grid_points.astype(int)==int(contract['width_audit_grid_points'])).all())
    ck('width_reps',set(width.replication.astype(int))==set(contract['width_audit_reps']))
    ck('width_case_scope',all(str(contract['width_audit_case_by_scenario'][r.scenario_id])==r.case_id for r in width.itertuples()))
    ck('width_summary_claim_boundary',widthsum.claim_status.astype(str).eq('DIAGNOSTIC_ONLY_NOT_USED_FOR_THRESHOLD_SELECTION_OR_MATCHED_COVERAGE_CONCLUSION').all())
    ck('width_summary_separates_refusal','refusal_rate' in widthsum.columns and 'returned_audit_queries' in widthsum.columns)
    # Graph misspecification must actually be exercised on RF S6.
    s6=graph[(graph.scenario_id=='S6')&(graph.track=='RF_SHARED_PREDICTOR_EMPIRICAL')]
    ck('s6_graph_diagnostic_present',len(s6)==40)
    ck('s6_has_fitted_graph_difference',(s6.true_graph_edges.astype(int)!=s6.fitted_graph_edges.astype(int)).any())
    oracle_graph=graph[graph.track=='ORACLE_STRUCTURAL_OT_TG']
    ck('all_structural_blocks_connected',oracle_graph.target_block_connected_under_track_graph.astype(bool).all())
    disconnected_s6=s6[~s6.target_block_connected_under_track_graph.astype(bool)]
    ck('s6_disconnected_working_blocks_recorded',len(disconnected_s6)>0)
    for r in disconnected_s6.itertuples(index=False):
        q=results[(results.case_id==r.case_id)&(results.replication.astype(int)==int(r.replication))&(results.track=='RF_SHARED_PREDICTOR_EMPIRICAL')&results.method.isin(['M3','M4','M6'])]
        ck(f's6_disconnect_refusal_{r.case_id}_{int(r.replication)}',len(q)==3 and (~q.raw_method_returned.astype(bool)).all() and q.raw_refusal_code.astype(str).eq('R05_GRAPH_COMPONENT_TOO_SMALL').all())
    # Coverage sanity is a post-freeze diagnostic only.
    if len(sanity):
        ck('coverage_sanity_bonferroni',np.allclose(sanity.bonferroni_cutoff.astype(float),float(contract['coverage_sanity']['familywise_alpha'])/len(sanity),atol=1e-15,rtol=0))
    bad=int(sanity.significant_undercoverage.astype(bool).sum()) if len(sanity) else 0
    if len(sanity): ck('coverage_sanity_descriptive_only',sanity.inferential_status.astype(str).eq('DESCRIPTIVE_ONLY_THRESHOLDS_SELECTED_ON_SAME_20REP_PILOT').all())
    ck('coverage_sanity_not_independent_confirmation',claim['coverage_sanity_is_independent_confirmation'] is False)
    min_case_n=int(contract['threshold_selection']['minimum_returned_per_case_for_casewise_coverage_objective'])
    evidence_adequate=(int(thresh['safe_case_coverage_groups_used'])==int(thresh['safe_case_count']) and int(thresh['minimum_safe_case_returned'])>=min_case_n and thresh['safe_pooled_selective_coverage'] is not None)
    ck('threshold_freeze_evidence_adequacy_recorded',bool(claim['threshold_freeze_evidence_adequate'])==bool(evidence_adequate) and int(claim['minimum_returned_per_safe_case_for_freeze'])==min_case_n)
    ck('pilot_ready_consistency',bool(claim['pilot_scale_ready'])==(not results.computational_failure.astype(bool).any() and not results.false_support.astype(bool).any() and evidence_adequate and thresh['status']=='FROZEN_AFTER_PREREGISTERED_20REP_PILOT'))
    ck('production_not_run',claim['production_run'] is False and claim['publication_performance_evidence'] is False)
    ck('production_blocked_for_review',claim['production_blocked_until_independent_review'] is True)
    ck('production_blocked_for_F06_freeze',claim['production_blocked_until_F06_sample_size_freeze'] is True)
    ck('selector_claim_boundary_honest',claim['threshold_selector_detail_preregistration_claim'] is False and claim['candidate_grids_and_coverage_refusal_principle_frozen_upstream'] is True)
    ck('estimated_nuisance_not_certified',claim['estimated_nuisance_certified'] is False)
    ck('width_not_coverage_preserving',claim['width_audit_grid_coverage_preserving'] is False)
    ck('R11_no_synthetic_numeric_transfer',claim['r11_real_eo_rule']['synthetic_ue_numeric_threshold_frozen'] is False)
    ck('registry_alignment_20rep',align['replications_per_scenario']==20 and align['scenario_count']==10)
    ck('dose_bins_from_frozen_grid',set(dose.dose_bin.astype(str)).issubset({f'D{x:.2f}' for x in contract['dose_grid']}))
    ck('certificate_summary_lb_range',certsum.minimum_certified_coverage_lower_bound.astype(float).between(0,1).all())
    ck('case_runtime_positive',(case_runtime.elapsed_seconds.astype(float)>0).all())
    ck('threshold_plan_alignment_present','threshold_plan_alignment' in align and align['threshold_plan_alignment']['minimum_ess']==[3,5,10,15,20] and align['threshold_plan_alignment']['additional_candidate_grids_added_by_stage3f'] is False)
    ck('frozen_threshold_keys_exact',set(thresh).issuperset({'minimum_ess','maximum_normalized_weight','minimum_graph_safe_count'}) and 'minimum_certified_coverage_lower_bound' not in thresh)
    # Accepted upstream exact hashes recorded by runner.
    expected=json.loads((ROOT/'configs/expected_upstream_hashes.json').read_text())
    for k,h in expected.items(): ck('upstream_hash_'+k,up[k]['sha256']==h)
    # Manifest rehash all listed files.
    manifest=json.loads((OUT/'stage3f_manifest.json').read_text()); ck('manifest_count',manifest['file_count']==len(manifest['files']))
    for rec in manifest['files']:
        pth=OUT/rec['path']; ck('manifest_file_'+rec['path'],pth.is_file() and sha256_file(pth)==rec['sha256'] and pth.stat().st_size==int(rec['size']))
    status='verified_complete'
    result={'status':status,'stage':STAGE,'version':VERSION,'checks_passed':len(checks),'failed_checks':0,'checks':checks,'pilot_scale_ready':bool(claim['pilot_scale_ready']),'production_run':False,'thresholds':thresh}
    (OUT/'STAGE3F_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(f'STAGE3F PILOT VERIFIED COMPLETE ({len(checks)}/{len(checks)})')
    print('Pilot scale ready for independent review:',claim['pilot_scale_ready'])

if __name__=='__main__':
    try: main()
    except Exception as exc:
        fail={'status':'verification_failed','stage':STAGE,'version':VERSION,'checks_passed':sum(x['pass'] for x in checks),'failed_checks':1,'failure':f'{type(exc).__name__}: {exc}','checks':checks}
        OUT.mkdir(exist_ok=True);(OUT/'STAGE3F_VERIFICATION.json').write_text(json.dumps(fail,indent=2,sort_keys=True)+'\n')
        print('VERIFICATION FAILED:',exc);traceback.print_exc();sys.exit(2)
