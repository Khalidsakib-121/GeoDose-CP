from __future__ import annotations
import json,math,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs_stage4'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from geodose_stage4.io import sha256_file
from geodose_stage4.version import STAGE,VERSION
from geodose_stage3f.pilot_math import operational_gate
checks=[]
def ck(name,cond,detail=''):
    ok=bool(cond);checks.append({'check':name,'pass':ok,'detail':str(detail)})
    if not ok:raise RuntimeError(f'{name}: {detail}')
def read(name):return pd.read_csv(OUT/name)
def bcol(s):
    if s.dtype==bool:return s
    return s.astype(str).str.lower().isin(['true','1','yes'])
def finite_or_nan(x):
    a=pd.to_numeric(x,errors='coerce').to_numpy(float);return np.isfinite(a)|np.isnan(a)
def asbool(v):
    if isinstance(v,(bool,np.bool_)):return bool(v)
    return str(v).strip().lower() in {'true','1','yes'}

def main():
    ck('output_dir_exists',OUT.is_dir())
    c=yaml.safe_load((ROOT/'configs/production_contract.yaml').read_text())
    runtime=json.loads((OUT/'stage4_runtime.json').read_text());claim=json.loads((OUT/'stage4_claim_boundary.json').read_text());regsum=json.loads((OUT/'stage4_registry_summary.json').read_text());th=json.loads((OUT/'stage4_frozen_operational_thresholds.json').read_text());f06freeze=json.loads((OUT/'stage4_f06_DGP_FREEZE.json').read_text());up=json.loads((OUT/'stage4_upstream_provenance_audit.json').read_text());srcinv=json.loads((OUT/'stage4_source_inventory.json').read_text())
    ck('official_not_qa_smoke',runtime.get('qa_smoke') is False and claim.get('qa_smoke') is False)
    ck('version_stage',runtime['version']==VERSION and claim['version']==VERSION and runtime['stage']==STAGE and claim['stage']==STAGE)
    ck('python_3_10',str(runtime['python']).startswith('3.10.'),runtime['python'])
    ck('thresholds_exact',(float(th['minimum_ess']),float(th['maximum_normalized_weight']),int(th['minimum_graph_safe_count']))==(3.0,.5,20),th)
    ck('thresholds_claim_exact',claim['stage3f_thresholds']==th)
    ck('no_rethresholding',claim['thresholds_retuned_from_production'] is False and c['threshold_retuning_allowed'] is False)
    ck('no_nsw_tuning',claim['nsw_used_for_tuning'] is False and c['nsw_tuning_allowed'] is False)
    ck('f06_freeze_byte_semantics',f06freeze==c['f06_freeze'])
    ck('f06_levels_exact',{k:(v['rows'],v['cols'],v['n']) for k,v in f06freeze['levels'].items()}=={'small':(17,17,289),'primary':(25,25,625),'large':(35,35,1225)})
    ck('f06_fixed_90m',f06freeze['resolution_m']==90 and f06freeze['adjacency']=='queen' and f06freeze['boundary']=='nonperiodic')
    ck('f06_not_s10_resolution',f06freeze['levels']['small']['n']==289 and 'not_resolution_or_MAUP' in f06freeze['scientific_claim'])
    ck('registry_summary_exact',regsum=={'main_rows':3000,'f06_extension_rows':600,'stress_extension_rows':590,'method_evaluable_rows':4090,'structural_only_rows':100,'total_case_replication_rows':4190},regsum)
    # Core tables.
    results=read('stage4_production_query_results.csv.gz');support=read('stage4_support_certificate_inputs.csv.gz');targets=read('stage4_target_selection_audit.csv');registry=read('stage4_frozen_production_registry.csv');fqa=read('stage4_f06_structural_QA.csv');graphs=read('stage4_graph_diagnostics.csv');outfits=read('stage4_outcome_fit_audit.csv');temporal=read('stage4_temporal_stress_diagnostic.csv');spatial=read('stage4_spatial_residual_diagnostics.csv');heur=read('stage4_heuristic_ablation_results.csv');metrics=read('stage4_production_metrics.csv');dose=read('stage4_dose_bin_metrics.csv');local=read('stage4_local_coverage_metrics.csv');localsum=read('stage4_local_coverage_summary.csv');cert=read('stage4_certificate_summary.csv');hero=read('stage4_s4_hero_results.csv');f06m=read('stage4_f06_sample_size_metrics.csv');abl=read('stage4_ablation_summary.csv');design=read('stage4_target_design_ablation.csv');cov=read('stage4_independent_coverage_confirmation.csv');width=read('stage4_m6_m5_width_audit.csv');eff=read('stage4_m6_m5_matched_coverage_efficiency.csv');prop=read('stage4_propensity_ablation_query_results.csv');propfits=read('stage4_propensity_ablation_fit_audit.csv');propsum=read('stage4_propensity_ablation_summary.csv');case_rt=read('stage4_case_rep_runtime.csv')
    ck('registry_rows',len(registry)==4190,len(registry));ck('support_rows',len(support)==4190,len(support));ck('target_rows',len(targets)==4190,len(targets));ck('spatial_diag_rows',len(spatial)==4190,len(spatial));ck('case_runtime_rows',len(case_rt)==4190,len(case_rt))
    ck('result_rows_exact',len(results)==52740,len(results));ck('runtime_result_rows',int(runtime['result_rows'])==52740);ck('runtime_support_rows',int(runtime['support_rows'])==4190)
    # Registry scopes and main allocation exact.
    ck('registry_scope_counts',registry.scope.value_counts().to_dict()=={'main_production':3000,'f06_extension':600,'registered_stress_extension':590},registry.scope.value_counts().to_dict())
    expected={(a['case_id'],int(a['seed_rep_end'])-int(a['seed_rep_start'])+1) for a in c['main_production_allocation']};main=registry[registry.scope=='main_production'];got={(cid,len(g)) for cid,g in main.groupby('case_id')};ck('main_case_allocation_exact',got==expected,(len(got),len(expected)))
    ck('main_seed_pairs_unique',len(main[['scenario_id','seed_replication']].drop_duplicates())==3000)
    # F06 mapping: small/large use corresponding main master seed window.
    fr=registry[registry.scope=='f06_extension'];ck('f06_registry_600',len(fr)==600)
    for cid,start in [('S1_BASE',1),('S4_RHO060',301),('S5_POOR_OVERLAP',1)]:
        for level in ['small','large']:
            q=fr[(fr.case_id==cid)&(fr.f06_level==level)].sort_values('case_local_index');ck(f'f06_{cid}_{level}_100',len(q)==100);ck(f'f06_{cid}_{level}_seed_window',q.seed_replication.astype(int).tolist()==list(range(start,start+100)))
    # F06 structural QA is structural only and must all pass.
    ck('f06_qa_three_levels',len(fqa)==3 and set(fqa.level)=={'small','primary','large'})
    bool_q=[x for x in fqa.columns if x.endswith('_exact') or x.startswith('no_') or x in ['full_graph_connected','role_cut_graph_three_components']]
    for col in bool_q:ck('f06qa_'+col,bcol(fqa[col]).all())
    ck('f06_primary_geometry',tuple(fqa[fqa.level=='primary'][['n','full_queen_edges','role_cut_dgp_edges','nuisance_training','support_audit','calibration','test_target']].iloc[0].astype(int))==(625,2352,2206,250,125,125,125))
    # Result track counts are algorithmically determined by the frozen contract.
    tc=results.track.value_counts().to_dict();expected_tracks={'ORACLE_STRUCTURAL_OT_TG':24540,'RF_SHARED_PREDICTOR_EMPIRICAL':18000,'XGB_SHARED_PREDICTOR_EMPIRICAL':8400,'RF_ORACLE_SPATIAL_LAW_DIAGNOSTIC':1800};ck('result_track_counts',tc==expected_tracks,tc)
    ck('oracle_method_counts',all(len(results[(results.track=='ORACLE_STRUCTURAL_OT_TG')&(results.method==m)])==4090 for m in ['M1','M2','M3','M4','M5','M6']))
    ck('main_oracle_rows',len(results[(results.production_scope=='main_production')&(results.track=='ORACLE_STRUCTURAL_OT_TG')])==18000)
    ck('f06_oracle_rows',len(results[(results.production_scope=='f06_extension')&(results.track=='ORACLE_STRUCTURAL_OT_TG')])==3600)
    ck('stress_oracle_rows',len(results[(results.production_scope=='registered_stress_extension')&(results.track=='ORACLE_STRUCTURAL_OT_TG')])==2940)
    ck('no_computational_failures',not bcol(results.computational_failure).any())
    ck('no_operational_false_support',not bcol(results.false_support).any())
    ck('all_pvalues_valid',pd.to_numeric(results.pvalue_conservative,errors='coerce').dropna().between(0,1).all() and pd.to_numeric(results.pvalue_randomized,errors='coerce').dropna().between(0,1).all())
    ck('finite_or_nan_numeric_outputs',finite_or_nan(results.width).all() and finite_or_nan(results.ess).all() and finite_or_nan(results.operational_coverage_lower_bound).all())
    # Recompute the frozen operational gate for every scientific result row.
    mismatch=0;reason_mismatch=0
    for r in results.itertuples(index=False):
        row=pd.Series(r._asdict());gate,reason=operational_gate(str(r.method),row,th);expected_return=asbool(r.raw_method_returned) and bool(gate)
        if expected_return!=asbool(r.operational_returned):mismatch+=1
        expected_reason=str(r.raw_refusal_code) if not asbool(r.raw_method_returned) else ('' if gate else reason)
        actual='' if pd.isna(r.operational_refusal_code) else str(r.operational_refusal_code)
        if actual!=expected_reason:reason_mismatch+=1
    ck('operational_gate_recompute_zero_mismatch',mismatch==0,mismatch);ck('operational_refusal_reason_recompute_zero_mismatch',reason_mismatch==0,reason_mismatch)
    # Target selection must be structural/outcome-blind and blocks unique by frozen eligible set hash.
    ck('target_selection_no_A',not bcol(targets.selection_uses_A).any());ck('target_selection_no_Y',not bcol(targets.selection_uses_Y).any());ck('target_selection_no_support_diag',not bcol(targets.selection_uses_support_diagnostics).any())
    ck('target_selection_block_nodes_present',targets.block_nodes.astype(str).str.len().gt(0).all())
    # Diagnostics counts and boundaries.
    ck('temporal_100',len(temporal)==100);ck('temporal_no_method_claim',not bcol(temporal.method_level_temporal_conformal_claim).any())
    ck('spatial_diagnostic_only',bcol(spatial.diagnostic_only).all() and not bcol(spatial.used_in_conformal_deficit).any())
    ck('moran_semivariogram_finite',np.isfinite(pd.to_numeric(spatial.morans_I,errors='coerce')).all() and np.isfinite(pd.to_numeric(spatial.empirical_semivariogram_true_graph_edges,errors='coerce')).all())
    ck('heuristic_rows',len(heur)==8100,len(heur));ck('heuristics_never_theorem',not bcol(heur.theorem_eligible).any());ck('heuristic_ids',set(heur.heuristic_id)=={'H1_GEOGRAPHIC','H4_DOSE_X_GEOGRAPHIC'})
    ck('graph_rows',len(graphs)==8750,len(graphs));ck('outcome_fit_rows',len(outfits)==4700,len(outfits))
    # S6 fitted-graph failures are refusal, not retargeting/computation failures.
    s6g=graphs[(graphs.scenario_id=='S6')&(graphs.track=='RF_SHARED_PREDICTOR_EMPIRICAL')];disc=s6g[~bcol(s6g.target_block_connected_under_track_graph)];ck('s6_working_graph_disconnection_exercised',len(disc)>0,len(disc))
    for r in disc.itertuples(index=False):
        q=results[(results.case_id==r.case_id)&(results.replication.astype(int)==int(r.replication))&(results.track=='RF_SHARED_PREDICTOR_EMPIRICAL')&results.method.isin(['M3','M4','M6'])]
        ck(f's6_disconnect_failclosed_{r.case_id}_{int(r.replication)}',len(q)==3 and (~bcol(q.raw_method_returned)).all() and q.raw_refusal_code.astype(str).eq('R05_GRAPH_COMPONENT_TOO_SMALL').all())
    # ST1 unaudited endpoints must fail closed exactly.
    una=results[results.extension_id.astype(str).str.contains('EXT_ST1_UNAUDITED',na=False)];ck('unaudited_endpoint_rows',len(una)==240,len(una));ck('unaudited_endpoint_R02',(~bcol(una.raw_method_returned)).all() and una.raw_refusal_code.astype(str).eq('R02_ENDPOINT_NOT_AUDITED').all())
    # ST3 is an identification negative control, never presented as conformal repair.
    st3=results[results.extension_id.astype(str).eq('EXT_ST3_HIDDEN_C2')];ck('st3_rows',len(st3)==600);ck('st3_claim_boundary','causal_identification_negative_control' in claim['ST3_interpretation'])
    # Nonidentity target-design stress is diagnostic only.
    ds=results[results.analysis_case_id.astype(str).str.contains('DESIGN_SHIFT',na=False)];ck('design_shift_rows',len(ds)==600,len(ds));ck('design_shift_not_theorem',not bcol(ds.theorem_eligible).any());ck('design_shift_no_certified_performance_claim',pd.to_numeric(ds.certified_coverage_lower_bound,errors='coerce').isna().all())
    ck('target_design_ablation_100',len(design)==100,len(design));ck('target_design_ablation_no_pvalue_claim',not bcol(design.pvalue_ablation_claim).any());ck('target_design_constant_loss_cancellation',np.allclose(pd.to_numeric(design.difference),0,atol=1e-15,rtol=0))
    # Primary metrics and local/dose summaries.
    ck('metrics_nonempty',len(metrics)>0);ck('metrics_rates_range',metrics.raw_coverage.astype(float).between(0,1).all() and metrics.refusal_rate.astype(float).between(0,1).all())
    ck('dose_metrics_nonempty',len(dose)>0);ck('dose_rates_range',dose.raw_coverage.astype(float).between(0,1).all() and dose.refusal_rate.astype(float).between(0,1).all())
    ck('local_metrics_nonempty',len(local)>0 and len(localsum)>0);ck('local_band_labels',set(local.target_row_band).issubset({'low_row','middle_row','high_row'}));ck('local_fifth_percentile_range',pd.to_numeric(localsum.fifth_percentile_local_coverage,errors='coerce').dropna().between(0,1).all())
    ck('certificate_summary_nonempty',len(cert)>0);ck('certified_lb_range',pd.to_numeric(cert.minimum_certified_coverage_lower_bound,errors='coerce').between(0,1).all())
    # S4 hero is exactly 5 rho x 6 methods x 100 independent production reps.
    ck('hero_rows_30',len(hero)==30,len(hero));ck('hero_rho_grid',set(np.round(hero.rho.astype(float),10))=={0,.2,.4,.6,.8});ck('hero_methods',set(hero.method)=={'M1','M2','M3','M4','M5','M6'});ck('hero_requested_100',(hero.requested.astype(int)==100).all());ck('hero_mcse_recompute',np.allclose(hero.coverage_mcse.astype(float),np.sqrt(hero.raw_coverage.astype(float)*(1-hero.raw_coverage.astype(float))/100),atol=1e-15,rtol=1e-12))
    # F06 results: 3 bases x 3 sizes x 6 methods; 100 each.
    ck('f06_metric_rows_54',len(f06m)==54,len(f06m));ck('f06_metric_levels',set(f06m.f06_level)=={'small','primary','large'});ck('f06_metric_methods',set(f06m.method)=={'M1','M2','M3','M4','M5','M6'});ck('f06_metric_requested_100',(f06m.requested.astype(int)==100).all())
    # Ablations ABL0-ABL9 all have an explicit output route (ABL2/ABL8 separate tables).
    ck('ablation_main_ids',{'ABL0','ABL1','ABL3','ABL4','ABL5','ABL6','ABL7','ABL9'}.issubset(set(abl.ablation_id)))
    ck('abl9_present',len(abl[abl.ablation_id=='ABL9'])==24,len(abl[abl.ablation_id=='ABL9']));ck('abl9_graph_tracks',set(abl[abl.ablation_id=='ABL9'].graph_track.dropna())=={'RF_SHARED_PREDICTOR_EMPIRICAL','RF_ORACLE_SPATIAL_LAW_DIAGNOSTIC'})
    ck('propensity_diag_rows_800',len(prop)==800,len(prop));ck('propensity_fit_rows_200',len(propfits)==200,len(propfits));ck('propensity_summary_rows_80',len(propsum)==80,len(propsum));ck('propensity_never_theorem',not bcol(prop.theorem_certified).any() and not bcol(propfits.finite_sample_theorem_certified).any() and not bcol(propsum.theorem_certified).any())
    ck('propensity_modes',set(prop.propensity_mode)=={'estimated','misspecified'});ck('propensity_methods',set(prop.method)=={'M2','M4','M5','M6'})
    # Coverage confirmation is independent because thresholds were frozen in Stage3F.
    if len(cov):
        ck('coverage_confirmation_status',cov.inferential_status.astype(str).eq('INDEPENDENT_PRODUCTION_CONFIRMATION_THRESHOLDS_FROZEN_ON_STAGE3F').all());ck('coverage_not_retuned',(~bcol(cov.thresholds_retuned_from_production)).all());ck('coverage_bonferroni',np.allclose(cov.bonferroni_cutoff.astype(float),float(c['coverage_confirmation']['familywise_alpha'])/len(cov),atol=1e-15,rtol=0))
    # Width/efficiency claim is finite-domain only. M6 must have exactly one row per fixed query.
    ck('width_m6_50',len(width[width.method=='M6'])==50,len(width[width.method=='M6']));ck('width_full_line_false',not bcol(width.full_real_line_claim.fillna(False)).any());ck('eff_rows_5',len(eff)==5,len(eff));ck('eff_full_line_false',not bcol(eff.full_real_line_efficiency_claim).any());ck('eff_coverage_match_arithmetic',all(bool(r.coverage_matched)==(np.isfinite(float(r.absolute_coverage_difference)) and float(r.absolute_coverage_difference)<=float(r.matched_coverage_tolerance)+1e-15) for r in eff.itertuples(index=False)))
    if bcol(width[width.method=='M6'].domain_truncated.fillna(False)).any():ck('truncated_M6_no_full_line_claim',not bcol(eff.full_real_line_efficiency_claim).any())
    # Runtime/checkpoint completeness.
    ck('case_runtime_positive',(case_rt.elapsed_seconds.astype(float)>0).all());ck('checkpoint_count',int(runtime['checkpoint_files'])==4190,runtime['checkpoint_files'])
    # Claim boundary must remain conservative.
    ck('production_evidence_true',claim['production_performance_evidence'] is True and claim['production_main_rows_generated']==3000)
    ck('estimated_nuisance_not_certified',claim['estimated_nuisance_finite_sample_certified'] is False)
    ck('design_shift_theorem_not_claimed',claim['design_shift_performance_theorem_claim'] is False)
    ck('st2_theorem_not_claimed',claim['ST2_method_level_temporal_conformal_claim'] is False)
    ck('m6_full_line_width_not_claimed',claim['M6_full_real_line_width_claim'] is False)
    ck('next_stage_blocked_for_review',claim['next_stage_blocked_until_independent_review'] is True)
    # Accepted upstream hashes and source counts recorded by scientific runner.
    expected=json.loads((ROOT/'configs/expected_upstream_hashes.json').read_text())
    for k,h in expected.items():ck('upstream_'+k,up[k]['sha256']==h)
    locks=json.loads((ROOT/'configs/source_locks.json').read_text())
    for k,lock in locks.items():ck('source_lock_'+k,int(up[k]['verified_files'])==len(lock['files']))
    # Source inventory: exact current package bytes at time of run (machine-resolved path excluded).
    current_files={r['path']:r for r in srcinv['files']};ck('source_inventory_count',int(srcinv['file_count'])==len(current_files));ck('resolved_paths_excluded','configs/resolved_paths_windows.json' not in current_files)
    for rel,rec in current_files.items():
        p=ROOT/rel;ck('source_file_'+rel,p.is_file() and sha256_file(p)==rec['sha256'] and p.stat().st_size==int(rec['size']))
    # Manifest: independently rehash every tracked output.
    manifest=json.loads((OUT/'stage4_manifest.json').read_text());ck('manifest_count',manifest['file_count']==len(manifest['files']))
    for rec in manifest['files']:
        p=OUT/rec['path'];ck('manifest_'+rec['path'],p.is_file() and sha256_file(p)==rec['sha256'] and p.stat().st_size==int(rec['size']))
    flags={'significant_independent_undercoverage_groups':int(bcol(cov.significant_undercoverage).sum()) if len(cov) else 0,'m6_width_queries_domain_truncated':int(bcol(width[width.method=='M6'].domain_truncated.fillna(False)).sum()),'matched_coverage_s4_groups':int(bcol(eff.coverage_matched).sum()),'scientific_note':'flags are results for independent review, not software-verification failures'}
    result={'status':'verified_complete','stage':STAGE,'version':VERSION,'checks_passed':len(checks),'failed_checks':0,'checks':checks,'scientific_flags':flags,'production_main_rows':3000,'total_case_replication_rows':4190,'result_rows':52740,'thresholds':th,'next_stage_blocked_until_independent_review':True}
    (OUT/'STAGE4_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(f'STAGE4 PRODUCTION VERIFIED COMPLETE ({len(checks)}/{len(checks)})');print('Scientific review flags:',flags)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True);fail={'status':'verification_failed','stage':STAGE,'version':VERSION,'checks_passed':sum(x['pass'] for x in checks),'failed_checks':1,'failure':f'{type(exc).__name__}: {exc}','checks':checks};(OUT/'STAGE4_VERIFICATION.json').write_text(json.dumps(fail,indent=2,sort_keys=True)+'\n');print('VERIFICATION FAILED:',exc);traceback.print_exc();sys.exit(2)
