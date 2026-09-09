from __future__ import annotations
from pathlib import Path
import sys,json,hashlib,math,re,zipfile
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from geodose_stage5b.common import require,read_json,sha256_file,aggregate_manifest,Stage5BError
from geodose_stage5b.authority import verify_authorities

checks=[]
def ck(name,cond,detail=''):
    if not bool(cond):raise Stage5BError(f'VERIFY FAIL {name}: {detail}')
    checks.append(name);print('PASS',name+(f': {detail}' if detail else ''))

def readgz(name):return pd.read_csv(ROOT/'STAGE5B_PRODUCTION_OUTPUTS'/name,compression='gzip',low_memory=False)

def main():
    out=ROOT/'STAGE5B_PRODUCTION_OUTPUTS';ck('outputs_present',out.exists())
    contract=read_json(ROOT/'configs/stage5b_contract.json');exp=contract['expected_counts'];auth=read_json(ROOT/'configs/expected_authorities.json')
    report=read_json(out/'STAGE5B_PRODUCTION_REPORT.json');claim=read_json(out/'STAGE5B_CLAIM_BOUNDARY.json')
    ck('report_status',report['status']=='STAGE5B_PRODUCTION_COMPLETE_PENDING_INDEPENDENT_AUDIT')
    ck('report_version',report['version']==contract['version'])
    ck('stage5a_sha',report['stage5a_release_sha256']==auth['stage5a_release_zip_sha256'])
    ck('stage5a_audit_sha',report['stage5a_audit_sha256']==auth['stage5a_audit_sha256'])
    ck('no_threshold_retune',report['thresholds_retuned'] is False)
    ck('no_hyperparameter_retune',report['hyperparameters_retuned_from_production'] is False)
    ck('stage5a_immutable',report['stage5a_modified'] is False)
    ck('refusal_separate',report['refusals_counted_as_noncoverage'] is False)
    ck('primary_methods_exact',report['primary_methods']==['M2','M3','M4','M5','M6'])
    ck('domain_exact',report['candidate_domain']==[-1.0,1.0])
    # Manifest.
    man=read_json(out/'OUTPUT_FILE_MANIFEST.json');rows=man['files'];
    for r in rows:
        p=out/r['file'];ck('manifest_present::'+r['file'],p.exists());ck('manifest_size::'+r['file'],p.stat().st_size==int(r['bytes']));ck('manifest_sha::'+r['file'],sha256_file(p)==r['sha256'])
    ck('manifest_aggregate',aggregate_manifest(rows)==man['aggregate_sha256'],man['aggregate_sha256'])
    sm=read_json(out/'SCIENTIFIC_OUTPUT_MANIFEST.json');ck('scientific_manifest_aggregate',aggregate_manifest(sm['files'])==sm['aggregate_sha256']);ck('scientific_manifest_excludes_runtime',all(r['file'] not in {'STAGE5B_RUNTIME.csv','STAGE5B_PRODUCTION_REPORT.json'} for r in sm['files']))
    # Raw tables and exact counts.
    q=readgz('STAGE5B_QUERY_RESULTS.csv.gz');ex=readgz('STAGE5B_EXACT_AUDIT_RESULTS.csv.gz');inv=readgz('STAGE5B_FULL_INVERSION_RESULTS.csv.gz');ep=readgz('STAGE5B_ENDPOINT_AUDIT_RESULTS.csv.gz');dose=readgz('STAGE5B_DOSE_RESPONSE_PREDICTIONS.csv.gz');fit=readgz('STAGE5B_NUISANCE_FIT_AUDIT.csv.gz');sp=readgz('STAGE5B_SPATIAL_DIAGNOSTICS.csv.gz');lo=readgz('STAGE5B_LOMO_RESULTS.csv.gz');rt=pd.read_csv(out/'STAGE5B_RUNTIME.csv')
    got={'query_rows':len(q),'exact_audit_rows':len(ex),'full_inversion_rows':len(inv),'endpoint_audit_rows':len(ep),'dose_response_prediction_rows':len(dose),'nuisance_fit_rows':len(fit),'lomo_rows':len(lo),'runtime_rows':len(rt),'spatial_diagnostic_rows':len(sp)}
    for k,v in got.items():ck('count::'+k,int(v)==int(exp[k]),f'{v}')
    ck('runtime_tasks_unique',len(rt[['case_id','replication']].drop_duplicates())==540)
    ck('runtime_no_failure',not rt.computational_failure.astype(bool).any())
    # Query key contract.
    key=['case_id','replication','MineID','target_rank','track','method','truth_scale'];ck('query_unique_key',not q.duplicated(key).any())
    ck('methods_only_M2_M6',set(q.method)=={'M2','M3','M4','M5','M6'})
    ck('no_M1_H1_H4',not q.method.astype(str).isin(['M1','H1','H4']).any())
    ck('truth_scales',set(q.truth_scale).issubset({'observed','latent'}))
    ck('latent_only_S9',set(q.loc[q.truth_scale=='latent','scenario_id'])=={'S9'})
    ck('XGB_scope',set(q.loc[q.track=='XGB_ET_FG_CONFIRMATION','scenario_id']).issubset({'S1','S4','S5','S8'}))
    ck('RF_all_cases',q[q.track=='RF_ET_FG_PRIMARY'].case_id.nunique()==27)
    ck('oracle_all_cases',q[q.track=='ORACLE_OT_TG_DIAGNOSTIC'].case_id.nunique()==27)
    ck('target_ranks_1_5',set(q.target_rank.astype(int))=={1,2,3,4,5})
    # Refusal semantics.
    ck('operational_return_code_equivalence',((q.operational_return.astype(bool))==(q.refusal_code.fillna('').astype(str)=='')).all())
    ck('covered_implies_return',not q.loc[~q.operational_return.astype(bool),'covered'].astype(bool).any())
    ck('false_support_definition',((q.false_support.astype(bool))==(q.operational_return.astype(bool)&~q.oracle_support_truth.astype(bool))).all())
    ck('comp_failure_code',((q.computational_failure.astype(bool))==(q.refusal_code.fillna('').astype(str)=='R10_COMPUTATIONAL_FAILURE')).all())
    ck('M3_no_ESS_refusal',not q[(q.method=='M3')].refusal_code.fillna('').isin(['R04_ESS_BELOW_FROZEN_THRESHOLD','R05_MAX_WEIGHT_ABOVE_FROZEN_THRESHOLD']).any())
    ck('N3_refusal_M6_only',set(q.loc[q.refusal_code.fillna('')=='R08_N3_DIAGNOSTIC_VACUOUS','method']).issubset({'M6'}))
    ck('M4_status',set(q.loc[q.method=='M4','method_family_status'])=={'naive_non_theorem_comparator'})
    ck('M6_status',set(q.loc[q.method=='M6','method_family_status'])=={'GeoDose_CP'})
    ck('estimated_N3_not_theorem_certified',not q[q.track!='ORACLE_OT_TG_DIAGNOSTIC'].n3_theorem_certified.astype(bool).any())
    # Numeric and physical truth.
    ck('truth_finite',np.isfinite(pd.to_numeric(q.truth_y,errors='coerce')).all())
    ck('truth_physical',((q.truth_y>=-1)&(q.truth_y<=1)).all())
    ck('Astar_01',((q.A_star>=0)&(q.A_star<=1)).all())
    pvals=pd.to_numeric(q.candidate_pvalue,errors='coerce');ck('pvalues_01',((pvals.dropna()>=0)&(pvals.dropna()<=1)).all())
    ck('M2M5_interval_order',((q.loc[q.method.isin(['M2','M5'])&q.operational_return.astype(bool),'lower']<=q.loc[q.method.isin(['M2','M5'])&q.operational_return.astype(bool),'upper'])).all())
    # N2/N3 diagnostic discipline.
    ck('sparse_diag_nonnegative',(pd.to_numeric(q.n2_localization_kl,errors='coerce').dropna()>=-1e-12).all())
    tv=pd.to_numeric(q.n2_pinsker_tv,errors='coerce').dropna();ck('sparse_tv_01',((tv>=0)&(tv<=1)).all())
    ck('claim_estimated_not_certified',claim['estimated_nuisance_finite_sample_theorem_certified'] is False)
    ck('claim_n2_diagnostic_label','not relabelled' in claim['n2_localization_metric_status'])
    ck('claim_M4_non_theorem','non-theorem' in claim['M4_status'])
    ck('claim_ST2_structural','structural' in claim['ST2_status'])
    ck('claim_ST3_negative','negative control' in claim['ST3_status'])
    ck('claim_no_full_real_line',claim['full_real_line_claim'] is False)
    # Fits and predictor leakage.
    ck('fit_unique',not fit.duplicated(['case_id','replication','track']).any())
    ck('fit_info_separation',not fit[['support_outcomes_used_for_fit','calibration_outcomes_used_for_fit','test_outcomes_used_for_fit']].astype(bool).any().any())
    forbidden={'mapped_rehabilitation_fraction','pv_median_2025','benchmark_role','hidden_u_truth','Y_true_at_A','Y_observed_at_A'}
    features=set('|'.join(fit.feature_list.astype(str)).split('|'));ck('fit_forbidden_predictors_absent',not (forbidden&features),str(forbidden&features))
    ck('fit_train_positive',(fit.n_train>20).all())
    ck('fit_theorem_status_false',not fit.empirical_nuisance_theorem_certified.astype(bool).any())
    # Spatial diagnostics are diagnostic only.
    ck('spatial_diag_only',sp.diagnostic_only.astype(bool).all())
    ck('spatial_diag_types',set(sp.diagnostic)=={'morans_I','semivariogram'})
    # Exact audit.
    ck('exact_methods',set(ex.method)=={'M3','M4','M6'})
    ck('exact_refs',set(ex.reference)=={'full_graph_exact','sparse_m64'})
    ck('exact_unique',not ex.duplicated(['case_id','replication','MineID','method','reference']).any())
    closure=read_json(out/'STAGE5B_EXACT_PVALUE_NUMERICAL_CLOSURE.json');ck('exact_roundoff_closure_patch',closure['patch_id']=='stage5b-v1.0.2-exact-pvalue-roundoff-canonicalization');ck('exact_roundoff_only',float(closure['maximum_absolute_correction'])<=1e-12 and closure['scientific_contract_changed'] is False and closure['thresholds_changed'] is False and closure['coverage_changed'] is False,str(closure))
    ck('exact_p_finite',np.isfinite(pd.to_numeric(ex.pvalue,errors='coerce')).all())
    ck('exact_p_01',((ex.pvalue>=0)&(ex.pvalue<=1)).all())
    ck('exact_kl_nonnegative',(ex.full_vs_sparse_orbit_kl>=-1e-12).all())
    ck('exact_no_failure',not ex.computational_failure.astype(bool).any())
    # Full inversion.
    ck('inversion_methods',set(inv.method)=={'M3','M4','M6'})
    ck('inversion_RF_only',set(inv.track)=={'RF_ET_FG_PRIMARY'})
    ck('inversion_cases',set(inv.case_id)==set(contract['full_inversion']['case_ids']))
    ck('inversion_reps',set(inv.replication.astype(int))=={1,20})
    ck('inversion_rank1',set(inv.target_rank.astype(int))=={1})
    ck('inversion_unique',not inv.duplicated(['case_id','replication','MineID','method']).any())
    ck('inversion_domain',((inv.hull_lower.dropna()>=-1-1e-9)&(inv.hull_upper.dropna()<=1+1e-9)).all())
    ck('inversion_width_order',(inv.raw_set_width<=inv.hull_width+1e-10).all())
    ck('inversion_gate_recorded','operational_return' in inv.columns and 'refusal_code' in inv.columns)
    # Endpoint audit.
    ck('endpoint_cases',set(ep.case_id)=={'MDB_ST1_MIXED_ATOMS','MDB_ST1_INTERIOR_ONLY'})
    ck('endpoint_doses',set(ep.endpoint_dose.astype(float))=={0.0,1.0})
    interior=ep[ep.case_id=='MDB_ST1_INTERIOR_ONLY'];mixed=ep[ep.case_id=='MDB_ST1_MIXED_ATOMS']
    ck('interior_endpoint_expected_refusal',interior.expected_refusal.astype(bool).all())
    ck('interior_endpoint_all_refused',not interior.operational_return.astype(bool).any())
    ck('interior_endpoint_code',set(interior.refusal_code)=={'R02_ENDPOINT_NOT_AUDITED'})
    ck('mixed_endpoint_audited',mixed.endpoint_audited.astype(bool).all())
    # Dose response.
    ck('dose_grid',set(np.round(dose.dose.astype(float),12))=={0.0,.1,.25,.5,.75,.9,1.0})
    ck('dose_finite',np.isfinite(dose[['prediction','Y_true_hard_dose','error']].to_numpy(float)).all())
    ck('hard_truth_physical',((dose.Y_true_hard_dose>=-1)&(dose.Y_true_hard_dose<=1)).all())
    # LOMO.
    ck('lomo_cases',set(lo.case_id)==set(contract['lomo']['case_ids']))
    ck('lomo_reps',set(lo.replication.astype(int))==set(contract['lomo']['replications']))
    loc=lo[lo.method.isin(['M3','M4','M6'])];ck('lomo_local_orbit_refuses',not loc.operational_return.astype(bool).any());ck('lomo_refusal_code',set(loc.refusal_code)=={'R18_LOMO_DISCONNECTED_LOCAL_ORBIT'})
    # Summary recomputation: selective coverage excludes refusals.
    cov=pd.read_csv(out/'SUMMARY_COVERAGE.csv')
    sample=cov.iloc[::max(1,len(cov)//25)].head(25)
    for i,r in sample.iterrows():
        g=q[(q.case_id==r.case_id)&(q.scenario_id==r.scenario_id)&(q.track==r.track)&(q.method==r.method)&(q.truth_scale==r.truth_scale)];ret=g.operational_return.astype(bool);val=float(g.loc[ret,'covered'].astype(bool).mean()) if ret.any() else np.nan
        ck(f'coverage_recompute::{i}',(pd.isna(val) and pd.isna(r.selective_coverage)) or abs(val-float(r.selective_coverage))<1e-12)
    match=pd.read_csv(out/'SUMMARY_S4_MATCHED_COVERAGE_EFFICIENCY.csv');ck('matched_coverage_rule',((match.coverage_matched.astype(bool))==(match.coverage_difference<=float(contract['full_inversion']['coverage_match_tolerance'])+1e-15)).all())
    ck('efficiency_claim_rule',((match.efficiency_claim_allowed.astype(bool))==match.coverage_matched.astype(bool)).all())
    pe=pd.read_csv(out/'SUMMARY_S4_M6_PAIRED_EFFECTS.csv');ck('paired_effect_rows',len(pe)==20);ck('paired_effect_comparators',set(pe.comparator)=={'M2','M3','M4','M5'});ck('paired_effect_rhos',set(np.round(pe.spatial_rho.astype(float),12))=={0.,.2,.4,.6,.8})
    # Figures.
    figdir=out/'figures';png=list(figdir.glob('*.png'));pdf=list(figdir.glob('*.pdf'));ck('figure_png_count',len(png)==6,str(len(png)));ck('figure_pdf_count',len(pdf)==6,str(len(pdf)));ck('figure_nonempty',all(p.stat().st_size>1000 for p in png+pdf))
    # Source scan: no network and no post-hoc threshold-selection machinery.
    source='\n'.join(p.read_text(encoding='utf-8',errors='ignore') for p in (ROOT/'src/geodose_stage5b').glob('*.py'))
    ck('no_network_requests','requests.' not in source and 'urllib.' not in source and 'http://' not in source and 'https://' not in source)
    ck('no_mapped_rehab_source','mapped_rehabilitation_fraction' not in (ROOT/'src/geodose_stage5b/evaluation.py').read_text())
    ck('no_2025_source','pv_median_2025' not in (ROOT/'src/geodose_stage5b/evaluation.py').read_text())
    ck('no_auto_threshold_selection','candidate_threshold' not in source and 'threshold_grid' not in source)
    # Source inventory hashes its claimed evaluator source.
    sinv=read_json(out/'STAGE5B_SOURCE_INVENTORY.json');ck('source_inventory_aggregate',aggregate_manifest(sinv['files'])==sinv['aggregate_sha256'])
    for r in sinv['files']:
        p=ROOT/r['file'];ck('source_inventory_present::'+r['file'],p.exists());ck('source_inventory_sha::'+r['file'],sha256_file(p)==r['sha256'])
    result={'status':'STAGE5B_VERIFIED_COMPLETE_PENDING_INDEPENDENT_AUDIT','stage':'Stage5B','version':contract['version'],'checks_passed':len(checks),'checks_failed':0,'stage5a_release_sha256':auth['stage5a_release_zip_sha256'],'query_rows':len(q),'M2_M6_performance_evaluated':True,'independent_audit_required_before_next_stage':True}
    (out/'STAGE5B_VERIFICATION.json').write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
    print(f"STAGE5B VERIFIED COMPLETE ({len(checks)}/{len(checks)})")
if __name__=='__main__':main()
