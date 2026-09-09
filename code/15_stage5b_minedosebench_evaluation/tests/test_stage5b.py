from pathlib import Path
import sys,json,hashlib,math,tempfile
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from geodose_stage5b.common import require,read_json,sha256_file,atomic_json_gz,read_json_gz,interval_score
from geodose_stage5b.authority import verify_authorities
from geodose_stage5b.data import BenchmarkData
from geodose_stage5b.evaluation import _refusal,_oracle_support_truth,_weight_diagnostics,_canonical_probability
from geodose_stage5b.orbit import eval_m3,eval_m4,eval_m6,eval_m3_m4_grid,eval_m6_grid,invert_candidate,orbit_kl
from geodose_stage5b.runner import _canonicalize_exact_audit_frame
from minedosebench.math import q_at

PASS=[]
def ck(name,cond):
    require(cond,name);PASS.append(name);print('PASS',name)

def main():
    c=read_json(ROOT/'configs/stage5b_contract.json');a=read_json(ROOT/'configs/expected_authorities.json')
    work=ROOT/'_UNIT_TEST_WORK';stage5a,_=verify_authorities(ROOT,work)
    ck('stage5a_release_sha',sha256_file(ROOT/'inputs/GeoDose_MineDoseBench_v1_2_0_RELEASE_CANDIDATE.zip')==a['stage5a_release_zip_sha256'])
    ck('stage5a_audit_sha',sha256_file(ROOT/'inputs/Stage5A_MineDoseBench_v1_2_0_Independent_Freeze_Audit.md')==a['stage5a_audit_sha256'])
    b=BenchmarkData(stage5a)
    ck('scenarios_27',len(b.scenarios)==27 and b.scenarios.case_id.nunique()==27)
    ck('seeds_540',len(b.seeds)==540)
    ck('targets_13500',len(b.targets)==13500)
    ck('exact_targets_2700',len(b.exact_targets)==2700)
    ck('substrate_90_27042',len(b.s90)==27042)
    ck('retained_23618',int(b.s90.benchmark_baseline_eligible.sum())==23618)
    ck('substrate_180_7295',len(b.s180)==7295)
    ck('eligible_180_5788',int(b.s180.benchmark_baseline_eligible.sum())==5788)
    ck('allowed_pretreatment',(b.allowed.timing.astype(str)=='pretreatment').all())
    forbidden={'mapped_rehabilitation_fraction','pv_median_2025','benchmark_role','hidden_u_truth','Y_true_at_A','Y_observed_at_A'}
    ck('allowed_no_forbidden',not (forbidden&set(b.allowed.predictor.astype(str))))
    # Representative task regeneration, before any method evaluation.
    t=b.build_task('MDB_S1_BASE',1);ck('S1_units',len(t.units)>18000);ck('S1_targets_25',len(t.targets)==25);ck('S1_hard_truth_175',len(t.hard_truth)==175);ck('S1_features_26',len(t.allowed_predictors)==26)
    t180=b.build_task('MDB_S10_180M',1);ck('S10_180_targets_25',len(t180.targets)==25);ck('S10_180_features_finite',len(t180.allowed_predictors)>=20)
    ck('truth_domain_S1',((t.units.Y_observed_at_A>=-1)&(t.units.Y_observed_at_A<=1)).all())
    # Frozen methods, thresholds, route and counts.
    ck('methods_M2_M6',c['primary_methods']==['M2','M3','M4','M5','M6'])
    ck('threshold_ess',c['support_thresholds']['minimum_ess']==3.0)
    ck('threshold_maxw',c['support_thresholds']['maximum_normalized_weight']==.5)
    ck('threshold_graph',c['support_thresholds']['minimum_graph_safe_count']==20)
    ck('domain_minus1_1',c['candidate_domain']==[-1.0,1.0])
    ck('D2_score_abs_tol',c['numerical_tolerances']['score_tie_abs_tolerance']==1e-12)
    ck('D2_score_rel_tol',c['numerical_tolerances']['score_tie_rel_tolerance']==1e-12)
    ck('D2_non_gaussian_min',c['numerical_tolerances']['non_gaussian_min_abs_residual']==1e-14)
    ck('D2_grid_4096',c['full_inversion']['grid_points']==4096 and c['full_inversion']['boundary_bisection_iterations']==40)
    ck('route_m64',c['spatial_route']['m']==64)
    ck('route_m32_sensitivity',c['spatial_route']['sensitivity_m']==32)
    ck('five_methods_only','M1' not in c['primary_methods'] and 'H1' not in c['primary_methods'] and 'H4' not in c['primary_methods'])
    # Stage3F gate mapping exactly: M3 ignores ESS/max but needs graph; M4/M5/M6 use ESS/max+graph; M6 also N3 non-vacuity.
    good={'support_status':'positive','support_ess':1.0,'support_max_weight':.9}
    ck('gate_M3_ignores_weight',_refusal('M3',True,good,25,np.nan,True,c)=='')
    ck('gate_M3_graph',_refusal('M3',True,good,19,np.nan,True,c)=='R06_GRAPH_SAFE_COUNT_BELOW_FROZEN_THRESHOLD')
    ck('gate_M4_ess',_refusal('M4',True,good,25,np.nan,True,c)=='R04_ESS_BELOW_FROZEN_THRESHOLD')
    good2={'support_status':'positive','support_ess':5.0,'support_max_weight':.4}
    ck('gate_M6_n3',_refusal('M6',True,good2,25,np.nan,True,c)=='R08_N3_DIAGNOSTIC_VACUOUS')
    ck('gate_M6_pass',_refusal('M6',True,good2,25,.85,True,c)=='')
    ck('oracle_M6_n3_truth_gate',not _oracle_support_truth('M6',True,good2,good2,25,np.nan,c) and _oracle_support_truth('M6',True,good2,good2,25,.85,c))
    ck('gate_endpoint',_refusal('M2',False,good2,0,np.nan,True,c)=='R02_ENDPOINT_NOT_AUDITED')
    ck('gate_no_support',_refusal('M2',True,{'support_status':'no_positive','support_ess':0.,'support_max_weight':1.},0,np.nan,True,c)=='R03_NO_POSITIVE_INTERVENTION_SUPPORT')
    ck('gate_compute_failure',_refusal('M2',True,good2,0,np.nan,False,c)=='R10_COMPUTATIONAL_FAILURE')
    # Mixed-measure q endpoint/interior behavior.
    aa=np.array([0.,.1,.5,.9,1.]);q0=q_at(aa,0.,None,True);q1=q_at(aa,1.,None,True);qi=q_at(aa,.5,.1,False)
    ck('q_endpoint0_exact',q0[0]>0 and np.all(q0[1:]==0));ck('q_endpoint1_exact',q1[-1]>0 and np.all(q1[:-1]==0));ck('q_interior_positive',qi[2]>0 and qi[0]==0 and qi[-1]==0)
    # Orbit algebra: normalized, M4 one normalization, scalar/vector grid equality, KL nonnegative.
    e=np.array([-.4,-.2,-.1,.15,.3,.45]);P=np.eye(6);m3=eval_m3(e,P);d=np.log(np.array([.8,.9,1.,1.1,1.2,1.3]));m4=eval_m4(m3,e,d)
    ck('M3_pvalue_01',0<=m3['pvalue']<=1);ck('M3_source_normalized',abs(m3['source_marginal'].sum()-1)<1e-12);ck('M4_one_normalization',m4['normalization_count']==1);ck('M4_prob_normalized',abs(m4['source_probs'].sum()-1)<1e-12)
    grid=np.linspace(-.25,.25,9);p3,p4=eval_m3_m4_grid(e[:5],grid,0.,P,1.,d)
    sc3=np.array([eval_m3(np.r_[e[:5],y],P)['pvalue'] for y in grid]);sc4=np.array([eval_m4(eval_m3(np.r_[e[:5],y],P),np.r_[e[:5],y],d)['pvalue'] for y in grid])
    ck('M3_vector_scalar',np.max(np.abs(p3-sc3))<1e-12);ck('M4_vector_scalar',np.max(np.abs(p4-sc4))<1e-12)
    # M6 vector/scalar with mandatory inverse-Jacobian matrix explicitly present (identity scale => zero log Jacobian).
    R=np.tile(e,(6,1))+np.arange(6)[:,None]*.01;T=np.zeros((6,6));J=np.zeros((6,6));m6=eval_m6(R,T,J,P,1.0);ck('M6_pvalue_01',0<=m6['pvalue']<=1);ck('M6_probs_normalized',abs(m6['orbit_probs'].sum()-1)<1e-12)
    R0=R.copy();R0[:,5]-=R0[5,5];pg=eval_m6_grid(R0,T,J,grid,P,1.0);sg=[]
    for y in grid:
        rr=R0.copy();rr[:,5]+=y;sg.append(eval_m6(rr,T,J,P,1.0,keep_probs=False)['pvalue'])
    ck('M6_vector_scalar',np.max(np.abs(pg-np.array(sg)))<1e-12)
    ck('orbit_kl_identity',orbit_kl(m3['orbit_probs'],m3['orbit_probs'])<1e-14)
    # Candidate inversion remains inside the physical domain and preserves disconnected components.
    inv=invert_candidate(lambda y: .2 if (-.7<y<-.3 or .2<y<.6) else .05,-1,1,401,.1,20);ck('inversion_two_components',inv['component_count']==2);ck('inversion_hull_inflates',inv['hull_width']>inv['raw_set_width']);ck('inversion_domain_bounds',inv['hull_lower']>=-1 and inv['hull_upper']<=1)
    ck('infinite_WIS_preserved',math.isinf(interval_score(-np.inf,np.inf,0.,.1)))
    # Checkpoint codec preserves NaN and infinities exactly as special values and never relies on invalid JSON NaN literals.
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x.json.gz';atomic_json_gz(p,{'a':np.nan,'b':np.inf,'c':-np.inf,'d':1.25});z=read_json_gz(p);ck('checkpoint_nonfinite_codec',np.isnan(z['a']) and np.isposinf(z['b']) and np.isneginf(z['c']) and z['d']==1.25)
    # Count arithmetic fixed before production.
    ec=c['expected_counts'];ck('count_query_arithmetic',ec['query_rows']==187500);ck('count_exact_arithmetic',ec['exact_audit_rows']==16200);ck('count_inversion_arithmetic',ec['full_inversion_rows']==150);ck('count_endpoint_arithmetic',ec['endpoint_audit_rows']==4000);ck('count_dose_arithmetic',ec['dose_response_prediction_rows']==227500);ck('count_fit_arithmetic',ec['nuisance_fit_rows']==1300);ck('count_lomo_arithmetic',ec['lomo_rows']==1875);ck('count_spatial_arithmetic',ec['spatial_diagnostic_rows']==7800)
    # Claim discipline.
    ck('estimated_not_theorem_certified',c['claim_boundaries']['estimated_nuisance_finite_sample_theorem_certified'] is False)
    ck('M4_not_theorem',c['claim_boundaries']['M4_theorem_backed'] is False)
    ck('ST2_no_temporal_claim',c['claim_boundaries']['ST2_temporal_method_claim'] is False)
    ck('ST3_no_positive_causal_claim',c['claim_boundaries']['ST3_positive_causal_identification_claim'] is False)
    ck('no_full_real_line',c['claim_boundaries']['full_real_line_prediction_claim'] is False)
    ck('refusal_not_noncoverage',c['claim_boundaries']['refusal_counted_as_noncoverage'] is False)
    # Static source leakage / network / tuning scan.
    src='\n'.join(p.read_text(encoding='utf-8',errors='ignore') for p in (ROOT/'src/geodose_stage5b').glob('*.py'))
    ck('no_network','requests.' not in src and 'urllib.' not in src and 'http://' not in src and 'https://' not in src)
    ck('no_threshold_grid_tuning','candidate_threshold' not in src and 'threshold_grid' not in src)
    ck('no_mapped_rehab_evaluator','mapped_rehabilitation_fraction' not in (ROOT/'src/geodose_stage5b/evaluation.py').read_text())
    ck('no_2025_evaluator','pv_median_2025' not in (ROOT/'src/geodose_stage5b/evaluation.py').read_text())
    # v1.0.1 regression: primary S8 calibration subsampling stays frozen at 8%,
    # while the separate structural exact-size6 audit uses the full frozen calibration-role frame.
    from geodose_stage5b.evaluation import _sorted_calibration, evaluate_exact_audit
    from geodose_stage5b.spatial import graph_info, local_context
    s8=b.build_task('MDB_S8_SMALL_CAL',1)
    primary_cal=_sorted_calibration(s8); full_cal=s8.units[s8.units.benchmark_role.astype(str)=='calibration']
    ck('s8_small_cal_primary_subsample_preserved',len(primary_cal)<len(full_cal) and abs(float(s8.case['calibration_subsample_fraction'])-0.08)<1e-15)
    s8idx={b:i for i,b in enumerate(s8.units.block_id.astype(str))}; full_idx=np.array([s8idx[b] for b in full_cal.block_id.astype(str)],int); g8=graph_info(s8.units,s8.working_edges); em=s8.exact_map.set_index('target_block_id')
    all_slots=True
    for er in s8.exact_registry.itertuples(index=False):
        if not bool(er.exact_audit_available): continue
        tid=str(er.exact_audit_target_block_id); mm=em.loc[tid]; bcal=[s8idx[str(mm[f'calibration_block_{k}'])] for k in range(1,6)]; ti=s8idx[tid]; ctx=set(map(int,local_context(ti,full_idx,s8.units,int(c['spatial_route']['m'])))); all_slots &= set(bcal).issubset(ctx)
    ck('s8_exact_slots_in_full_m64_context',all_slots)
    rows=evaluate_exact_audit(s8,c,g8,np.array([s8idx[b] for b in primary_cal.block_id.astype(str)],int))
    ck('s8_exact_audit_regression_rep1',len(rows)==30 and all(np.isfinite(float(r['pvalue'])) for r in rows))
    # v1.0.2 numerical closure: only machine-roundoff excursions at [0,1] are canonicalized.
    ck('canonical_probability_upper_roundoff',_canonical_probability(1.0+2.0e-16)==1.0)
    ck('canonical_probability_lower_roundoff',_canonical_probability(-2.0e-16)==0.0)
    refused_prob=False
    try:_canonical_probability(1.0+1.0e-6)
    except Exception:refused_prob=True
    ck('canonical_probability_material_violation_refused',refused_prob)
    exdf=pd.DataFrame([
      {'case_id':'C','scenario_id':'S','replication':1,'MineID':'M','target_block_id':'T','method':'M3','reference':'full_graph_exact','pvalue':1.0+2.0e-16,'covered':True,'pvalue_absolute_difference':0.0},
      {'case_id':'C','scenario_id':'S','replication':1,'MineID':'M','target_block_id':'T','method':'M3','reference':'sparse_m64','pvalue':0.8,'covered':True,'pvalue_absolute_difference':0.0}])
    ex2,aud=_canonicalize_exact_audit_frame(exdf,.1);ck('exact_roundoff_canonicalization',float(ex2.pvalue.max())==1.0 and abs(float(ex2.pvalue_absolute_difference.iloc[0])-.2)<1e-15 and aud['canonicalized_row_count']==1 and not aud['coverage_changed'])
    bad=exdf.copy();bad.loc[0,'pvalue']=1.000001;refused_bad=False
    try:_canonicalize_exact_audit_frame(bad,.1)
    except Exception:refused_bad=True
    ck('exact_material_violation_refused',refused_bad)
    ck('software_patch_declared',c.get('software_patch')=='1.0.2-exact-pvalue-roundoff-canonicalization')
    # Resume compatibility: legacy v1.0.0 checkpoints are reusable for unaffected
    # cases, but an affected S8_SMALL_CAL checkpoint without the patch ID is refused.
    from geodose_stage5b.runner import _validate_checkpoint, EVALUATOR_PATCH_ID
    payload0={'query_rows':[],'exact_rows':[],'inversion_rows':[],'endpoint_rows':[],'dose_rows':[],'fit_rows':[],'spatial_rows':[],'runtime_row':{},'lomo_rows':[]}
    basecp={'checkpoint_schema':'Stage5BTaskCheckpoint-v1','stage5b_version':c['version'],'stage5a_release_sha256':a['stage5a_release_zip_sha256'],'replication':1,'payload':payload0}
    okcp=dict(basecp,case_id='MDB_S8_SEVERE_TAIL'); _validate_checkpoint(okcp,c['version'],a['stage5a_release_zip_sha256'],'MDB_S8_SEVERE_TAIL',1); ck('legacy_unaffected_checkpoint_reusable',True)
    badcp=dict(basecp,case_id='MDB_S8_SMALL_CAL'); refused=False
    try: _validate_checkpoint(badcp,c['version'],a['stage5a_release_zip_sha256'],'MDB_S8_SMALL_CAL',1)
    except Exception: refused=True
    ck('legacy_affected_checkpoint_refused',refused)
    goodcp=dict(badcp,evaluator_patch_id=EVALUATOR_PATCH_ID); _validate_checkpoint(goodcp,c['version'],a['stage5a_release_zip_sha256'],'MDB_S8_SMALL_CAL',1); ck('patched_affected_checkpoint_reusable',True)
    print(f'STAGE5B UNIT/AUTHORITY TESTS PASSED ({len(PASS)}/{len(PASS)})')
if __name__=='__main__':main()
