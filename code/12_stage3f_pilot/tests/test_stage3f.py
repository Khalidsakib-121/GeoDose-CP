from __future__ import annotations
import gzip,hashlib,inspect,json,math,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from geodose_stage3f.io import deterministic_gzip_csv,sha256_file
from geodose_stage3f.provenance import source_inventory
from geodose_stage3f.pilot_math import (
    adjacency_from_edges,choose_connected_block,block_is_connected,eligible_target_blocks,component_calibration_count,
    summarize_logweights,interval_components_from_grid,component_summary,choose_thresholds,operational_gate,
    select_uniform_eligible_target,derived_tie_uniform
)
from geodose_stage3f.version import VERSION,STAGE

passed=0
def test(name,fn):
 global passed
 try: fn();passed+=1;print('PASS',name)
 except Exception as e: print('FAIL',name,':',e);raise

def req(x,msg='assertion failed'):
 if not x: raise AssertionError(msg)

def eq(a,b,tol=0):
 if tol: req(abs(float(a)-float(b))<=tol,f'{a}!={b}')
 else: req(a==b,f'{a}!={b}')

def contract(): return yaml.safe_load((ROOT/'configs/pilot_contract.yaml').read_text())

def resolved():
 p=ROOT/'configs/resolved_paths_windows.json'; req(p.is_file(),'preflight must create resolved_paths_windows.json before unit tests'); return {k:Path(v) for k,v in json.loads(p.read_text()).items()}

def simple_units_edges():
 # calibration nodes 0..4, test targets 5,6; only node5 directly connected to calibration graph.
 u=pd.DataFrame({'node_index':range(7),'role':['calibration']*5+['test_target']*2})
 e=pd.DataFrame({'source_node':[0,1,2,3,4,3,5],'target_node':[1,2,3,4,5,5,6]})
 return u,e

def main():
 c=contract()
 test('version_stage',lambda:(eq(c['version'],VERSION),eq(c['stage'],STAGE)))
 test('twenty_replications',lambda:eq(int(c['replications']),20))
 test('twenty_case_variants',lambda:req(len(c['pilot_case_ids'])==20 and len(set(c['pilot_case_ids']))==20))
 test('xgb_scope_frozen',lambda:req(set(c['xgb_scenarios'])=={'S1','S4','S5','S8'}))
 test('nuisance_reps_frozen',lambda:req(c['nuisance_estimation_diagnostic_reps']==[1,10,20]))
 test('alpha_frozen',lambda:eq(float(c['alpha']),0.1))
 test('m64_primary',lambda:eq(int(c['primary_scalable_neighborhood_size']),64))
 test('no_nsw_tuning',lambda:req(c['claims']['nsw_tuning_allowed'] is False))
 test('threshold_selection_uses_controlled_pilot_coverage_not_width',lambda:req(c['threshold_selection']['uses_outcomes'] is True and c['threshold_selection']['uses_controlled_pilot_outcomes_only'] is True and c['threshold_selection']['width_used'] is False and c['claims']['threshold_selection_uses_width'] is False))
 test('stage3c_threshold_candidate_grids_exact',lambda:req(set(c['threshold_candidate_grid'])=={'minimum_ess','maximum_normalized_weight','minimum_graph_safe_count'} and c['threshold_candidate_grid']['minimum_ess']==[3.0,5.0,10.0,15.0,20.0] and c['threshold_candidate_grid']['maximum_normalized_weight']==[0.1,0.15,0.2,0.3,0.5] and c['threshold_candidate_grid']['minimum_graph_safe_count']==[5,10,15,20]))
 test('no_unregistered_certificate_threshold_grid',lambda:req(c['threshold_grid_provenance']['additional_candidate_grids_added_by_stage3f'] is False and 'minimum_certified_coverage_lower_bound' not in c['threshold_candidate_grid']))
 test('R11_no_synthetic_numeric_transfer',lambda:req(c['r11_eo_quality_rule']['synthetic_ue_numeric_threshold_frozen'] is False))
 test('target_population_preoutcome',lambda:req(not c['target_population_selection']['uses_treatment_A'] and not c['target_population_selection']['uses_outcome_Y'] and not c['target_population_selection']['uses_support_diagnostics']))
 test('candidate_domain',lambda:req(c['candidate_domain']==[-8.0,8.0] and int(c['width_audit_grid_points'])==161))
 u,e=simple_units_edges()
 test('adjacency_symmetric',lambda:req(1 in adjacency_from_edges(7,e)[0] and 0 in adjacency_from_edges(7,e)[1]))
 test('connected_block_size6',lambda:req(len(choose_connected_block(u,e,5,6))==6 and choose_connected_block(u,e,5,6)[-1]==5 and block_is_connected(choose_connected_block(u,e,5,6),e)))
 test('eligible_target_excludes_noncalibration_bridge',lambda:req(set(eligible_target_blocks(u,e,6))=={5}))
 test('component_calibration_count',lambda:eq(component_calibration_count(u,e,5),5))
 test('equal_logweights_ess',lambda:(eq(summarize_logweights(np.zeros(5))['ess'],5,1e-12),eq(summarize_logweights(np.zeros(5))['max_normalized_weight'],.2,1e-12)))
 test('single_positive_weight',lambda:(eq(summarize_logweights(np.array([0,-np.inf]))['ess'],1,1e-12),eq(summarize_logweights(np.array([0,-np.inf]))['max_normalized_weight'],1,1e-12)))
 def gridcomp():
  g=np.arange(5.);a=np.array([1,1,0,1,0],bool);co=interval_components_from_grid(g,a,0,4);req(len(co)==2);s=component_summary(co);req(s['component_count']==2 and s['hull_inflation']>=0)
 test('grid_components',gridcomp)
 def thselect():
  rows=[];res=[]
  for cid in c['threshold_selection']['safe_hard_case_ids']:
   for r in range(20):
    rows.append({'case_id':cid,'replication':r+1,'ess':25.,'max_normalized_weight':.05,'graph_safe_count':30,'certified_coverage_lower_bound':.899,'target_supported_by_generator':True,'target_population_eligible':True,'endpoint_requested':False,'endpoint_audited':True})
    res.append({'case_id':cid,'replication':r+1,'track':'ORACLE_STRUCTURAL_OT_TG','method':'M6','raw_covered':(r<18),'raw_method_returned':True,'computational_failure':False,'target_supported':True})
  for cid in c['threshold_selection']['expected_low_information_case_ids']:
   for r in range(20):
    rows.append({'case_id':cid,'replication':r+1,'ess':1.,'max_normalized_weight':.99,'graph_safe_count':4,'certified_coverage_lower_bound':.899,'target_supported_by_generator':True,'target_population_eligible':True,'endpoint_requested':False,'endpoint_audited':True})
    res.append({'case_id':cid,'replication':r+1,'track':'ORACLE_STRUCTURAL_OT_TG','method':'M6','raw_covered':True,'raw_method_returned':True,'computational_failure':False,'target_supported':True})
  ch,a=choose_thresholds(pd.DataFrame(rows),pd.DataFrame(res),c);req(ch['selection_used_controlled_pilot_outcomes'] is True and ch['selection_used_width'] is False and ch['expected_low_information_return_rate']==0 and ch['selection_rule_status']=='STAGE3F_FROZEN_LEXICOGRAPHIC_NO_POSTHOC_RETENTION_CUTOFF')
 test('threshold_selector_frozen_coverage_refusal_rule',thselect)
 test('no_posthoc_hard_retention_cutoffs',lambda:req('minimum_safe_retention' not in c['threshold_selection'] and 'minimum_s3_retention' not in c['threshold_selection'] and 'minimum_strong_s4_retention' not in c['threshold_selection']))
 def gates():
  th={'minimum_ess':5.,'maximum_normalized_weight':.3,'minimum_graph_safe_count':10}
  base={'target_population_eligible':True,'target_supported':True,'endpoint_requested':False,'endpoint_audited':True,'ess':4.,'max_normalized_weight':.2,'graph_safe_count':20,'operational_coverage_lower_bound':.9}
  s=pd.Series(base);req(operational_gate('M6',s,th)[1]=='R03_LOW_EFFECTIVE_CALIBRATION_SIZE')
  s.ess=6;s.max_normalized_weight=.4;req(operational_gate('M6',s,th)[1]=='R04_WEIGHT_CONCENTRATION')
  s.max_normalized_weight=.2;s.graph_safe_count=5;req(operational_gate('M6',s,th)[1]=='R05_GRAPH_COMPONENT_TOO_SMALL')
  s.graph_safe_count=20;s.operational_coverage_lower_bound=0.0;req(operational_gate('M6',s,th)[1]=='R08_CERTIFIED_DEFICIT_VACUOUS')
  s.operational_coverage_lower_bound=.9;s.target_supported=False;req(operational_gate('M6',s,th)[1]=='R01_UNSUPPORTED_DOSE')
 test('operational_gate_order',gates)
 test('M1_still_obeys_structural_support_gate',lambda:req(operational_gate('M1',pd.Series({'target_population_eligible':True,'target_supported':False,'endpoint_requested':False,'endpoint_audited':True}),{'minimum_ess':5,'maximum_normalized_weight':.3,'minimum_graph_safe_count':10})[1]=='R01_UNSUPPORTED_DOSE'))
 def dgzip():
  with tempfile.TemporaryDirectory() as td:
   d=pd.DataFrame({'a':[1,2],'b':[.1,.2]});p1=Path(td)/'a.gz';p2=Path(td)/'b.gz';deterministic_gzip_csv(d,p1);deterministic_gzip_csv(d,p2);eq(sha256_file(p1),sha256_file(p2))
 test('deterministic_gzip',dgzip)
 def invskip():
  inv=source_inventory(ROOT);req('configs/resolved_paths_windows.json' not in {r['path'] for r in inv['files']})
 test('source_inventory_excludes_machine_resolved_paths',invskip)
 # Provenance config/locks.
 exp=json.loads((ROOT/'configs/expected_upstream_hashes.json').read_text());locks=json.loads((ROOT/'configs/source_locks.json').read_text())
 test('four_accepted_upstream_hashes',lambda:req(set(exp)=={'stage3a_output_zip','stage3b_output_zip','stage3d_d4_output_zip','stage3e_output_zip'} and all(len(v)==64 for v in exp.values())))
 test('source_lock_sets',lambda:req(set(locks)=={'stage3b_source_root','stage3c_source_root','stage3d_d2_source_root','stage3d_d4_source_root','stage3e_source_root'}))
 test('source_lock_counts',lambda:req(len(locks['stage3c_source_root']['files'])==18 and len(locks['stage3d_d2_source_root']['files'])==41 and len(locks['stage3d_d4_source_root']['files'])==39 and len(locks['stage3e_source_root']['files'])==30))
 # Exact source/API recheck after preflight.
 rp=resolved()
 def sourcehashes():
  for key,lock in locks.items():
   for rel,h in lock['files'].items(): req(sha256_file(rp[key]/rel)==h,f'{key}/{rel}')
 test('all_locked_source_bytes',sourcehashes)
 def stage3cplan():
  plan=yaml.safe_load((rp['stage3c_source_root']/'configs/stage3c_pilot_threshold_plan.yaml').read_text());req(plan['candidate_grids_frozen_before_pilot']['minimum_ess']==[3,5,10,15,20]);req(plan['candidate_grids_frozen_before_pilot']['maximum_normalized_weight']==[.10,.15,.20,.30,.50]);req(plan['candidate_grids_frozen_before_pilot']['minimum_graph_safe_count']==[5,10,15,20]);req('coverage/refusal criterion' in plan['selection_rule'])
 test('accepted_Stage3C_threshold_plan_lock',stage3cplan)
 def d2api():
  import importlib,sys as _s
  src=rp['stage3d_d2_source_root']/'src';_s.path.insert(0,str(src));m=importlib.import_module('geodose_stage3d.candidate_inversion');ex=importlib.import_module('geodose_stage3d.exact_law')
  eq(str(inspect.signature(ex.prepare_orbit)),"(fixture: 'ExactFixture', *, minimum_precision_eigenvalue_required: 'float') -> 'PreparedOrbit'")
  eq(str(inspect.signature(m.CandidateEvaluator)),"(prepared: 'PreparedOrbit', *, alpha: 'float', tie_uniform: 'float', score_abs_tolerance: 'float', score_rel_tolerance: 'float', non_gaussian_min_abs_residual: 'float') -> 'None'")
  eq(str(inspect.signature(m.CandidateEvaluator.evaluate)),"(self, candidate_y: 'float', *, keep_state: 'bool' = False) -> 'CandidateResult'")
 test('D2_exact_API_lock',d2api)
 def eapi():
  import importlib,sys as _s
  src=rp['stage3e_source_root']/'src';_s.path.insert(0,str(src));o=importlib.import_module('geodose_stage3e.ordering');n=importlib.import_module('geodose_stage3e.gaussian_n2')
  eq(str(inspect.signature(o.deterministic_maximin_order)),"(coords: 'np.ndarray', node_ids: 'np.ndarray') -> 'np.ndarray'")
  req('minimum_precision_eigenvalue' in inspect.signature(n.build_vecchia_family).parameters)
 test('Stage3E_exact_API_lock',eapi)
 # Frozen Stage3A target population explicitly names Stage3F and uniform theorem-eligible slots.
 def stage3a_targetpop():
  import zipfile
  with zipfile.ZipFile(rp['stage3a_output_zip']) as z:
   d=json.loads(z.read('outputs_stage3a/target_population_contract.json').decode());req('uniform over theorem-eligible final-test slots' in d['controlled_target_population']);req('Stage3F' in d['target_population_freeze_stage'])
 test('Stage3A_target_population_alignment',stage3a_targetpop)
 # One real paired target-selection audit using exact frozen generator source. This is light and catches future selector drift.
 def paired_selection():
  from geodose_stage3f.bridges import Stage3BBridge
  b=Stage3BBridge(rp['stage3b_source_root'],rp['stage3a_output_zip'],rp['stage3b_output_zip'])
  vals=[]
  for cid in ['S4_RHO020','S4_RHO060']:
   case=b.cases[cid];g=b.generate(cid,1);ed=g['true_edges'][['source_node','target_node']];el=eligible_target_blocks(g['units'],ed);node,block,meta=select_uniform_eligible_target(b.core,b.stage3a,case,1,el);vals.append((meta['target_selection_uniform'],meta['target_selection_rank'],node,len(el)))
  eq(vals[0][0],vals[1][0],1e-15);eq(vals[0][1],vals[1][1]);eq(vals[0][2],vals[1][2]);eq(vals[0][3],25)
 test('paired_factor_target_selection',paired_selection)
 def all_rep1_blocks_true_graph_connected():
  from geodose_stage3f.bridges import Stage3BBridge
  b=Stage3BBridge(rp['stage3b_source_root'],rp['stage3a_output_zip'],rp['stage3b_output_zip'])
  for cid in c['pilot_case_ids']:
   case=b.cases[cid];g=b.generate(cid,1);ed=g['true_edges'][['source_node','target_node']];el=eligible_target_blocks(g['units'],ed)
   req(all(block_is_connected(bl,ed) for bl in el.values()),cid)
 test('all_pilot_rep1_structural_blocks_induced_connected',all_rep1_blocks_true_graph_connected)
 def fitted_s6_disconnection_is_possible_and_scientific_refusal():
  from geodose_stage3f.bridges import Stage3BBridge,D2Bridge
  b=Stage3BBridge(rp['stage3b_source_root'],rp['stage3a_output_zip'],rp['stage3b_output_zip']);d=D2Bridge(rp['stage3d_d2_source_root'],0.1)
  found=False
  for cid in ['S6_ROOK','S6_OMIT50']:
   case=b.cases[cid]
   for rep in range(1,21):
    g=b.generate(cid,rep);te=g['true_edges'][['source_node','target_node']];fe=g['fitted_edges'][['source_node','target_node']];el=eligible_target_blocks(g['units'],te);node,bl,_=select_uniform_eligible_target(b.core,b.stage3a,case,rep,el)
    req(block_is_connected(bl,te),f'true:{cid}:{rep}')
    if not d.gp.block_is_connected(np.asarray(bl,int),fe): found=True
  req(found,'registered S6 fitted-graph diagnostic should exercise at least one disconnected frozen structural block')
 test('S6_fitted_graph_can_disconnect_frozen_block_without_retargeting',fitted_s6_disconnection_is_possible_and_scientific_refusal)
 def tie_seed():
  from geodose_stage3f.bridges import Stage3BBridge
  b=Stage3BBridge(rp['stage3b_source_root'],rp['stage3a_output_zip'],rp['stage3b_output_zip']);case=b.cases['S1_BASE'];a=derived_tie_uniform(b.core,b.stage3a,case,1,320,'A');bb=derived_tie_uniform(b.core,b.stage3a,case,1,320,'B');req(a[0]!=bb[0] and a[2]!=bb[2])
 test('track_specific_tie_seed',tie_seed)
 # Static scientific guardrails.
 runner=(ROOT/'src/geodose_stage3f/runner.py').read_text();pm=(ROOT/'src/geodose_stage3f/pilot_math.py').read_text();pre=(ROOT/'preflight_windows.py').read_text()
 test('runner_no_production_seed_phase',lambda:req('production_provisional' not in runner))
 test('threshold_function_uses_coverage_but_not_width',lambda:req('raw_covered' in inspect.getsource(choose_thresholds) and "['width']" not in inspect.getsource(choose_thresholds) and '.width' not in inspect.getsource(choose_thresholds)))
 test('runner_separates_oracle_empirical_tracks',lambda:req('ORACLE_STRUCTURAL_OT_TG' in runner and 'RF_SHARED_PREDICTOR_EMPIRICAL' in runner and 'XGB_SHARED_PREDICTOR_EMPIRICAL' in runner))
 test('runner_s9_latent_separation',lambda:req("case.scenario_id=='S9'" in runner and 'covered_latent' in runner))
 test('runner_uses_stage3e_sparse_precision',lambda:req('e3.sparse_precision' in runner and 'patch_sparse_precision' in runner))
 test('runner_uses_accepted_D2_candidate_evaluator',lambda:req('d2.evaluator' in runner and '.evaluate(' in runner))
 test('runner_m4_one_normalization_vendor',lambda:req('_m3m4_candidate' in runner and 'm4_source_logd' in runner))
 test('runner_fitted_graph_disconnect_fail_closed',lambda:req('block_connected' in runner and "R05_GRAPH_COMPONENT_TOO_SMALL" in runner and 'reason to retarget' in runner))
 test('width_diagnostic_not_claimed',lambda:req('diagnostic_grid_not_coverage_preserving' in runner and c['claims']['production_performance_evidence'] is False))
 test('preflight_hash_before_environment',lambda:req('verify' in pre.lower() and 'expected_upstream_hashes' in pre))
 test('runtime_pins',lambda:req((ROOT/'requirements_py310.txt').read_text().splitlines()[:7]==['numpy==2.2.6','pandas==2.3.3','scipy==1.15.3','scikit-learn==1.7.2','xgboost==3.0.5','joblib==1.5.2','PyYAML==6.0.3']))
 print(f'STAGE3F UNIT TESTS PASSED ({passed}/{passed})')

if __name__=='__main__': main()
