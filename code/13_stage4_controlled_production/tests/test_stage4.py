from __future__ import annotations
import ast,hashlib,inspect,json,math,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from geodose_stage4.version import STAGE,VERSION
from geodose_stage4.io import sha256_file,deterministic_gzip_csv
from geodose_stage4.provenance import source_inventory
from geodose_stage4.production_registry import build_main_registry,build_f06_registry,build_extension_registry,registry_summary
from geodose_stage4.f06 import install_stage3b_production_adapter,install_stage3c_production_nuisance_adapter,f06_case,f06_structural_qa
from geodose_stage3f.bridges import Stage3BBridge,Stage3CBridge,D2Bridge,Stage3EBridge
from geodose_stage3f.pilot_math import eligible_target_blocks,select_uniform_eligible_target,block_is_connected,operational_gate

passed=0
def test(name,fn):
 global passed
 try: fn();passed+=1;print('PASS',name)
 except Exception as e: print('FAIL',name,':',e);raise

def req(x,msg='assertion failed'):
 if not bool(x):raise AssertionError(msg)
def eq(a,b,tol=0):
 if tol:req(abs(float(a)-float(b))<=tol,f'{a}!={b}')
 else:req(a==b,f'{a}!={b}')
def contract():return yaml.safe_load((ROOT/'configs/production_contract.yaml').read_text())
def resolved():
 p=ROOT/'configs/resolved_paths_windows.json';req(p.is_file(),'preflight must create resolved_paths_windows.json before tests');return {k:Path(v) for k,v in json.loads(p.read_text()).items()}

def main():
 c=contract();rp=resolved()
 test('version_stage',lambda:(eq(c['version'],VERSION),eq(c['stage'],STAGE)))
 test('alpha_0p1',lambda:eq(float(c['alpha']),.1))
 test('main_rows_3000',lambda:eq(int(c['production_main_rows_expected']),3000))
 test('m64_primary',lambda:eq(int(c['primary_scalable_neighborhood_size']),64))
 test('no_threshold_retuning',lambda:req(c['threshold_retuning_allowed'] is False))
 test('no_nsw_tuning',lambda:req(c['nsw_tuning_allowed'] is False))
 test('stage3f_thresholds_frozen',lambda:req(c['frozen_stage3f_thresholds']['minimum_ess']==3.0 and c['frozen_stage3f_thresholds']['maximum_normalized_weight']==.5 and c['frozen_stage3f_thresholds']['minimum_graph_safe_count']==20))
 test('n3_nonvacuity_not_retuned',lambda:req(c['frozen_stage3f_thresholds']['n3_nonvacuity_rule']=='finite_coverage_lower_bound_strictly_greater_than_0'))
 test('f06_levels_exact',lambda:req({k:(v['rows'],v['cols'],v['n']) for k,v in c['f06_freeze']['levels'].items()}=={'small':(17,17,289),'primary':(25,25,625),'large':(35,35,1225)}))
 test('f06_fixed_resolution',lambda:req(c['f06_freeze']['resolution_m']==90 and c['f06_freeze']['adjacency']=='queen' and c['f06_freeze']['boundary']=='nonperiodic'))
 test('f06_not_MAUP',lambda:req('not_resolution_or_MAUP' in c['f06_freeze']['scientific_claim']))
 test('f06_random_stream_claim_honest',lambda:req('no_strict_nested_CRN_claim' in c['f06_freeze']['random_stream_pairing']))
 test('f06_edges_frozen',lambda:req([(c['f06_freeze']['levels'][k]['full_geographic_queen_edges'],c['f06_freeze']['levels'][k]['role_cut_dgp_edges']) for k in ['small','primary','large']]==[(1056,958),(2352,2206),(4692,4486)]))
 test('f06_primary_stage3b_roles',lambda:req(c['f06_freeze']['levels']['primary']['role_columns']=={'nuisance_training':10,'support_audit':5,'calibration':5,'test_target':5}))
 test('f06_s10_not_reused',lambda:req(c['f06_freeze']['levels']['small']['n']!=169 and c['f06_freeze']['resolution_m']==90))
 test('main_allocation_20_cases',lambda:eq(len(c['main_production_allocation']),20))
 test('main_scenario_totals',lambda:req({s:sum(a['seed_rep_end']-a['seed_rep_start']+1 for a in c['main_production_allocation'] if a['scenario_id']==s) for s in [f'S{i}' for i in range(1,11)]}=={'S1':300,'S2':300,'S3':300,'S4':500,'S5':300,'S6':300,'S7':300,'S8':300,'S9':200,'S10':200}))
 test('s4_rho_windows',lambda:req([(a['case_id'],a['seed_rep_start'],a['seed_rep_end']) for a in c['main_production_allocation'] if a['scenario_id']=='S4']==[('S4_RHO000',1,100),('S4_RHO020',101,200),('S4_RHO040',201,300),('S4_RHO060',301,400),('S4_RHO080',401,500)]))
 test('extensions_count_expected',lambda:eq(sum(e['seed_rep_end']-e['seed_rep_start']+1 for e in c['extensions']),590))
 test('st2_structural_only',lambda:req(next(e for e in c['extensions'] if e['extension_id']=='EXT_ST2_TEMPORAL')['method_evaluation'] is False))
 test('st3_identification_negative_control',lambda:req('causal_identification_negative_control' in next(e for e in c['extensions'] if e['extension_id']=='EXT_ST3_HIDDEN_C2')['theorem_status']))
 test('endpoint_negative_controls',lambda:req({e['extension_id']:e.get('expected_structural_refusal') for e in c['extensions'] if 'UNAUDITED' in e['extension_id']}=={'EXT_ST1_UNAUDITED0':'R02_ENDPOINT_NOT_AUDITED','EXT_ST1_UNAUDITED1':'R02_ENDPOINT_NOT_AUDITED'}))
 test('xgb_scope',lambda:req(set(c['tracks']['xgb_scenarios'])=={'S1','S4','S5','S8'}))
 test('oracle_primary_theorem_track',lambda:req(c['tracks']['theorem_certified_primary']=='ORACLE_STRUCTURAL_OT_TG' and c['tracks']['empirical_tracks_theorem_certified'] is False))
 test('heuristics_not_theorem',lambda:req(c['heuristics']['theorem_eligible'] is False))
 test('ablations_0_to_9',lambda:req(set(c['ablation_mapping'])=={f'ABL{i}' for i in range(10)}))
 test('width_domain_frozen',lambda:req(c['width_efficiency']['candidate_domain']==[-8.0,8.0] and c['width_efficiency']['full_real_line_width_claim_allowed'] is False))
 test('width_subset_50',lambda:req(len(c['width_efficiency']['hero_case_ids'])*len(c['width_efficiency']['fixed_case_local_rep_indices'])==50))
 test('matched_coverage_tolerance',lambda:eq(c['width_efficiency']['matched_coverage_absolute_tolerance'],.03))
 test('coverage_independent_from_tuning',lambda:req(c['coverage_confirmation']['independent_of_threshold_tuning'] is True))
 # Accepted inputs and source locks.
 hashes=json.loads((ROOT/'configs/expected_upstream_hashes.json').read_text());locks=json.loads((ROOT/'configs/source_locks.json').read_text())
 test('five_accepted_output_hashes',lambda:req(set(hashes)=={'stage3a_output_zip','stage3b_output_zip','stage3d_d4_output_zip','stage3e_output_zip','stage3f_output_zip'} and all(len(v)==64 for v in hashes.values())))
 def exact_outputs():
  for k,h in hashes.items():eq(sha256_file(rp[k]),h)
 test('accepted_output_bytes_exact',exact_outputs)
 def exact_sources():
  for key,lock in locks.items():
   for rel,h in lock['files'].items():eq(sha256_file(rp[key]/rel),h)
 test('locked_source_bytes_exact',exact_sources)
 test('source_lock_counts',lambda:req({k:len(v['files']) for k,v in locks.items()}=={'stage3b_source_root':2,'stage3c_source_root':18,'stage3d_d2_source_root':41,'stage3d_d4_source_root':39,'stage3e_source_root':30}))
 # Stage3F vendored scientific adapter byte lock.
 vl=json.loads((ROOT/'configs/stage3f_vendor_lock.json').read_text())
 test('stage3f_vendor_7_files',lambda:eq(len(vl['files']),7))
 def vendor_exact():
  for rel,h in vl['files'].items():eq(sha256_file(ROOT/'vendor/geodose_stage3f'/rel),h)
 test('stage3f_vendor_bytes_exact',vendor_exact)
 # Frozen Stage3F output thresholds.
 def stage3f_threshold_zip():
  with zipfile.ZipFile(rp['stage3f_output_zip']) as z:
   n=[x for x in z.namelist() if x.endswith('stage3f_operational_thresholds_FROZEN.json')];req(len(n)==1);t=json.loads(z.read(n[0]));req((float(t['minimum_ess']),float(t['maximum_normalized_weight']),int(t['minimum_graph_safe_count']))==(3.0,.5,20))
 test('stage3f_threshold_zip_exact',stage3f_threshold_zip)
 # Registry exactness.
 b=Stage3BBridge(rp['stage3b_source_root'],rp['stage3a_output_zip'],rp['stage3b_output_zip']);adapter=install_stage3b_production_adapter(b.core,b.stage3a)
 seeds=b.stage3a['frames']['seed_registry.csv'];mainreg=build_main_registry(c,seeds);f06reg=build_f06_registry(c);extreg=build_extension_registry(c);rs=registry_summary(mainreg,f06reg,extreg)
 test('registry_counts',lambda:req(rs=={'main_rows':3000,'f06_extension_rows':600,'stress_extension_rows':590,'method_evaluable_rows':4090,'structural_only_rows':100,'total_case_replication_rows':4190}))
 test('main_uses_all_3000_unique_seed_rows',lambda:req(len({(r['scenario_id'],r['seed_replication']) for r in mainreg})==3000))
 test('f06_s4_matching_seed_window',lambda:req([r['seed_replication'] for r in f06reg if r['case_id']=='S4_RHO060' and r['f06_level']=='small'][:3]==[301,302,303] and [r['seed_replication'] for r in f06reg if r['case_id']=='S4_RHO060' and r['f06_level']=='small'][-1]==400))
 test('f06_s1_s5_seed_windows',lambda:req(all(min(r['seed_replication'] for r in f06reg if r['case_id']==cid and r['f06_level']=='small')==1 and max(r['seed_replication'] for r in f06reg if r['case_id']==cid and r['f06_level']=='small')==100 for cid in ['S1_BASE','S5_POOR_OVERLAP'])))
 test('f06_case_local_indices_1_100',lambda:req(set(r['case_local_index'] for r in f06reg)==set(range(1,101))))
 test('production_seed_adapter',lambda:req(adapter['seed_phase']=='production_provisional'))
 test('f06_structural_qa_exact',lambda:req(len(f06_structural_qa(b.core,c))==3))
 # Real generator structural checks on F06 representative cases, no performance outcomes inspected.
 def real_f06_generation():
  for baseid in c['f06_freeze']['representative_case_ids']:
   for level in ['small','large']:
    base=b.cases[baseid];case=f06_case(base,level);rep=301 if baseid=='S4_RHO060' else 1;g=b.core.generate_case(case,b.stage3a,b.basis_cache,replication=rep);u=g['units'];e=g['true_edges'];req(len(u)==(289 if level=='small' else 1225));req(set(u.role.astype(str))=={'nuisance_training','support_audit','calibration','test_target'});req(len(eligible_target_blocks(u,e[['source_node','target_node']],6))>0);req(np.isfinite(g['hard_truth'].Y_true.to_numpy(float)).all())
 test('real_f06_generation_structurally_valid',real_f06_generation)
 # Primary remains exact original configuration.
 test('primary_case_unchanged_by_f06',lambda:req(f06_case(b.cases['S1_BASE'],'primary')==b.cases['S1_BASE']))
 # Target selection remains outcome-blind structural and connected.
 def target_connected():
  case=f06_case(b.cases['S4_RHO060'],'small');g=b.core.generate_case(case,b.stage3a,b.basis_cache,replication=301);ed=g['true_edges'][['source_node','target_node']];el=eligible_target_blocks(g['units'],ed,6);node,block,meta=select_uniform_eligible_target(b.core,b.stage3a,case,301,el);req(block_is_connected(block,ed));req(meta['target_selection_rank']>=0)
 test('f06_target_selection_connected',target_connected)
 # Operational gate cannot drift from frozen thresholds.
 def gates():
  th={'minimum_ess':3.,'maximum_normalized_weight':.5,'minimum_graph_safe_count':20}
  base={'target_population_eligible':True,'target_supported':True,'endpoint_requested':False,'endpoint_audited':True,'ess':2.9,'max_normalized_weight':.1,'graph_safe_count':30,'operational_coverage_lower_bound':.8}
  s=pd.Series(base);eq(operational_gate('M6',s,th)[1],'R03_LOW_EFFECTIVE_CALIBRATION_SIZE');s.ess=4;s.max_normalized_weight=.6;eq(operational_gate('M6',s,th)[1],'R04_WEIGHT_CONCENTRATION');s.max_normalized_weight=.2;s.graph_safe_count=19;eq(operational_gate('M6',s,th)[1],'R05_GRAPH_COMPONENT_TOO_SMALL');s.graph_safe_count=21;s.operational_coverage_lower_bound=0.;eq(operational_gate('M6',s,th)[1],'R08_CERTIFIED_DEFICIT_VACUOUS')
 test('operational_gate_stage3f_thresholds',gates)
 # Runtime/API locks.
 c3=Stage3CBridge(rp['stage3c_source_root']);install_stage3c_production_nuisance_adapter(c3.runner)
 def nuisance_seed_prod():
  src,seed=c3.runner.frozen_nuisance_seed(seeds,'S4',301);req(src=='S4' and isinstance(seed,int))
 test('stage3c_nuisance_seed_production_phase',nuisance_seed_prod)
 def d2api():
  import importlib
  src=rp['stage3d_d2_source_root']/'src';sys.path.insert(0,str(src));m=importlib.import_module('geodose_stage3d.candidate_inversion');ex=importlib.import_module('geodose_stage3d.exact_law')
  eq(str(inspect.signature(ex.prepare_orbit)),"(fixture: 'ExactFixture', *, minimum_precision_eigenvalue_required: 'float') -> 'PreparedOrbit'")
  req('non_gaussian_min_abs_residual' in inspect.signature(m.CandidateEvaluator).parameters)
 test('accepted_d2_api_lock',d2api)
 def eapi():
  import importlib
  src=rp['stage3e_source_root']/'src';sys.path.insert(0,str(src));n=importlib.import_module('geodose_stage3e.gaussian_n2');req('minimum_precision_eigenvalue' in inspect.signature(n.build_vecchia_family).parameters)
 test('accepted_stage3e_api_lock',eapi)
 # Deterministic and provenance guards.
 def dgzip():
  with tempfile.TemporaryDirectory() as td:
   df=pd.DataFrame({'a':[1,2],'b':[.1,.2]});p1=Path(td)/'a.gz';p2=Path(td)/'b.gz';deterministic_gzip_csv(df,p1);deterministic_gzip_csv(df,p2);eq(sha256_file(p1),sha256_file(p2))
 test('deterministic_gzip',dgzip)
 test('source_inventory_excludes_resolved',lambda:req('configs/resolved_paths_windows.json' not in {r['path'] for r in source_inventory(ROOT)['files']}))
 test('requirements_pinned',lambda:req((ROOT/'requirements_py310.txt').read_text().splitlines()[:7]==['numpy==2.2.6','pandas==2.3.3','scipy==1.15.3','scikit-learn==1.7.2','xgboost==3.0.5','joblib==1.5.2','PyYAML==6.0.3']))
 # Static scientific guardrails.
 runner=(ROOT/'src/geodose_stage4/runner.py').read_text();f06txt=(ROOT/'src/geodose_stage4/f06.py').read_text();regtxt=(ROOT/'src/geodose_stage4/production_registry.py').read_text();pre=(ROOT/'preflight_windows.py').read_text()
 test('runner_reuses_stage3f_evaluator',lambda:req('sf_runner._evaluate_track' in runner))
 test('runner_reuses_stage3e_sparse_precision',lambda:req('sf_runner._get_sparse' in runner and 'patch_sparse_precision' in runner))
 test('runner_width_full_line_claim_false',lambda:req("'full_real_line_claim':False" in runner and "'M6_full_real_line_width_claim':False" in runner))
 test('runner_no_threshold_selector',lambda:req('choose_thresholds' not in runner))
 test('runner_no_nsw_data_or_tuning',lambda:req("contract['nsw_tuning_allowed'] is False" in runner and "'nsw_used_for_tuning':False" in runner))
 test('runner_fail_closed_false_support',lambda:req('PRODUCTION_FALSE_SUPPORT_PRESENT' in runner and 'PRODUCTION_COMPUTATIONAL_FAILURE_PRESENT' in runner))
 test('runner_checkpoint_provenance',lambda:req('CHECKPOINT_PROVENANCE_MISMATCH' in runner and 'atomic_gzip_json' in runner))
 test('runner_ABL8_holds_response_graph_fixed',lambda:req('coverage_difference_vs_oracle_treatment' in runner and "RF_ORACLE_SPATIAL_LAW_DIAGNOSTIC' if scen=='S6'" in runner))
 test('runner_ABL9_summary_present',lambda:req("'ABL9'" in runner and 'registered fitted graph versus oracle true graph' in runner))
 test('runner_local_coverage_fifth_percentile',lambda:req('_local_coverage_metrics' in runner and 'fifth_percentile_local_coverage' in runner))
 test('runner_spatial_diagnostics_not_certificate',lambda:req('_spatial_residual_diagnostic' in runner and "'used_in_conformal_deficit':False" in runner))
 test('f06_delegates_primary',lambda:req('return orig_role' in f06txt and 'return orig_cuts' in f06txt))
 test('f06_registry_matching_seed_window_static',lambda:req('start=int(a[\'seed_rep_start\'])' in regtxt and 'range(start,start+n)' in regtxt))
 test('preflight_before_environment_design',lambda:req('BEFORE environment setup' in pre and 'resolved_paths_windows.json' in pre))
 test('no_dense_precision_inverse_scientific_src',lambda:req('np.linalg.inv' not in ''.join(p.read_text(errors='ignore') for p in (ROOT/'src').rglob('*.py'))))
 test('no_network_scientific_src',lambda:req(all(tok not in ''.join(p.read_text(errors='ignore') for p in (ROOT/'src').rglob('*.py')) for tok in ['requests.','urllib.request','http://','https://'])))
 print(f'STAGE4 UNIT TESTS PASSED ({passed}/{passed})')

if __name__=='__main__':main()
