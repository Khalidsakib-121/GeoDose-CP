from __future__ import annotations
import json, math, sys, tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT/'vendor',ROOT/'src']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))

from geodose_stage3d_d4.d2_runtime import (
    D2SourceAdapter, _as_bool, _construct_candidate_evaluator, _candidate_result_summary,
    _mutate_config, _prepare_orbit_with_frozen_precision, _tie_uniform, _load_api_lock,
)
from geodose_stage3d_d4.runner import _verification_failure_summary
from geodose_stage3d_d4.data import frozen_target_unit
from geodose_stage3d_d4.fixture import canonical_edges, block_connected, edge_degrees, discover_data_fields
from geodose_stage3d_d4.io import write_csv_gz
from geodose_stage3d_d3_m4.orbit import numeric_key
from geodose_stage3d_d3_m4.m4_oracle import evaluate_m4

passed=0
def check(name, cond):
    global passed
    if not cond: raise AssertionError(name)
    passed+=1; print('PASS',name)

# Signed-zero patch inherited byte-for-byte from accepted M4 source.
check('accepted_signed_zero_key',numeric_key(-0.0)==numeric_key(+0.0)==numeric_key(0.0))

# M4 one-normalization smoke test.
s=np.asarray([0.1,0.2,0.3,0.15,0.15,0.1]); logd=np.log(np.asarray([1.,2.,3.,4.,5.,6.])); r=np.asarray([-2,-1,-.5,.5,1,2.])
out=evaluate_m4(s,r,logd,5,tie_tolerance=1e-12)
raw=s*np.exp(logd); q=raw/raw.sum()
check('M4_one_normalization',np.max(np.abs(q-out.q4))<1e-14 and abs(out.q4.sum()-1)<1e-14)

# Graph helpers are deterministic and undirected-canonical.
e=pd.DataFrame({'source_node':[2,1,1,3],'target_node':[1,2,3,1]})
ce=canonical_edges(e)
check('canonical_edges',len(ce)==2 and set(map(tuple,ce.to_numpy()))=={(1,2),(1,3)})
check('block_connectivity',block_connected([1,2,3],ce))
d=edge_degrees(ce,4); check('edge_degrees',d.tolist()==[0,2,1,1])


# Stage3B contains many test_target units; D4 exact inference must freeze node 320 specifically.
target_fixture=pd.DataFrame({
    'node_index':[20,320,321],
    'role':['test_target','test_target','calibration'],
    'unit_id':['wrong_test_target','frozen_target_320','calibration_321'],
})
tsel=frozen_target_unit(target_fixture,320)
check('frozen_target_selector_uses_node320_not_role_only',int(tsel['node_index'])==320 and str(tsel['unit_id'])=='frozen_target_320')
try:
    frozen_target_unit(target_fixture[target_fixture.node_index!=320],320)
    missing_target_rejected=False
except Exception:
    missing_target_rejected=True
check('frozen_target_selector_missing_node320_fails_closed',missing_target_rejected)

# D2 oracle Stage3B container may legitimately omit a separate fitted-edge field.
class TinyData:
    pass
td=TinyData()
td.units=pd.DataFrame({'case_id':['c'],'unit_id':['u'],'node_index':[0],'role':['calibration'],'A':[.2],'pi_atom_0':[.1],'beta_alpha':[2.]})
td.cases=pd.DataFrame({'case_id':['c'],'scenario_id':['S'],'spatial_rho':[.2],'residual_law':['gaussian_gmrf']})
td.edges=pd.DataFrame({'case_id':['c'],'source_node':[0],'target_node':[1]})
td.fixtures={'size6_unique':{}}
flds=discover_data_fields(td)
check('D2_data_container_fitted_edges_optional',flds['units']=='units' and flds['true_edges']=='edges' and 'fitted_edges' not in flds)

# D2 config adapter must mutate required semantic fields without changing input.
base={'case_id':'OLD','target_dose':0.9,'fixture_source':'size6_unique','candidate_rule':'x'}
new=_mutate_config(base,case_id='NEW',target_dose=.75,fixture_source='size6_unique',candidate_rule='stored_simulation_truth')
check('D2_config_copy',base['case_id']=='OLD' and new['case_id']=='NEW' and abs(new['target_dose']-.75)<1e-15)

# CandidateEvaluator bridge is locked to the exact accepted D2 v1.2 constructor.
class FakeResult:
    conservative_p=.73; randomized_p=.61; tie_uniform=.4
    conservative_accept=True; randomized_accept=True
    singular_conservative_inclusion=False; distinct_states=720
class FakeEvaluator:
    def __init__(self,prepared,*,alpha,tie_uniform,score_abs_tolerance,score_rel_tolerance,non_gaussian_min_abs_residual):
        self.prepared=prepared; self.alpha=alpha; self.tie_uniform=tie_uniform
        self.values=(score_abs_tolerance,score_rel_tolerance,non_gaussian_min_abs_residual)
    def evaluate(self,y,*,keep_state=False): return FakeResult()
def fake_acceptance(result,mode):
    return bool(result.conservative_accept if mode=='conservative' else result.randomized_accept)
exact_tol={'score_tie_abs_tolerance':1e-12,'score_tie_rel_tolerance':1e-12,'non_gaussian_min_abs_residual':1e-14}
ev,meta=_construct_candidate_evaluator(FakeEvaluator,object(),.1,.4,exact_tol)
summary=_candidate_result_summary(ev.evaluate(0),fake_acceptance,.1)
check('D2_CandidateEvaluator_exact_bridge',abs(summary['conservative_p']-.73)<1e-15 and abs(summary['randomized_p']-.61)<1e-15 and summary['conservative_accept'] and summary['randomized_accept'])
check('D2_CandidateEvaluator_non_gaussian_tolerance_bound',ev.values==(1e-12,1e-12,1e-14) and 'non_gaussian_min_abs_residual' in meta['bound_parameters'])
check('safe_boolean_parser',_as_bool(True) and _as_bool('true') and (not _as_bool('False')) and (not _as_bool(0)))

# Missing non-Gaussian tolerance must fail closed rather than guess.
try:
    _construct_candidate_evaluator(FakeEvaluator,object(),.1,.4,{'score_tie_abs_tolerance':1e-12,'score_tie_rel_tolerance':1e-12})
    missing_ng_rejected=False
except Exception:
    missing_ng_rejected=True
check('D2_CandidateEvaluator_missing_non_gaussian_tolerance_rejected',missing_ng_rejected)

# Endpoint/interior D2 configuration family selection must be preserved by the adapter.
class FakeD2IO:
    def load_d1_fixture_config(self,root,eid):
        return {'case_id':'OLD','target_dose':.9,'fixture_source':'size6_unique','candidate_rule':'stored_simulation_truth','loaded_id':eid}
    def build_d2_fixture(self,data,root,eid):
        return {'eid':eid,'cfg':self.load_d1_fixture_config(root,eid)}
class FakeExact:
    def __init__(self): self.seen_threshold=None
    def prepare_orbit(self,fixture,*,minimum_precision_eigenvalue_required):
        self.seen_threshold=float(minimum_precision_eigenvalue_required); return fixture
a=object.__new__(D2SourceAdapter); a.source_root=Path('.'); a.d2_io=FakeD2IO(); a.exact=FakeExact(); a.tolerances={'minimum_precision_eigenvalue':1e-10}
_,m0=a._build_prepared({},'NEW_CASE',0.0); _,m1=a._build_prepared({},'NEW_CASE',1.0); _,mi=a._build_prepared({},'NEW_CASE',0.75)
check('D2_endpoint_config_family',m0['base_evaluation_id']=='G6_GAUSS_ENDPOINT0' and m1['base_evaluation_id']=='G6_GAUSS_ENDPOINT1' and mi['base_evaluation_id']=='G6_GAUSS_INTERIOR_TRUTH')
check('D2_prepare_orbit_frozen_precision_bound',abs(a.exact.seen_threshold-1e-10)<1e-30 and abs(mi['prepare_orbit']['minimum_precision_eigenvalue_required']-1e-10)<1e-30)

# Accepted D2 prepare_orbit signature is fail-closed if the required frozen precision keyword disappears.
class BadExact:
    def prepare_orbit(self,fixture): return fixture
try:
    _prepare_orbit_with_frozen_precision(BadExact(),{}, {'minimum_precision_eigenvalue':1e-10})
    bad_rejected=False
except Exception:
    bad_rejected=True
check('D2_prepare_orbit_signature_fail_closed',bad_rejected)

# Exact D2 randomization scheme: always S4 replication 1, then label + fixture id.
class SeedIO:
    calls=[]
    @staticmethod
    def frozen_orbit_seed(root,scenario,replication):
        SeedIO.calls.append(('base',scenario,replication)); return 12345
    @staticmethod
    def derive_seed(base,*labels):
        SeedIO.calls.append(('derive',base,*labels)); return 98765
u,sm=_tie_uniform(SeedIO(),Path('.'),'FIXTURE_X',replication=1,label='D2_G3_TIE_UNIFORM')
check('D2_tie_seed_uses_S4_rep1',SeedIO.calls[0]==('base','S4',1))
check('D2_tie_seed_uses_label_and_fixture',SeedIO.calls[1]==('derive',12345,'D2_G3_TIE_UNIFORM','FIXTURE_X'))
check('D2_tie_uniform_fixed_metadata',sm['seed_scenario']=='S4' and sm['tie_fixture_id']=='FIXTURE_X' and sm['derived_subseed_label']=='D2_G3_TIE_UNIFORM' and 0<=u<=1)

# Candidate-result acceptance must use accepted D2 acceptance API; no silent p>alpha fallback.
def raising_acceptance(result,mode): raise TypeError('deliberate API failure')
try:
    _candidate_result_summary(FakeResult(),raising_acceptance,.1)
    acceptance_failure_rejected=False
except Exception:
    acceptance_failure_rejected=True
check('D2_acceptance_API_failure_is_fail_closed',acceptance_failure_rejected)

# D2 API lock shipped in the package contains every discovered frozen required argument.
api_lock=_load_api_lock(ROOT)
check('D2_API_lock_source_tree_41',api_lock['accepted_d2_source_file_count']==41 and len(api_lock['accepted_d2_source_tree_hash'])==64)
check('D2_API_lock_CandidateEvaluator_non_gaussian','non_gaussian_min_abs_residual' in api_lock['signatures']['CandidateEvaluator'])
check('D2_API_lock_prepare_orbit_precision','minimum_precision_eigenvalue_required' in api_lock['signatures']['prepare_orbit'])
check('D2_API_lock_randomization_exact',api_lock['randomization_contract']['scenario_mapping']=='all_D2_fixtures_use_S4' and api_lock['randomization_contract']['derived_subseed_label']=='D2_G3_TIE_UNIFORM')
check('D2_API_lock_frozen_tolerances',api_lock['frozen_tolerances']['minimum_precision_eigenvalue']==1e-10 and api_lock['frozen_tolerances']['non_gaussian_min_abs_residual']==1e-14)

# Deterministic gzip writing: same frame -> same bytes.
with tempfile.TemporaryDirectory() as td:
    td=Path(td); f=pd.DataFrame({'a':[1.,2.],'b':['x','y']}); p1=td/'a.csv.gz'; p2=td/'b.csv.gz'; write_csv_gz(f,p1); write_csv_gz(f,p2)
    check('deterministic_gzip',p1.read_bytes()==p2.read_bytes())

# Frozen upstream verification JSON schemas differ legitimately across accepted stages.
stage3b_schema={'status':'verified_complete','checks':{'a':'pass','b':'pass'}}
stage3c_schema={'status':'verified_complete','checks':{'a':'pass'},'failed_checks':{}}
new_schema={'status':'verified_complete','checks':{'a':True},'failed_checks':0}
failed_mapping_schema={'status':'verified_complete','checks':{'a':'pass','b':'fail'},'failed_checks':{'b':'failed'}}
check('verification_schema_stage3b_missing_failed_checks',_verification_failure_summary(stage3b_schema)['count']==0)
check('verification_schema_stage3c_dict_failed_checks',_verification_failure_summary(stage3c_schema)['count']==0)
check('verification_schema_new_integer_failed_checks',_verification_failure_summary(new_schema)['count']==0)
check('verification_schema_nonempty_failure_mapping_detected',_verification_failure_summary(failed_mapping_schema)['count']==1)

# Configuration preserves claim boundary and accepted hashes count.
contract=json.loads((ROOT/'configs'/'expected_upstream_hashes.json').read_text())
check('eight_frozen_upstream_hashes',len(contract)==8 and all(len(v)==64 for v in contract.values()))


# v1.1.0 launcher hardening regression tests.
bat=(ROOT/'RUN_STAGE3D_D4.bat').read_text(encoding='utf-8',errors='replace')
required_launcher_tokens=['\\AppData\\Local\\Temp\\','.zip.','robocopy','GeoDose_Stage3D_D4_M6_NewData_Nuisance_Integration_v1_1_0']
check('launcher_zip_temp_self_relocation',all(x.lower() in bat.lower() for x in required_launcher_tokens))
check('launcher_refuses_incomplete_temp_materialization','Windows did not materialize the full package from the ZIP' in bat and 'configs\\d4_contract.yaml' in bat and 'src\\geodose_stage3d_d4\\runner.py' in bat)
check('launcher_rebuilds_partial_venv','virtual environment is incomplete or incompatible. Rebuilding' in bat and 'rmdir /s /q ".venv"' in bat)

# v1.1.0 regression: the accepted D3 integration is v1_0_1, never the nonexistent v1_0_2.
local_paths=json.loads((ROOT/'configs'/'local_paths_windows.json').read_text())
d3_paths=local_paths['stage3d_d3_integration_output_zip']
check('accepted_D3_integration_path_is_v1_0_1',all('v1_0_1' in x for x in d3_paths) and all('v1_0_2' not in x for x in d3_paths))
fallback=json.loads((ROOT/'configs'/'fallback_globs_windows.json').read_text())
check('D3_integration_hash_validated_fallback_configured','stage3d_d3_integration_output_zip' in fallback and any('M1_M6_Exact_Integration_v1_0*' in x for x in fallback['stage3d_d3_integration_output_zip']))


# Current release/version provenance must be internally consistent.
from geodose_stage3d_d4 import __version__ as d4_version
import yaml
contract_yaml=yaml.safe_load((ROOT/'configs'/'d4_contract.yaml').read_text())
prov_text=(ROOT/'src'/'geodose_stage3d_d4'/'provenance.py').read_text(encoding='utf-8')
check('D4_current_version_metadata_consistent',d4_version=='1.1.0-stage3d-d4' and contract_yaml.get('script_version')=='1.1.0-stage3d-d4' and "'version':'1.1.0'" in prov_text)
current_files=['START_HERE.txt','README.md','RUN_STAGE3D_D4.bat','RUN_STAGE3D_D4.ps1','docs/PRE_SEND_VALIDATION.md']
check('D4_current_paths_all_v1_1_0',all('v1_1_0' in (ROOT/f).read_text(encoding='utf-8',errors='replace') for f in current_files))


# v1.1.0 pre-venv path/provenance hardening: fail fast before downloading/installing dependencies.
exact_expected={
    'stage3a_output_zip':'9a46e21f4f488447f0829f079c8f19a345e6ac1748bc6907e20917bbb096d406',
    'stage3b_output_zip':'a7e7dc14c73a7d6e88ad7ce40ed3d0f506ee862554ce9de25d922c730dced62e',
    'stage3c_output_zip':'48690382a271295593c5edfa755945a88cc1a5bc50a2133184ce988666a2b67a',
    'stage3d_d1_output_zip':'2b58cbe85fa177d23181d37a4f40764868a86edf12c9114a77a69a9c2b9b9df4',
    'stage3d_d2_output_zip':'fd2af9c4fb4f6563a3b2e0a56559425b832a1dde330624964470c29ea983ddf1',
    'stage3d_d3_m3_output_zip':'cac7be71bae87020b4106863ab178da02812ec3665cfc1331e811e1831c3111b',
    'stage3d_d3_m4_output_zip':'8aa59f04612c229ded1874a7062354f999f0f78b348ff5059b16b7d28e69e775',
    'stage3d_d3_integration_output_zip':'f20320ec72ae115709bc809d72a7c8f37f75f12a335c271a895c521900d1a51d',
}
check('exact_frozen_upstream_hash_registry',contract==exact_expected)
preflight=(ROOT/'preflight_windows.py').read_text(encoding='utf-8')
check('prevenv_windows_input_preflight_present','D4 WINDOWS INPUT PREFLIGHT PASSED' in preflight and 'stage3d_d2_source_hashes.json' in preflight and 'stage3c_source_hashes.json' in preflight and 'stage3d_d3_m4_source_hashes.json' in preflight)
idx_pre=bat.find('py -3.10 preflight_windows.py'); idx_venv=bat.find('py -3.10 -m venv .venv')
check('prevenv_input_preflight_runs_before_venv',idx_pre>=0 and idx_venv>=0 and idx_pre<idx_venv)

print(f'STAGE3D D4 UNIT TESTS PASSED ({passed}/{passed})')
