from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
for p in [ROOT/'vendor',ROOT/'src']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from geodose_stage3d_d3_integration.io import sha256_file, read_zip_csv
from geodose_stage3d_d3_integration.provenance import source_inventory

OUT=ROOT/'outputs_stage3d_d3_integration'
checks={}
def ck(name,cond): checks[name]=bool(cond)

def loadj(n): return json.loads((OUT/n).read_text())

manifest=loadj('stage3d_d3_integration_manifest.json')
for r in manifest['files']:
    fp=OUT/r['name']; ck('manifest_'+r['name'],fp.is_file() and fp.stat().st_size==r['size'] and sha256_file(fp)==r['sha256'])
source=loadj('stage3d_d3_integration_source_hashes.json'); current=source_inventory(ROOT)
ck('source_aggregate',source['aggregate_sha256']==current['aggregate_sha256'])
ck('source_count',source['file_count']==current['file_count'])
input_audit=loadj('stage3d_d3_integration_input_audit.json')
ck('upstream',input_audit['all_upstream_pass'])
ck('stage3c_source_tree',input_audit['stage3c_source_tree']['pass'])
ck('d2_source_tree',input_audit['d2_source_tree']['pass'])
ck('vendor_source',input_audit['vendor_frozen_source_audit']['all_pass'])
replay=loadj('stage3d_d3_stage3c_baseline_replay_summary.json')
ck('stage3c_replay',replay['all_groups_pass'] and set(replay['principal_methods_replayed'])=={'M1','M2','M5'})
m3m4=loadj('stage3d_d3_m3_m4_regression.json')
ck('m3m4_status',m3m4['m3_status']=='verified_complete' and m3m4['m4_status']=='verified_complete')
ck('factorizing_reduction',m3m4['factorizing_max_abs_error']<=2e-12)
ck('nonfactorization',m3m4['s4_all_nonfactorization_detected'] and m3m4['s4_max_delta_fact']>1e-8)
d2=loadj('stage3d_d3_m6_d2_regression.json')
ck('d2_regression',d2['conservative_rule_mismatch_count']==0 and d2['randomized_not_subset_count']==0 and d2['max_probability_sum_abs_error']<=1e-10)
trace=pd.read_csv(OUT/'stage3d_d3_s4_common_candidate_trace.csv.gz')
ck('six_methods',set(trace.method.unique())=={'M1','M2','M3','M4','M5','M6'})
counts=trace.groupby('method').size(); ck('equal_candidate_counts',counts.nunique()==1 and int(counts.iloc[0])>=2000)
ck('no_heuristics_promoted',not trace.method.isin(['H1','H4']).any())
for m in ['M3','M4','M6']:
    g=trace[trace.method==m]; ck('pvalues_'+m,g.p_value.between(0,1).all())
# At least one M4/M6 p-value differs on the accepted full S4 grid.
p4=trace[trace.method=='M4'][['candidate_hex','p_value']].rename(columns={'p_value':'p4'})
p6=trace[trace.method=='M6'][['candidate_hex','p_value']].rename(columns={'p_value':'p6'})
mm=p4.merge(p6,on='candidate_hex',validate='one_to_one'); ck('m4_m6_not_identical',float(np.max(np.abs(mm.p4-mm.p6)))>1e-8)
red=pd.read_csv(OUT/'stage3d_d3_reduction_status.csv')
status=dict(zip(red.reduction_id,red.status)); ck('R1',status.get('R1')=='PASS'); ck('R2',status.get('R2')=='PASS'); ck('R3_deferred',status.get('R3')=='DEFERRED_STAGE3E'); ck('R4',status.get('R4')=='PASS')
claim=loadj('stage3d_d3_claim_boundary.json')
ck('claim_no_pilot',claim['pilot_run'] is False and claim['production_run'] is False and claim['publication_evidence'] is False)
ck('claim_m6_not_pilot_ready',claim['M6_new_data_pilot_engine_ready'] is False)
ck('domain_claim',claim['D2_full_real_line_claim'] is False and claim['D2_registered_domain']==[-8.0,8.0])
api=loadj('stage3d_d3_d2_api_inventory.json')
mods=set(api['modules'])
ck('d2_api_inventory',all(any(x.endswith(s) for x in mods) for s in ['candidate_inversion.py','exact_law.py','d2_runner.py']))
failed=[k for k,v in checks.items() if not v]
ver={'status':'verified_complete' if not failed else 'failed','check_count':len(checks),'failed_checks':len(failed),'failed_check_names':failed,'checks':checks,'stage':'Stage3D_D3','script_version':'1.0.1-stage3d-d3-m1-m6-exact-integration','pilot_run':False,'production_run':False,'M1_M6_common_interface':True,'M6_new_data_pilot_engine_ready':False,'R3_deferred_to_Stage3E':True}
(OUT/'STAGE3D_D3_INTEGRATION_VERIFICATION.json').write_text(json.dumps(ver,indent=2,sort_keys=True),encoding='utf-8')
if failed:
    print('VERIFICATION FAILED',failed); raise SystemExit(1)
print(f"STAGE 3D D3 M1-M6 EXACT INTEGRATION VERIFIED_COMPLETE ({len(checks)}/{len(checks)})")
print('M6 new-data pilot engine: NOT YET; R3: DEFERRED STAGE3E')
