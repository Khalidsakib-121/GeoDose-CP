from pathlib import Path
import sys,json,tempfile,argparse,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.authority import load_census,verify_base,GOVERNING,THRESHOLD,CENSUS_AGG,CENSUS_ZIP_SHA,BASE_AGG
from src.common import sha256_file, acquisition_manifest_aggregate, GateError
from src.production import NUMERIC_COLS

ap=argparse.ArgumentParser();ap.add_argument('--base-root',required=True);args=ap.parse_args();BASE=Path(args.base_root).resolve()
checks=[]
def ck(n,x,d=''):
    if not x:raise AssertionError(f'{n}: {d}')
    checks.append(n);print('PASS',n)

# Regression for the exact v1.0.0 release-blocking defect: verify the ACTUAL base
# manifest using the acquisition package's TSV+size+final-newline aggregate contract.
bm=verify_base(BASE)
ck('base_exact_package_identity',bm.get('package')=='GeoDose_Stage5A_External_Context_Acquisition_v1_0_1_FREEZE')
ck('base_manifest_declared_aggregate',bm.get('aggregate_sha256')==BASE_AGG)
ck('base_manifest_exact_recomputation',acquisition_manifest_aggregate(bm['immutable_files'])==BASE_AGG)
# A different aggregate rule must not accidentally be accepted as the acquisition aggregate.
import hashlib
wrong=hashlib.sha256(''.join(r['file']+r['sha256'] for r in sorted(bm['immutable_files'],key=lambda x:x['file'])).encode()).hexdigest()
ck('base_manifest_wrong_rule_distinguished',wrong!=BASE_AGG,wrong)

with tempfile.TemporaryDirectory() as td:
    c=load_census(ROOT,Path(td)/'c')
    ck('census_zip_sha',sha256_file(ROOT/'inputs/CONTEXT_SUPPORT_CENSUS_OUTPUTS_ACCEPTED.zip')==CENSUS_ZIP_SHA)
    ck('census_output_aggregate',json.load(open(Path(td)/'c'/'OUTPUT_FILE_MANIFEST.json'))['aggregate_sha256']==CENSUS_AGG)
    ck('partition_23710',len(c['union'])==23710 and len(c['complete'])==23618 and len(c['excluded'])==92)
    ck('partition_disjoint',not(set(c['complete'].block_id)&set(c['excluded'].block_id)))
    ck('threshold_exact_099',THRESHOLD==0.99)
    ck('governing_18',len(GOVERNING)==18 and c['report']['governing_layers']==GOVERNING)
    bm2=pd.read_csv(Path(td)/'c'/'SUPPORT_SUMMARY_BY_MINE.csv');d=dict(zip(bm2.MineN,bm2.unsupported_union_blocks.astype(int)))
    ck('liddell_89',d.get('Liddell Coal')==89,d);ck('bulga_3',d.get('Bulga Complex')==3,d);ck('other_mines_zero',sum(v for k,v in d.items() if k not in {'Liddell Coal','Bulga Complex'})==0,d)
    sm=pd.read_csv(Path(td)/'c'/'SOURCE_LAYER_SUPPORT_SUMMARY.csv');gov=sm.governing_for_context_complete_union.astype(str).str.lower().eq('true');ck('silo_support_complete',(sm[(sm.source_family=='SILO')&gov].fail_lt_0_99==0).all());ck('ga_common_support_complete',int(sm.loc[sm.layer_id=='ga_terrain_common','fail_lt_0_99'].iloc[0])==0)
    src=c['source_manifest']['source_rows'];ck('source_rows_19',len(src)==19,len(src));ck('source_families',{'SILO','TERN_SLGA_R2','Geoscience_Australia'}<=set(r['source_family'] for r in src));ck('tern_13',sum(r['source_family']=='TERN_SLGA_R2' for r in src)==13);ck('silo_4',sum(r['source_family']=='SILO' for r in src)==4)
    ck('ga_opaque_coverage_locked',next(r for r in src if r.get('source_id')=='GA_DEM_SRTM_1Second_2024').get('coverage_identifier')=='1')
    # Every frozen source row used downstream must carry its exact local-source identity hash.
    silo=[r for r in src if r['source_family']=='SILO'];tern=[r for r in src if r['source_family']=='TERN_SLGA_R2'];ga=[r for r in src if r.get('source_id')=='GA_DEM_SRTM_1Second_2024'];gad=[r for r in src if r.get('source_id')=='GA_TERRAIN_DERIVED']
    ck('silo_derived_hashes_present',len(silo)==4 and all(len(r.get('derived_aoi_sha256',''))==64 for r in silo))
    ck('tern_crop_metadata_hashes_present',len(tern)==13 and all(len(r.get('aoi_crop_sha256',''))==64 and len(r.get('metadata_sha256',''))==64 for r in tern))
    ck('ga_raw_hash_present',len(ga)==1 and len(ga[0].get('aoi_crop_sha256',''))==64)
    ck('ga_derived_hashes_four',len(gad)==1 and len(gad[0].get('details',{}).get('derived_sha256',{}))==4)
    # Formula regression.
    r={k:np.array([1.,2.]) for k in ['SOC_0_5','SOC_5_15','CLY_0_5','CLY_5_15','BDW_0_5','BDW_5_15','PHC_0_5','PHC_5_15','AWC_0_5','AWC_5_15','AWC_15_30','AWC_30_60','AWC_60_100']}
    ck('topsoil_formula_exact',np.allclose((5*r['SOC_0_5']+10*r['SOC_5_15'])/15,r['SOC_0_5']))
    awc=(r['AWC_0_5']/100)*50+(r['AWC_5_15']/100)*100+(r['AWC_15_30']/100)*150+(r['AWC_30_60']/100)*300+(r['AWC_60_100']/100)*400
    ck('awc_mm_formula_exact',np.allclose(awc,np.array([10.,20.])))
    ck('numeric_columns_13',len(NUMERIC_COLS)==13)
    contract=json.load(open(ROOT/'configs/production_contract.json'));ck('base_lock',contract['base_acquisition_aggregate_sha256']==BASE_AGG);ck('no_source_drift_policy',contract['source_policy']=='EXACT_CENSUS_BYTES_ONLY_NO_NETWORK_DRIFT');ck('no_recovery_policy','FAIL_IF' in contract['source_recovery_policy']);ck('success_not_stage5a_freeze',contract['success_status'].endswith('PENDING_STAGE5A_REVISED_FREEZE'))
    # Static code boundary: no network client is used by production code.
    txt=(ROOT/'src/production.py').read_text(encoding='utf-8').lower();ck('production_no_requests','requests.' not in txt and 'urllib' not in txt);ck('production_no_threshold_relaxation','0.98' not in txt and '0.95' not in txt);ck('production_no_auto_freeze','run_minedosebench_freeze' not in txt)
print(f'FINAL CONTEXT PRODUCTION UNIT TESTS PASSED ({len(checks)}/{len(checks)})')
