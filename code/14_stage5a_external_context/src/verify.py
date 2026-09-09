from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np,pandas as pd
from .common import require,read_json,write_json,sha256_file,deterministic_zip,utc_now
from .authority import load_census,verify_base,CENSUS_AGG,CENSUS_ZIP_SHA,BASE_AGG,THRESHOLD,GOVERNING
from .production import NUMERIC_COLS

def verify(package_root: Path, base: Path, out: Path):
    package_root=Path(package_root);base=Path(base);out=Path(out);verify_base(base);census=load_census(package_root,out/'_verify_census_work');checks=[]
    def ck(name,ok,detail=''):
        require(ok,f'{name}: {detail}');checks.append({'check':name,'pass':True,'detail':str(detail)})
    required=['mine_context_covariates.csv','context_metadata.json','FROZEN_CONTEXT_COMPLETE_SUBSTRATE.csv','FROZEN_CONTEXT_SUPPORT_EXCLUSIONS.csv','CONTEXT_ATTRITION_AUDIT.json','CONTEXT_ATTRITION_BY_MINE.csv','CONTEXT_ATTRITION_FAILURE_PATTERNS.csv','CONTEXT_ATTRITION_PREOUTCOME_SPATIAL_AUDIT.csv','CONTEXT_RANGE_QA.csv','CONTEXT_RANGE_SUMMARY_BY_MINE.csv','CONTEXT_VALID_AREA_QA.csv.gz','CONTEXT_SUPPORT_RECONCILIATION.csv','SOIL_DEPTH_COMPONENT_AUDIT.csv.gz','CONTEXT_DERIVATION_FORMULA_AUDIT.csv','SOURCE_PROVENANCE_MANIFEST.json','FROZEN_SOURCE_LAYER_SUPPORT_SUMMARY.csv','GeoDose_Stage5A_CONTEXT_SOURCE_SNAPSHOT.zip','CONTEXT_PRODUCTION_REPORT.json']
    ck('required_files',all((out/x).exists() for x in required),[x for x in required if not(out/x).exists()])
    ctx=pd.read_csv(out/'mine_context_covariates.csv',dtype={'block_id':str});fr=pd.read_csv(out/'FROZEN_CONTEXT_COMPLETE_SUBSTRATE.csv',dtype={'block_id':str});ex=pd.read_csv(out/'FROZEN_CONTEXT_SUPPORT_EXCLUSIONS.csv',dtype={'block_id':str});meta=read_json(out/'context_metadata.json');rep=read_json(out/'CONTEXT_PRODUCTION_REPORT.json')
    cid=set(census['complete'].block_id.astype(str));eid=set(census['excluded'].block_id.astype(str))
    ck('context_rows_23618',len(ctx)==23618,len(ctx));ck('context_unique_ids',ctx.block_id.is_unique);ck('context_exact_frozen_ids',set(ctx.block_id)==cid);ck('excluded_none_in_context',not(set(ctx.block_id)&eid));ck('frozen_registry_exact',len(fr)==23618 and set(fr.block_id)==cid);ck('exclusion_registry_exact',len(ex)==92 and set(ex.block_id)==eid);ck('partition_exact',not(cid&eid) and len(cid|eid)==23710)
    ck('context_columns_exact',list(ctx.columns)==['block_id']+NUMERIC_COLS,list(ctx.columns));arr=ctx[NUMERIC_COLS].apply(pd.to_numeric,errors='coerce').to_numpy(float);ck('all_context_finite',np.isfinite(arr).all())
    schema=read_json(base/'configs/context_schema.json')
    for c,(lo,hi) in schema['physical_ranges'].items():x=ctx[c].to_numpy(float);ck('range_'+c,((x>=lo)&(x<=hi)).all(),f'{x.min()}..{x.max()}')
    va=pd.read_csv(out/'CONTEXT_VALID_AREA_QA.csv.gz',dtype={'block_id':str});vcols=[f'{x}__valid_area_fraction' for x in GOVERNING];ck('valid_area_rows',len(va)==23618);ck('valid_area_exact_ids',set(va.block_id)==cid);ck('valid_area_18_layers',all(c in va for c in vcols));ck('valid_area_threshold_exact',va[vcols].min().min()>=THRESHOLD,va[vcols].min().min())
    rc=pd.read_csv(out/'CONTEXT_SUPPORT_RECONCILIATION.csv');ck('reconciliation_18',len(rc)==18 and set(rc.governing_layer)==set(GOVERNING));ck('reconciliation_pass',rc['pass'].astype(bool).all());ck('reconciliation_tight',rc.max_abs_valid_area_fraction_difference.max()<=5e-12,rc.max_abs_valid_area_fraction_difference.max())
    raw=pd.read_csv(out/'SOIL_DEPTH_COMPONENT_AUDIT.csv.gz',dtype={'block_id':str});ck('raw_soil_rows',len(raw)==23618 and set(raw.block_id)==cid);fa=pd.read_csv(out/'CONTEXT_DERIVATION_FORMULA_AUDIT.csv');ck('formula_five',len(fa)==5);ck('formula_pass',fa['pass'].astype(bool).all() and fa.max_abs_recompute_residual.max()<=1e-12,fa.max_abs_recompute_residual.max())
    att=read_json(out/'CONTEXT_ATTRITION_AUDIT.json');ck('attrition_counts',att['original_required_blocks']==23710 and att['retained_context_complete_blocks']==23618 and att['excluded_support_gap_blocks']==92);bm=pd.read_csv(out/'CONTEXT_ATTRITION_BY_MINE.csv');ck('attrition_by_mine_total',bm.original_required_blocks.sum()==23710 and bm.retained_context_complete_blocks.sum()==23618 and bm.excluded_support_gap_blocks.sum()==92)
    # Expected prospective support pattern from the independently accepted census: 89 Liddell + 3 Bulga, zero elsewhere.
    got=dict(zip(bm.MineN,bm.excluded_support_gap_blocks.astype(int)));ck('attrition_liddell_89',got.get('Liddell Coal')==89,got);ck('attrition_bulga_3',got.get('Bulga Complex')==3,got);ck('attrition_other_mines_zero',sum(v for k,v in got.items() if k not in {'Liddell Coal','Bulga Complex'})==0,got)
    ck('metadata_complete',meta.get('scientific_context_complete') is True);ck('metadata_area_weighted',meta.get('spatial_aggregation_contract')=='AREA_WEIGHTED_BLOCK_POLYGON' and all(meta.get(s,{}).get('spatial_aggregation')=='AREA_WEIGHTED_BLOCK_POLYGON' for s in ['climate','soil','terrain']));cs=meta.get('context_substrate',{});ck('metadata_substrate_counts',cs.get('original_required_blocks')==23710 and cs.get('frozen_context_complete_blocks')==23618 and cs.get('prospectively_excluded_for_external_support')==92);ck('metadata_threshold_unchanged',cs.get('frozen_valid_area_threshold')==0.99 and cs.get('threshold_changed') is False);ck('metadata_no_imputation',cs.get('imputation_performed') is False)
    prov=read_json(out/'SOURCE_PROVENANCE_MANIFEST.json');ck('provenance_base',prov.get('base_acquisition_aggregate_sha256')==BASE_AGG);ck('provenance_census',prov.get('support_census_zip_sha256')==CENSUS_ZIP_SHA and prov.get('support_census_output_aggregate_sha256')==CENSUS_AGG);ck('provenance_no_credentials',prov.get('credential_values_recorded') is False);ck('provenance_locked_bytes',prov.get('local_source_hash_validation_passed') is True and prov.get('source_policy')=='EXACT_CENSUS_BYTES_ONLY_NO_NETWORK_DRIFT')
    ck('claim_boundary',rep.get('benchmark_outcomes_read') is False and rep.get('M2_M6_results_read') is False and rep.get('thresholds_retuned') is False and rep.get('imputation_performed') is False and rep.get('unsupported_blocks_recovered') is False and rep.get('automatic_minedosebench_freeze') is False)
    # No outcome/method artifact names in final context folder.
    names=[p.name.lower() for p in out.iterdir() if p.is_file()];forbidden=['oracle_truth','method_results','m2_','m3_','m4_','m5_','m6_','validation_public_units'];ck('no_outcome_method_artifacts',not any(any(f in n for f in forbidden) for n in names),names)
    ready=out/'READY_FOR_STAGE5A_REVISED_FREEZE';ck('handoff_exists',ready.exists() and all((ready/x).exists() for x in ['mine_context_covariates.csv','context_metadata.json','FROZEN_CONTEXT_COMPLETE_SUBSTRATE.csv','FROZEN_CONTEXT_SUPPORT_EXCLUSIONS.csv','IMPORTANT_NEXT_STEP.txt']))
    # Compact source snapshot has its own manifest.
    import zipfile, tempfile
    snap=out/'GeoDose_Stage5A_CONTEXT_SOURCE_SNAPSHOT.zip';ck('source_snapshot_nonempty',snap.stat().st_size>0)
    with zipfile.ZipFile(snap) as z:ck('snapshot_manifest_present','SNAPSHOT_MANIFEST.json' in z.namelist())
    status={'status':'FINAL_CONTEXT_PRODUCTION_PASSED_PENDING_STAGE5A_REVISED_FREEZE','stage':'GeoDose-CP Stage5A External Context Production','version':'1.0.1-stage5a-external-context-production-freeze-candidate','checks_passed':len(checks),'checks_failed':0,'original_required_blocks':23710,'frozen_context_complete_blocks':23618,'frozen_support_exclusions':92,'frozen_valid_area_threshold':0.99,'benchmark_outcomes_read':False,'M2_M6_results_read':False,'thresholds_retuned':False,'imputation_performed':False,'automatic_minedosebench_freeze':False,'verified_utc':utc_now(),'next_action':'Independent audit of this final context production. Then create/use a prospectively revised MineDoseBench Stage5A freeze package that accepts the frozen 23,618-block substrate. Do not run unchanged MineDoseBench v1.1.0.'};write_json(out/'CONTEXT_FINAL_VERIFICATION.json',status);write_json(out/'CONTEXT_FINAL_VERIFICATION_CHECKS.json',checks)
    # Build final user-facing immutable bundle after verification. Avoid self-hash recursion.
    exclude={'GeoDose_Stage5A_EXTERNAL_CONTEXT_FINAL_OUTPUTS.zip','CONTEXT_OUTPUT_BUNDLE_IDENTITY.json','_census_authority_work','_verify_census_work'}
    rows=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(out).as_posix()
        if rel.split('/')[0] in {'_census_authority_work','_verify_census_work'} or p.name in {'GeoDose_Stage5A_EXTERNAL_CONTEXT_FINAL_OUTPUTS.zip','CONTEXT_OUTPUT_BUNDLE_IDENTITY.json','OUTPUT_FILE_MANIFEST.json'}:continue
        rows.append({'file':rel,'size':p.stat().st_size,'sha256':sha256_file(p)})
    import hashlib
    agg=hashlib.sha256(''.join(r['file']+r['sha256'] for r in sorted(rows,key=lambda r:r['file'])).encode()).hexdigest();write_json(out/'OUTPUT_FILE_MANIFEST.json',{'files':rows,'file_count':len(rows),'aggregate_sha256':agg})
    include=[r['file'] for r in rows]+['OUTPUT_FILE_MANIFEST.json'];z=out/'GeoDose_Stage5A_EXTERNAL_CONTEXT_FINAL_OUTPUTS.zip';deterministic_zip(out,z,include);write_json(out/'CONTEXT_OUTPUT_BUNDLE_IDENTITY.json',{'outputs_zip_sha256':sha256_file(z),'outputs_zip_size':z.stat().st_size,'recorded_utc':utc_now(),'output_manifest_aggregate_sha256':agg})
    return status,checks
