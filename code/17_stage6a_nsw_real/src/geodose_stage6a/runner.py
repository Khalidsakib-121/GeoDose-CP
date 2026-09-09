from __future__ import annotations
from pathlib import Path
import hashlib,shutil,time
import numpy as np,pandas as pd,geopandas as gpd
from .common import require,read_json,write_json,sha256_file
from .authority import verify_authorities
from .data import build_90m,build_180m,validate_counts
from .evaluation import fit_model_and_spatial,evaluate_mine,leakage_diagnostic
from .summaries import method_applicability,sample_summary,interval_summary,refusal_summary,quality_sensitivity,maup_summary,reduction_summary
from .figures import make_all

VERSION='1.0.0-stage6a-nsw-real-demonstration-freeze-candidate'

def _checkpoint_sig(root,scale,track,mid):
    scientific=[
      root/'configs/stage6a_contract.json',
      root/'src/geodose_stage6a/data.py',root/'src/geodose_stage6a/models.py',root/'src/geodose_stage6a/intervals.py',root/'src/geodose_stage6a/evaluation.py',
      root/'vendor/geodose_stage5b/spatial.py',root/'vendor/geodose_stage5b/orbit.py',root/'vendor/minedosebench/splits.py']
    parts=[VERSION]+[sha256_file(p) for p in scientific]+[scale,track,str(mid)]
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()

def _read_checkpoint(cpdir,sig):
    meta=cpdir/'meta.json';qf=cpdir/'query.csv.gz';rf=cpdir/'reduction.csv.gz'
    if not(meta.exists() and qf.exists() and rf.exists()):return None
    m=read_json(meta)
    if m.get('signature')!=sig:return None
    if sha256_file(qf)!=m.get('query_sha256') or sha256_file(rf)!=m.get('reduction_sha256'):return None
    return pd.read_csv(qf),pd.read_csv(rf),m['runtime']
def _write_checkpoint(cpdir,sig,q,r,runtime):
    cpdir.mkdir(parents=True,exist_ok=True)
    tq=cpdir/'query.tmp.csv.gz';tr=cpdir/'reduction.tmp.csv.gz';q.to_csv(tq,index=False,compression='gzip');r.to_csv(tr,index=False,compression='gzip')
    qf=cpdir/'query.csv.gz';rf=cpdir/'reduction.csv.gz';tq.replace(qf);tr.replace(rf)
    write_json(cpdir/'meta.json',{'signature':sig,'query_sha256':sha256_file(qf),'reduction_sha256':sha256_file(rf),'runtime':runtime})

def _source_inventory(root):
    files=[]
    for top in ['src','vendor','configs','authorities','docs']:
        for p in sorted((root/top).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':files.append(p)
    for fn in ['stage6a_run.py','verify_stage6a.py','package_stage6a_release.py','verify_package.py','verify_runtime.py','requirements_py310.txt']:
        p=root/fn
        if p.exists():files.append(p)
    rows=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in files]
    agg=hashlib.sha256('\n'.join(f"{r['path']}|{r['bytes']}|{r['sha256']}" for r in rows).encode()).hexdigest()
    return {'files':rows,'file_count':len(rows),'aggregate_sha256':agg}
def _manifest(out):
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in {'OUTPUT_MANIFEST.json','GeoDose_Stage6A_NSW_REAL_DEMONSTRATION_RESULTS.zip','STAGE6A_RELEASE_MANIFEST.json'}:
            files.append({'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    agg=hashlib.sha256('\n'.join(f"{r['path']}|{r['bytes']}|{r['sha256']}" for r in files).encode()).hexdigest()
    return {'files':files,'file_count':len(files),'aggregate_sha256':agg}

def _merge_map_results(s90,q):
    z=q[(q.scale=='90m')&(q.track=='RF_PRIMARY')&q.method.isin(['M1','M3','M5'])].copy()
    g=s90[s90.benchmark_role=='test_target'].copy()
    for m in ['M1','M3','M5']:
        cols=['block_id','returned','refusal_code','covered_observed_product','lower','upper','width','pvalue','graph_safe_count','n3_diagnostic_lower_bound','delta_sparse_pinsker_tv']
        a=z[z.method==m][cols].copy();a.columns=['block_id']+[f'{m.lower()}_{c}' for c in cols[1:]]
        g=g.merge(a,on='block_id',how='left',validate='1:1')
    return gpd.GeoDataFrame(g,geometry='geometry',crs=s90.crs)

def run(root:Path):
    t0=time.time();root=Path(root);contract=read_json(root/'configs/stage6a_contract.json');require(contract['version']==VERSION,'Contract version mismatch')
    authority_checks=verify_authorities(root,contract);work=root/'_STAGE6A_WORK';work.mkdir(exist_ok=True);out=root/'STAGE6A_NSW_REAL_OUTPUTS'
    print('[1/7] Frozen authorities verified; Stage1 treatment infeasibility is enforced.',flush=True)
    print('[2/7] Building frozen 90 m / anchored 180 m NSW real-data substrates without outcome-dependent selection...',flush=True)
    s90,e90,axis90,sel=build_90m(root,work,contract);s90=s90.reset_index(drop=True);s180,e180,axis180=build_180m(s90,contract);s180=s180.reset_index(drop=True);count_checks=validate_counts(s90,s180,contract)
    require('mapped_rehabilitation_fraction' not in contract['predictors'],'Snapshot exposure predictor leakage');require(not any('2025' in c for c in contract['predictors']),'2025 predictor leakage')
    if out.exists():shutil.rmtree(out)
    (out/'figures').mkdir(parents=True);(out/'tables').mkdir();(out/'geodata').mkdir()
    s90.to_file(out/'geodata/REAL_NSW_SUBSTRATE_90M.gpkg',layer='real_nsw_90m',driver='GPKG');s180.to_file(out/'geodata/REAL_NSW_SUBSTRATE_180M.gpkg',layer='real_nsw_180m',driver='GPKG')
    e90.to_csv(out/'geodata/REAL_NSW_GRAPH_90M.csv',index=False);e180.to_csv(out/'geodata/REAL_NSW_GRAPH_180M.csv',index=False);axis90.to_csv(out/'tables/SPATIAL_AXIS_90M.csv',index=False);axis180.to_csv(out/'tables/SPATIAL_AXIS_180M.csv',index=False)
    sample_summary(s90,'90m').to_csv(out/'tables/SAMPLE_SUMMARY_90M.csv',index=False);sample_summary(s180,'180m').to_csv(out/'tables/SAMPLE_SUMMARY_180M.csv',index=False);method_applicability(contract).to_csv(out/'tables/METHOD_APPLICABILITY.csv',index=False)
    print('[3/7] Fitting frozen RF/XGBoost predictors and spatial nuisance diagnostics; running/resuming 12 mine-scale-track tasks...',flush=True)
    allq=[];allred=[];fitrows=[];imps=[];runtimes=[];reused=new=0
    for scale,(units,edges) in {'90m':(s90,e90),'180m':(s180,e180)}.items():
        for track in contract['predictor_tracks']:
            model,u,pred,resid,g,fit,imp=fit_model_and_spatial(units,edges,track,scale,contract);fitrows.append(fit);imps.append(imp)
            for mine in contract['selected_mines']:
                mid=mine['MineID'];tag=mid.strip('{}').split('-')[0];cp=work/'checkpoints'/f'{scale}_{track}_{tag}';sig=_checkpoint_sig(root,scale,track,mid);cached=_read_checkpoint(cp,sig)
                if cached is None:
                    q,r,rt=evaluate_mine(u,edges,track,scale,mid,contract,model,pred,resid,g,fit['spatial_by_mine']);_write_checkpoint(cp,sig,q,r,rt);new+=1
                else:q,r,rt=cached;reused+=1
                allq.append(q);allred.append(r);runtimes.append({**rt,'checkpoint_reused':cached is not None})
                print(f"  {scale} {track} {mine['MineN']}: {'reused' if cached is not None else 'computed'}; test={rt['test_targets']}; inversion={rt['inversion_sample_count']}",flush=True)
    q=pd.concat(allq,ignore_index=True);red=pd.concat(allred,ignore_index=True) if allred else pd.DataFrame()
    q.to_csv(out/'REAL_NSW_QUERY_RESULTS.csv.gz',index=False,compression='gzip');red.to_csv(out/'REAL_NSW_REDUCTION_AUDIT_ROWS.csv.gz',index=False,compression='gzip');pd.DataFrame(runtimes).to_csv(out/'tables/TASK_RUNTIME.csv',index=False)
    write_json(out/'MODEL_SPATIAL_FIT_AUDIT.json',fitrows);pd.concat(imps,ignore_index=True).to_csv(out/'tables/FEATURE_IMPORTANCE.csv',index=False)
    print('[4/7] Running compact random-vs-buffered-vs-LOMO leakage diagnostic and real-product sensitivity summaries...',flush=True)
    leak=leakage_diagnostic(s90,contract);leak.to_csv(out/'tables/SPATIAL_LEAKAGE_DIAGNOSTIC.csv',index=False)
    mine_sum,pooled=interval_summary(q);mine_sum.to_csv(out/'tables/INTERVAL_SUMMARY_BY_MINE.csv',index=False);pooled.to_csv(out/'tables/INTERVAL_SUMMARY_POOLED.csv',index=False)
    refusal_summary(q).to_csv(out/'tables/REFUSAL_SUMMARY.csv',index=False);qs=quality_sensitivity(q);qs.to_csv(out/'tables/PRODUCT_QUALITY_SENSITIVITY.csv',index=False);maup=maup_summary(q);maup.to_csv(out/'tables/MAUP_90M_180M.csv',index=False);reduction_summary(red).to_csv(out/'tables/REDUCTION_AUDIT_SUMMARY.csv',index=False)
    exp=s90[s90.real_baseline_eligible.astype(bool)].groupby(['MineID','MineN'],sort=True).mapped_rehabilitation_fraction.agg(['count','mean','median','min','max']).reset_index()
    exp['interpretation']='descriptive 2026 snapshot exposure only; not treatment and not predictor';exp.to_csv(out/'tables/SNAPSHOT_EXPOSURE_DESCRIPTIVE_SUMMARY.csv',index=False)
    _merge_map_results(s90,q).to_file(out/'geodata/REAL_NSW_APPLIED_INTERVAL_WARNING_MAP_90M_RF.gpkg',layer='real_nsw_test_targets',driver='GPKG')
    write_json(out/'REAL_NSW_CLAIM_BOUNDARY.json',contract['real_application_claim_boundary'])
    write_json(out/'TREATMENT_FEASIBILITY_GATE.json',{'status':'NOT_CONSTRUCTIBLE_FROM_FROZEN_PUBLIC_SNAPSHOT','authority':'STAGE1_CLEANING_DECISION.md','annual_A_it_constructed':False,
        'mapped_rehabilitation_fraction_used_as_treatment':False,'mapped_rehabilitation_fraction_used_as_predictor':False,'M2_operational':False,'M4_operational':False,'M6_operational':False,
        'real_demo_interpretation':contract['real_application_claim_boundary']['real_intervals_interpretation']})
    print('[5/7] Rendering seven publication-grade PNG/PDF figure pairs...',flush=True);make_all(s90,q,pooled,maup,leak,qs,out/'figures')
    write_json(out/'STAGE6A_SOURCE_INVENTORY.json',_source_inventory(root))
    report={'status':'STAGE6A_NSW_REAL_DEMONSTRATION_COMPLETE_PENDING_INDEPENDENT_AUDIT','stage':'Stage6A','version':VERSION,
      'selected_mines':[m['MineN'] for m in contract['selected_mines']],'authorities_verified':len(authority_checks),'count_checks':count_checks,'query_rows':len(q),'reduction_audit_rows':len(red),
      'task_count':12,'checkpoint_reused_this_run':reused,'checkpoint_newly_computed_this_run':new,'figures_png':len(list((out/'figures').glob('*.png'))),'figures_pdf':len(list((out/'figures').glob('*.pdf'))),
      'treatment_constructed':False,'mapped_snapshot_exposure_used_in_model':False,'counterfactual_causal_claim':False,'M2_M4_M6_operational_real_routes':False,'primary_real_methods':['M1','M3','M5'],
      'outcome':contract['outcome'],'support_thresholds':contract['support_thresholds'],'runtime_seconds':time.time()-t0,'next_action':'Independent Stage6A archive/scientific audit before manuscript consolidation.'}
    write_json(out/'STAGE6A_PRODUCTION_REPORT.json',report);write_json(out/'OUTPUT_MANIFEST.json',_manifest(out))
    print('[6/7] Stage6A scientific outputs generated. Run terminal verifier before release packaging.',flush=True);return report
