from __future__ import annotations
from pathlib import Path
import json,time,traceback
import numpy as np,pandas as pd
from .common import require,read_json,write_json,atomic_json_gz,read_json_gz,deterministic_csv_gz,file_manifest,aggregate_manifest,sha256_file
from .authority import verify_authorities
from .data import BenchmarkData
from .evaluation import evaluate_task,evaluate_lomo
from .summaries import build_summaries
from .figures import build_all

EVALUATOR_PATCH_ID='stage5b-v1.0.1-s8-exact-full-calibration-frame'
OUTPUT_CLOSURE_PATCH_ID='stage5b-v1.0.2-exact-pvalue-roundoff-canonicalization'
EXACT_PVALUE_ROUNDOFF_TOL=1.0e-12
AFFECTED_LEGACY_CHECKPOINT_CASES={'MDB_S8_SMALL_CAL'}

RAW_FILES={
 'query':'STAGE5B_QUERY_RESULTS.csv.gz',
 'exact':'STAGE5B_EXACT_AUDIT_RESULTS.csv.gz',
 'inversion':'STAGE5B_FULL_INVERSION_RESULTS.csv.gz',
 'endpoint':'STAGE5B_ENDPOINT_AUDIT_RESULTS.csv.gz',
 'dose':'STAGE5B_DOSE_RESPONSE_PREDICTIONS.csv.gz',
 'fit':'STAGE5B_NUISANCE_FIT_AUDIT.csv.gz',
 'spatial':'STAGE5B_SPATIAL_DIAGNOSTICS.csv.gz',
 'lomo':'STAGE5B_LOMO_RESULTS.csv.gz',
}
SUMMARY_FILES={
 'coverage':'SUMMARY_COVERAGE.csv','local':'SUMMARY_LOCAL_COVERAGE.csv','fifth':'SUMMARY_FIFTH_PERCENTILE_COVERAGE.csv',
 'refusal':'SUMMARY_REFUSALS.csv','efficiency_raw':'SUMMARY_S4_EFFICIENCY_QUERY_LEVEL.csv','efficiency':'SUMMARY_S4_EFFICIENCY.csv',
 'matched':'SUMMARY_S4_MATCHED_COVERAGE_EFFICIENCY.csv','hero':'SUMMARY_S4_HERO.csv','s9':'SUMMARY_S9_MEASUREMENT_ERROR.csv',
 's10':'SUMMARY_S10_MAUP.csv','exact_summary':'SUMMARY_EXACT_SPARSE_AUDIT.csv','endpoint_summary':'SUMMARY_ENDPOINT_AUDIT.csv',
 'dose_metrics':'SUMMARY_DOSE_RESPONSE_RMSE.csv','lomo_summary':'SUMMARY_LOMO.csv','diagnostic':'SUMMARY_COVERAGE_LOSS_DIAGNOSTIC.csv','primary_effects':'SUMMARY_S4_M6_PAIRED_EFFECTS.csv'
}

def _write_csv(df:pd.DataFrame,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True);df.to_csv(path,index=False,lineterminator='\n',float_format='%.17g')

def _validate_checkpoint(obj,version,release_sha,case_id,rep):
    require(isinstance(obj,dict) and obj.get('checkpoint_schema')=='Stage5BTaskCheckpoint-v1','Checkpoint schema mismatch')
    require(obj.get('stage5b_version')==version,'Checkpoint Stage5B version mismatch')
    require(obj.get('stage5a_release_sha256')==release_sha,'Checkpoint Stage5A authority mismatch')
    require(obj.get('case_id')==case_id and int(obj.get('replication',-1))==int(rep),'Checkpoint case/rep mismatch')
    # Scientific contract/version is unchanged, so unaffected v1.0.0 checkpoints remain valid.
    # The broken evaluator failed before writing MDB_S8_SMALL_CAL rep1. Any affected
    # legacy checkpoint is nevertheless rejected fail-closed.
    if case_id in AFFECTED_LEGACY_CHECKPOINT_CASES:
        require(obj.get('evaluator_patch_id')==EVALUATOR_PATCH_ID,'Affected S8_SMALL_CAL checkpoint predates v1.0.1 evaluator patch; remove only that checkpoint and rerun')
    payload=obj.get('payload');require(isinstance(payload,dict),'Checkpoint payload missing')
    required={'query_rows','exact_rows','inversion_rows','endpoint_rows','dose_rows','fit_rows','spatial_rows','runtime_row','lomo_rows'}
    require(required.issubset(payload),'Checkpoint payload fields missing')
    return payload

def _task_checkpoint_path(work:Path,case_id:str,rep:int)->Path:
    return work/'checkpoints'/f'{case_id}__rep{int(rep):02d}.json.gz'

def _canonicalize_exact_audit_frame(df:pd.DataFrame,alpha:float):
    """Canonicalize only floating-point roundoff at exact-audit probability boundaries.

    Full-graph exact p-values are finite subset sums of normalized orbit probabilities.
    IEEE-754 summation can return values such as 1+2e-16. This routine refuses any
    scientifically meaningful out-of-range value, then clips only values within the
    frozen numerical tolerance and recomputes dependent exact-audit fields.
    """
    x=df.copy();require('pvalue' in x.columns,'Exact audit pvalue column missing')
    raw=pd.to_numeric(x.pvalue,errors='coerce').to_numpy(float)
    require(np.isfinite(raw).all(),'Exact audit pvalue contains NaN/Inf; not a roundoff-only closure')
    rmin=float(raw.min()) if len(raw) else np.nan;rmax=float(raw.max()) if len(raw) else np.nan
    require(rmin>=-EXACT_PVALUE_ROUNDOFF_TOL and rmax<=1.0+EXACT_PVALUE_ROUNDOFF_TOL,
            f'Exact audit pvalue exceeds roundoff-only closure tolerance: min={rmin:.17g}, max={rmax:.17g}')
    low=raw<0.0;high=raw>1.0;changed=low|high
    canon=np.clip(raw,0.0,1.0);x['pvalue']=canon;x['covered']=canon>float(alpha)
    keys=['case_id','scenario_id','replication','MineID','target_block_id','method']
    require(all(k in x.columns for k in keys),'Exact audit pairing columns missing')
    for _,idxs in x.groupby(keys,sort=False).groups.items():
        ids=list(idxs);g=x.loc[ids];refs=set(g.reference.astype(str));require(len(g)==2 and refs=={'full_graph_exact','sparse_m64'},'Exact audit full/sparse pair malformed')
        pf=float(g.loc[g.reference.astype(str)=='full_graph_exact','pvalue'].iloc[0]);ps=float(g.loc[g.reference.astype(str)=='sparse_m64','pvalue'].iloc[0]);x.loc[ids,'pvalue_absolute_difference']=abs(pf-ps)
    audit={'patch_id':OUTPUT_CLOSURE_PATCH_ID,'roundoff_tolerance':EXACT_PVALUE_ROUNDOFF_TOL,'rows':int(len(x)),
           'raw_min_pvalue':rmin,'raw_max_pvalue':rmax,'raw_below_zero_count':int(low.sum()),'raw_above_one_count':int(high.sum()),
           'canonicalized_row_count':int(changed.sum()),'maximum_absolute_correction':float(np.max(np.abs(canon-raw))) if len(raw) else 0.0,
           'scientific_contract_changed':False,'thresholds_changed':False,'coverage_changed':False}
    # p-values can only move at numerical 0/1 boundaries, so alpha=.1 membership must not change.
    audit['coverage_changed']=bool(np.any((raw>float(alpha))!=(canon>float(alpha))))
    require(not audit['coverage_changed'],'Roundoff canonicalization unexpectedly changes exact-audit coverage membership')
    return x,audit

def run(package_root:Path)->dict:
    package_root=Path(package_root).resolve(); contract=read_json(package_root/'configs/stage5b_contract.json'); version=contract['version']; expected=contract['expected_counts']
    work=package_root/'_STAGE5B_WORK';work.mkdir(parents=True,exist_ok=True)
    stage5a,auth=verify_authorities(package_root,work)
    release_sha=auth['stage5a_release_zip_sha256']
    bench=BenchmarkData(stage5a)
    cases=bench.scenarios.case_id.astype(str).tolist(); reps=range(1,21); tasks=[(c,r) for c in cases for r in reps];require(len(tasks)==int(expected['case_replication_tasks']),'Task count mismatch before production')
    print(f"[STAGE5B] Frozen tasks: {len(tasks)} = 27 cases x 20 replications")
    print("[STAGE5B] Primary MineDoseBench methods: M2, M3, M4, M5, M6")
    print("[STAGE5B] Refusals remain abstentions; no thresholds/hyperparameters are tuned from production outcomes.")
    reused=computed=0;start=time.time()
    for pos,(case_id,rep) in enumerate(tasks,1):
        cp=_task_checkpoint_path(work,case_id,rep)
        if cp.exists():
            payload=_validate_checkpoint(read_json_gz(cp),version,release_sha,case_id,rep);reused+=1
        else:
            try:
                task=bench.build_task(case_id,rep)
                payload=evaluate_task(task,contract)
                payload['lomo_rows']=evaluate_lomo(task,contract,bench.lomo)
                obj={'checkpoint_schema':'Stage5BTaskCheckpoint-v1','stage5b_version':version,'stage5a_release_sha256':release_sha,'evaluator_patch_id':EVALUATOR_PATCH_ID,'case_id':case_id,'replication':int(rep),'payload':payload}
                atomic_json_gz(cp,obj);computed+=1
            except Exception as exc:
                fail={'case_id':case_id,'replication':int(rep),'exception_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
                write_json(work/'LAST_FAILURE.json',fail)
                raise
        if pos==1 or pos%5==0 or pos==len(tasks):
            print(f"  completed {pos}/{len(tasks)}; reused={reused}, newly_computed={computed}; checkpoints are resumable")
    # Aggregate only after all 540 atomic checkpoints exist and validate.
    buckets={k:[] for k in ['query_rows','exact_rows','inversion_rows','endpoint_rows','dose_rows','fit_rows','spatial_rows','lomo_rows']};runtime=[]
    for case_id,rep in tasks:
        p=_validate_checkpoint(read_json_gz(_task_checkpoint_path(work,case_id,rep)),version,release_sha,case_id,rep)
        for k in buckets:buckets[k].extend(p[k])
        runtime.append(p['runtime_row'])
    counts={'query_rows':len(buckets['query_rows']),'exact_audit_rows':len(buckets['exact_rows']),'full_inversion_rows':len(buckets['inversion_rows']),'endpoint_audit_rows':len(buckets['endpoint_rows']),'dose_response_prediction_rows':len(buckets['dose_rows']),'nuisance_fit_rows':len(buckets['fit_rows']),'lomo_rows':len(buckets['lomo_rows']),'runtime_rows':len(runtime),'spatial_diagnostic_rows':len(buckets['spatial_rows'])}
    for k,v in expected.items():
        if k=='case_replication_tasks':continue
        require(int(counts[k])==int(v),f'Final row-count gate failed: {k}: {counts[k]} != {v}')
    out=package_root/'STAGE5B_PRODUCTION_OUTPUTS';out.mkdir(parents=True,exist_ok=True)
    frames={
      'query':pd.DataFrame(buckets['query_rows']),'exact':pd.DataFrame(buckets['exact_rows']),'inversion':pd.DataFrame(buckets['inversion_rows']),
      'endpoint':pd.DataFrame(buckets['endpoint_rows']),'dose':pd.DataFrame(buckets['dose_rows']),'fit':pd.DataFrame(buckets['fit_rows']),
      'spatial':pd.DataFrame(buckets['spatial_rows']),'lomo':pd.DataFrame(buckets['lomo_rows'])}
    frames['exact'],exact_roundoff_audit=_canonicalize_exact_audit_frame(frames['exact'],float(contract['alpha']))
    write_json(out/'STAGE5B_EXACT_PVALUE_NUMERICAL_CLOSURE.json',exact_roundoff_audit)
    for key,fn in RAW_FILES.items():deterministic_csv_gz(frames[key],out/fn)
    runtime_df=pd.DataFrame(runtime);_write_csv(runtime_df,out/'STAGE5B_RUNTIME.csv')
    summaries=build_summaries(frames['query'],frames['inversion'],frames['exact'],frames['endpoint'],frames['dose'],frames['lomo'],contract)
    for key,fn in SUMMARY_FILES.items():_write_csv(summaries[key],out/fn)
    figs=out/'figures';build_all(summaries,frames['query'],frames['exact'],figs)
    # Frozen claim boundary: empirical benchmark results must never upgrade unavailable theorem assumptions.
    claim={
      'stage':'Stage5B','version':version,'status':'production_complete_pending_independent_audit',
      'primary_methods':['M2','M3','M4','M5','M6'],'M4_status':'naive generally non-theorem-backed comparator',
      'primary_M6_route':'m64 sparse/localized graph-residual reference with frozen operational gates',
      'estimated_nuisance_finite_sample_theorem_certified':False,
      'n2_localization_metric_status':'observable m64-versus-m32 sparse-localization sensitivity diagnostic; not relabelled a full N2 theorem certificate',
      'exact_audit_status':'full-graph structural oracle G2/G3 versus sparse-m64 on frozen exact-size6 targets',
      'refusals_counted_as_noncoverage':False,'candidate_domain':contract['candidate_domain'],
      'ST2_status':'structural/cross-sectional temporal stress only','ST3_status':'C2 causal-identification negative control; no positive causal validity claim',
      'full_real_line_claim':False,'width_superiority_without_matched_coverage':False,
      'external_context_substrate':'23,618 retained / 92 prospectively excluded; unchanged from Stage5A',
      'stage5a_release_sha256':release_sha
    };write_json(out/'STAGE5B_CLAIM_BOUNDARY.json',claim)
    # Source/provenance inventory for the evaluator package excluding generated work/output/cache.
    inv=[]
    # Record all immutable source/config/docs explicitly.
    for top in ['src','vendor','configs','authorities','docs']:
        base=package_root/top
        if base.exists():
            for p in sorted((x for x in base.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc'),key=lambda x:x.relative_to(package_root).as_posix()):
                inv.append({'file':p.relative_to(package_root).as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    for name in ['stage5b_run.py','verify_stage5b.py','package_stage5b_release.py','verify_runtime.py','requirements_py310.txt']:
        p=package_root/name
        if p.exists():inv.append({'file':name,'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    write_json(out/'STAGE5B_SOURCE_INVENTORY.json',{'files':inv,'aggregate_sha256':aggregate_manifest(inv)})
    elapsed=time.time()-start
    report={'status':'STAGE5B_PRODUCTION_COMPLETE_PENDING_INDEPENDENT_AUDIT','stage':'Stage5B','version':version,'stage5a_release_sha256':release_sha,'stage5a_audit_sha256':auth['stage5a_audit_sha256'],'tasks':len(tasks),'counts':counts,'support_thresholds':contract['support_thresholds'],'candidate_domain':contract['candidate_domain'],'primary_methods':contract['primary_methods'],'primary_predictor':'RF','confirmation_predictor':'XGBoost in frozen S1/S4/S5/S8 scope','oracle_track':'diagnostic only','checkpoint_reused_this_run':reused,'checkpoint_newly_computed_this_run':computed,'wall_seconds_this_run':elapsed,'thresholds_retuned':False,'hyperparameters_retuned_from_production':False,'stage5a_modified':False,'refusals_counted_as_noncoverage':False,'M2_M6_performance_now_evaluated':True,'output_numerical_closure_patch_id':OUTPUT_CLOSURE_PATCH_ID,'exact_pvalue_roundoff_canonicalized_rows':int(exact_roundoff_audit['canonicalized_row_count']),'next_action':'Run independent Stage5B archive/statistical audit before any NSW real-demonstration stage.'}
    write_json(out/'STAGE5B_PRODUCTION_REPORT.json',report)
    # Scientific manifest excludes wall-clock/runtime/procedural report so identical exact-runtime reruns can
    # reconcile the scientific evidence independently of machine speed.
    sci_ex={'STAGE5B_RUNTIME.csv','STAGE5B_PRODUCTION_REPORT.json','SCIENTIFIC_OUTPUT_MANIFEST.json','OUTPUT_FILE_MANIFEST.json','STAGE5B_VERIFICATION.json','GeoDose_Stage5B_MineDoseBench_PRODUCTION_RESULTS.zip','STAGE5B_RELEASE_MANIFEST.json'}
    scientific=file_manifest(out,exclude=sci_ex);write_json(out/'SCIENTIFIC_OUTPUT_MANIFEST.json',{'files':scientific,'aggregate_sha256':aggregate_manifest(scientific)})
    # Full output manifest excludes itself and terminal verifier/release artifacts.
    manifest=file_manifest(out,exclude={'OUTPUT_FILE_MANIFEST.json','STAGE5B_VERIFICATION.json','GeoDose_Stage5B_MineDoseBench_PRODUCTION_RESULTS.zip','STAGE5B_RELEASE_MANIFEST.json'})
    write_json(out/'OUTPUT_FILE_MANIFEST.json',{'files':manifest,'aggregate_sha256':aggregate_manifest(manifest)})
    return report
