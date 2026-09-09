from pathlib import Path
import sys,hashlib,json
import numpy as np,pandas as pd,geopandas as gpd
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from geodose_stage6a.common import require,read_json,sha256_file,write_json
PASSED=[]
def ck(name,cond,detail=''):
    require(cond,f'VERIFY FAIL {name}: {detail}');PASSED.append(name);print('PASS',name)
def bcol(s):
    if s.dtype==bool:return s
    return s.astype(str).str.lower().map({'true':True,'false':False}).fillna(False).astype(bool)
def main():
    out=ROOT/'STAGE6A_NSW_REAL_OUTPUTS';ck('outputs_present',out.exists());r=read_json(out/'STAGE6A_PRODUCTION_REPORT.json')
    ck('report_status',r['status']=='STAGE6A_NSW_REAL_DEMONSTRATION_COMPLETE_PENDING_INDEPENDENT_AUDIT');ck('no_treatment',r['treatment_constructed'] is False);ck('no_snapshot_model',r['mapped_snapshot_exposure_used_in_model'] is False);ck('no_counterfactual_claim',r['counterfactual_causal_claim'] is False)
    man=read_json(out/'OUTPUT_MANIFEST.json')
    for x in man['files']:
        p=out/x['path'];ck('manifest_present::'+x['path'],p.exists());ck('manifest_size::'+x['path'],p.stat().st_size==x['bytes']);ck('manifest_sha::'+x['path'],sha256_file(p)==x['sha256'])
    agg=hashlib.sha256('\n'.join(f"{x['path']}|{x['bytes']}|{x['sha256']}" for x in man['files']).encode()).hexdigest();ck('manifest_aggregate',agg==man['aggregate_sha256'],agg)
    q=pd.read_csv(out/'REAL_NSW_QUERY_RESULTS.csv.gz');q['returned']=bcol(q.returned);q['interval_inverted']=bcol(q.interval_inverted)
    c=read_json(ROOT/'configs/stage6a_contract.json');ck('query_rows',len(q)==c['expected_query_rows'],len(q));ck('method_set',set(q.method)=={'M1','M2','M3','M4','M5','M6'});ck('query_unique',not q.duplicated(['scale','track','method','block_id']).any());ck('truth_physical',q.observed_y.between(-1,1).all())
    for m in ['M2','M4','M6']:
        z=q[q.method==m];ck(f'{m}_all_refused',not z.returned.any());ck(f'{m}_reason',(z.refusal_code=='R20_NO_AUTHENTIC_LONGITUDINAL_TREATMENT').all())
    for m in ['M1','M3','M5']:ck(f'{m}_returns',q[q.method==m].returned.any())
    m3=q[q.method=='M3'];ret=m3[m3.returned];ck('M3_no_compute_failure',not (m3.refusal_code=='R10_COMPUTATIONAL_FAILURE').any(),m3.refusal_code.value_counts().to_dict());ck('M3_p_01',ret.pvalue.between(0,1).all());ck('M3_graph_gate',(ret.graph_safe_count>=20).all())
    ck('M3_inversion_count',int(m3.interval_inverted.sum())==c['expected_m3_inverted_rows'],int(m3.interval_inverted.sum()));inv=m3[m3.interval_inverted];ck('M3_inversion_bounds',((inv['lower']>=-1)&(inv['upper']<=1)&(inv['lower']<=inv['upper'])).all())
    for m in ['M1','M5']:
        z=q[(q.method==m)&q.returned];ck(f'{m}_interval_order',(z['lower']<=z['upper']).all());ck(f'{m}_physical',((z['lower']>=-1)&(z['upper']<=1)).all())
    red=pd.read_csv(out/'REAL_NSW_REDUCTION_AUDIT_ROWS.csv.gz');ck('reduction_rows_positive',len(red)>0);ck('M4_M3_reduction',float(red.M4_M3_abs_diff.max())<=1e-10,float(red.M4_M3_abs_diff.max()))
    leak=pd.read_csv(out/'tables/SPATIAL_LEAKAGE_DIAGNOSTIC.csv');ck('leakage_rows',len(leak)==5,len(leak));ck('leakage_finite',np.isfinite(leak[['coverage','mean_width','rmse']].to_numpy(float)).all())
    app=pd.read_csv(out/'tables/METHOD_APPLICABILITY.csv');op=bcol(app.operational_in_real_nsw);ck('applicability_six',len(app)==6);ck('app_operational_exact',set(app.loc[op,'method'])=={'M1','M3','M5'})
    s90=gpd.read_file(out/'geodata/REAL_NSW_SUBSTRATE_90M.gpkg');s180=gpd.read_file(out/'geodata/REAL_NSW_SUBSTRATE_180M.gpkg')
    for scale,s in [('90m',s90),('180m',s180)]:
        e=c['expected_counts'][scale];ck(scale+'_count',len(s)==e['inventory'],len(s));ck(scale+'_eligible',int(bcol(s.real_baseline_eligible).sum())==e['eligible']);ck(scale+'_test',int((s.benchmark_role=='test_target').sum())==e['test'])
    ck('no_forbidden_predictors',not set(c['predictors'])&set(c['forbidden_predictors']));ck('no_2025_predictors',not any('2025' in x for x in c['predictors']));ck('no_snapshot_predictor','mapped_rehabilitation_fraction' not in c['predictors'])
    # No forbidden role adjacency in either saved graph.
    forbidden={frozenset(('nuisance_training','support_audit')),frozenset(('nuisance_training','calibration')),frozenset(('nuisance_training','test_target')),frozenset(('support_audit','calibration')),frozenset(('support_audit','test_target'))}
    for scale,s in [('90m',s90),('180m',s180)]:
        e=pd.read_csv(out/f'geodata/REAL_NSW_GRAPH_{scale.upper()}.csv');role=dict(zip(s.block_id.astype(str),s.benchmark_role.astype(str)));bad=sum(frozenset((role.get(str(a),''),role.get(str(b),''))) in forbidden for a,b in zip(e.source_block_id,e.target_block_id));ck('forbidden_role_edges_'+scale,bad==0,bad)
    gate=read_json(out/'TREATMENT_FEASIBILITY_GATE.json');ck('gate_no_Ait',gate['annual_A_it_constructed'] is False);ck('gate_no_snapshot_treatment',gate['mapped_rehabilitation_fraction_used_as_treatment'] is False)
    figs=list((out/'figures').glob('*.png'));pdfs=list((out/'figures').glob('*.pdf'));ck('figure_png_count',len(figs)==c['figure_count'],len(figs));ck('figure_pdf_count',len(pdfs)==c['figure_count'],len(pdfs));ck('figures_nonempty',all(p.stat().st_size>10000 for p in figs+pdfs))
    src=read_json(out/'STAGE6A_SOURCE_INVENTORY.json');ck('source_inventory_count',src['file_count']>20)
    for x in src['files']:
        p=ROOT/x['path'];ck('source_present::'+x['path'],p.exists());ck('source_sha::'+x['path'],sha256_file(p)==x['sha256'])
    source_text='\n'.join(p.read_text(encoding='utf-8',errors='ignore') for p in (ROOT/'src').rglob('*.py'));ck('no_network_requests','requests.' not in source_text and 'urllib' not in source_text and 'http://' not in source_text and 'https://' not in source_text);ck('no_auto_threshold_tuning','GridSearchCV' not in source_text and 'RandomizedSearchCV' not in source_text)
    vr={'status':'STAGE6A_VERIFIED_COMPLETE','checks_passed':len(PASSED),'checks_failed':0,'treatment_constructed':False,'counterfactual_causal_claim':False};write_json(out/'STAGE6A_VERIFICATION.json',vr)
    print(f'STAGE6A NSW REAL DEMONSTRATION VERIFIED COMPLETE ({len(PASSED)}/{len(PASSED)})')
if __name__=='__main__':main()
