from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd
from .common import require,safe_extract_zip
from minedosebench.splits import largest_component,spatial_axis_roles

YEARS=(2023,2024,2025)

def _extract_inputs(root,work):
    f=work/'frozen_inputs';f.mkdir(parents=True,exist_ok=True)
    specs=[
      ('G3_STAGE1_MIN_AUDIT.zip','stage1'),
      ('G3_STAGE2_BLOCKS.zip','stage2a'),
      ('GeoDose_Stage2B_OUTPUTS_FINAL.zip','stage2b'),
      ('GeoDose_Stage5A_EXTERNAL_CONTEXT_FINAL_OUTPUTS.zip','context'),
      ('GeoDose_Stage3F_PILOT_S1_S10_20REP_THRESHOLD_FREEZE_OUTPUTS.zip','stage3f')]
    for fn,sub in specs:
        d=f/sub; marker=d/'.complete'
        if not marker.exists():
            if d.exists():
                import shutil;shutil.rmtree(d)
            safe_extract_zip(root/'inputs'/fn,d);marker.write_text('ok\n')
    return f

def _pivot_stage2b(df):
    fields=['pv_median','ue_median','water_fraction','unclear_fraction','valid_observation_count','valid_pixel_fraction',
            'block_year_eligible','all_years_eligible','high_ue_fraction_20','high_ue_fraction_25','high_ue_fraction_30']
    w=df.pivot(index=['block_id','MineID','MineN'],columns='year',values=fields)
    w.columns=[f'{a}_{int(y)}' for a,y in w.columns]
    return w.reset_index()

def _assign_real_roles(b,edges):
    out=[];axes=[]
    for mid,g in b.groupby('MineID',sort=True):
        ge=g[g.real_baseline_eligible.astype(bool)].copy()
        ee=edges[edges.MineID.astype(str)==str(mid)].copy()
        require(len(ge)>0,f'No eligible rows for {mid}')
        mask,_,_=largest_component(ge,ee)
        ge=ge.reset_index(drop=True);ge['primary_component']=mask
        sec=ge.loc[~mask].copy();sec['benchmark_role']='secondary_support_stratum';sec['spatial_axis_score']=np.nan;sec['spatial_axis_rank']=np.nan
        main=ge.loc[mask].copy()
        main2,axis=spatial_axis_roles(main,ee);main2['primary_component']=True
        axes.append((mid,float(axis[0]),float(axis[1])))
        out.extend([main2,sec])
    active=pd.concat(out,ignore_index=True)
    cols=['block_id','primary_component','benchmark_role','spatial_axis_score','spatial_axis_rank']
    full=b.drop(columns=[c for c in cols[1:] if c in b],errors='ignore').merge(active[cols],on='block_id',how='left',validate='1:1')
    full['primary_component']=full['primary_component'].eq(True)
    full['benchmark_role']=full['benchmark_role'].fillna('baseline_ineligible')
    return full,pd.DataFrame(axes,columns=['MineID','axis_x','axis_y'])

def build_90m(root:Path,work:Path,contract):
    f=_extract_inputs(root,work)
    blocks=gpd.read_file(f/'stage2a/G3_STAGE2_BLOCKS/mine_blocks_90m.gpkg',layer='blocks_90m')
    edges=pd.read_csv(f/'stage2a/G3_STAGE2_BLOCKS/block_graph_edges.csv')
    p=pd.read_csv(f/'stage2b/dea_block_year_prescreen.csv.gz')
    s=pd.read_csv(f/'stage2b/selected_three_mines.csv')
    ids=[str(x['MineID']) for x in contract['selected_mines']]
    # The frozen Stage2B selection must match the contract and must be outcome-blind.
    ss=s[s.selected_for_real_demonstration.astype(bool)].sort_values('selection_order')
    require(ss.MineID.astype(str).tolist()==ids,'Selected-mine authority mismatch')
    cfg=__import__('json').loads((f/'stage2b/stage2b_config.json').read_text())
    require(cfg['selection_uses_pv_outcome'] is False,'Stage2B selection used outcome')
    require(float(contract['quality_sensitivity']['strict_high_ue_threshold']) in [float(x) for x in cfg['high_ue_sensitivity_thresholds']],'UE20 sensitivity not frozen upstream')
    b=blocks[blocks.MineID.astype(str).isin(ids)].copy()
    w=_pivot_stage2b(p[p.MineID.astype(str).isin(ids)].copy())
    ctx=pd.read_csv(f/'context/mine_context_covariates.csv')
    exc=pd.read_csv(f/'context/FROZEN_CONTEXT_SUPPORT_EXCLUSIONS.csv')
    b=b.merge(w,on=['block_id','MineID','MineN'],how='left',validate='1:1')
    b=b.merge(ctx,on='block_id',how='left',validate='1:1',indicator='_ctx')
    b['context_complete']=b['_ctx'].eq('both');b.drop(columns='_ctx',inplace=True)
    b['frozen_context_support_excluded']=b.block_id.astype(str).isin(set(exc.block_id.astype(str)))
    # Public real-data eligibility is the already frozen three-year DEA quality rule plus frozen context completeness.
    elig=b['context_complete'].astype(bool)
    for y in YEARS:
        elig &= b[f'block_year_eligible_{y}'].eq(True)
        elig &= b[f'pv_median_{y}'].notna() & b[f'ue_median_{y}'].notna()
    b['real_baseline_eligible']=elig.astype(bool)
    # Outcome and strictly pretreatment predictors. No 2025 quantity enters X.
    b['pv_fraction_2023']=b.pv_median_2023/100.0;b['pv_fraction_2024']=b.pv_median_2024/100.0
    b['pv_change_2023_2024']=(b.pv_median_2024-b.pv_median_2023)/100.0
    b['delta_pv_2024_2025_fraction']=(b.pv_median_2025-b.pv_median_2024)/100.0
    ang=np.deg2rad(b.terrain_aspect_deg.astype(float));b['terrain_aspect_sin']=np.sin(ang);b['terrain_aspect_cos']=np.cos(ang)
    for key,name in [(contract['selected_mines'][0]['MineID'],'mine_is_mtarthur'),(contract['selected_mines'][1]['MineID'],'mine_is_hvo'),(contract['selected_mines'][2]['MineID'],'mine_is_bulga')]:
        b[name]=(b.MineID.astype(str)==str(key)).astype(float)
    thr=float(contract['quality_sensitivity']['strict_high_ue_threshold']);mx=float(contract['quality_sensitivity']['max_high_ue_fraction']);mw=float(contract['quality_sensitivity']['max_water_fraction'])
    b['strict_quality_ue20']=True
    for y in YEARS:
        b['strict_quality_ue20'] &= b[f'high_ue_fraction_{int(thr)}_{y}'].le(mx) & b[f'water_fraction_{y}'].le(mw)
    # Primary inference predictors must be complete for eligible rows; fail rather than impute.
    X=b.loc[b.real_baseline_eligible,contract['predictors']].to_numpy(float)
    require(np.isfinite(X).all(),'Eligible 90m predictor matrix contains missing/nonfinite values')
    require(b.loc[b.real_baseline_eligible,'delta_pv_2024_2025_fraction'].between(-1,1).all(),'Observed PV change outside physical domain')
    full,axis=_assign_real_roles(b,edges)
    return gpd.GeoDataFrame(full,geometry='geometry',crs=blocks.crs),edges,axis,ss

def build_180m(s90:gpd.GeoDataFrame,contract):
    x=s90.copy();x['grid_row180']=x.grid_row.astype(int)//2;x['grid_col180']=x.grid_col.astype(int)//2
    key=['MineID','MineN','grid_row180','grid_col180']
    # Aggregate all quantities needed downstream; invalid cells retain NaN model quantities.
    numeric=sorted(set(contract['predictors']+['delta_pv_2024_2025_fraction','mapped_rehabilitation_fraction',
        'pv_median_2023','pv_median_2024','pv_median_2025','ue_median_2023','ue_median_2024','ue_median_2025',
        'water_fraction_2023','water_fraction_2024','water_fraction_2025',
        'high_ue_fraction_20_2023','high_ue_fraction_20_2024','high_ue_fraction_20_2025']))
    recs=[]
    for k,g in x.groupby(key,sort=True):
        ww=g.mine_footprint_area_m2.to_numpy(float);rec=dict(zip(key,k));rec['block_id']=f'NSW180_{str(k[0]).strip("{}").replace("-","")[:12]}_{int(k[2])}_{int(k[3])}'
        rec['available_child_count']=int(len(g));rec['eligible_child_count']=int(g.real_baseline_eligible.astype(bool).sum())
        rec['eligible_child_fraction']=rec['eligible_child_count']/rec['available_child_count']
        rec['real_baseline_eligible']=bool(rec['eligible_child_count']==rec['available_child_count'] and rec['available_child_count']>0)
        rec['mine_footprint_area_m2']=float(ww.sum());rec['centroid_x']=float(np.average(g.centroid_x,weights=ww));rec['centroid_y']=float(np.average(g.centroid_y,weights=ww))
        rec['geometry']=g.geometry.union_all();rec['strict_quality_ue20']=bool(g.strict_quality_ue20.all()) if rec['real_baseline_eligible'] else False
        for c in numeric:
            rec[c]=float(np.average(g[c].astype(float),weights=ww)) if rec['real_baseline_eligible'] else np.nan
        recs.append(rec)
    b=gpd.GeoDataFrame(recs,geometry='geometry',crs=s90.crs)
    lookup={(r.MineID,int(r.grid_row180),int(r.grid_col180)):r.block_id for r in b.itertuples()};er=[]
    for r in b.itertuples():
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0:continue
                q=lookup.get((r.MineID,int(r.grid_row180)+dr,int(r.grid_col180)+dc))
                if q is not None and str(r.block_id)<str(q):er.append((r.MineID,r.block_id,q,'queen_8_neighbor'))
    edges=pd.DataFrame(er,columns=['MineID','source_block_id','target_block_id','contiguity'])
    X=b.loc[b.real_baseline_eligible,contract['predictors']].to_numpy(float)
    require(np.isfinite(X).all(),'Eligible 180m predictor matrix contains missing/nonfinite values')
    full,axis=_assign_real_roles(b,edges)
    return gpd.GeoDataFrame(full,geometry='geometry',crs=s90.crs),edges,axis

def validate_counts(s90,s180,contract):
    out={}
    for scale,s in [('90m',s90),('180m',s180)]:
        e=contract['expected_counts'][scale]
        got={'inventory':len(s),'eligible':int(s.real_baseline_eligible.sum()),'primary':int(s.primary_component.sum()),
             'secondary':int((s.benchmark_role=='secondary_support_stratum').sum()),
             'baseline_ineligible':int((s.benchmark_role=='baseline_ineligible').sum()),
             'test':int((s.benchmark_role=='test_target').sum())}
        require(got==e,f'{scale} count authority mismatch: {got} != {e}');out[scale]=got
    return out
