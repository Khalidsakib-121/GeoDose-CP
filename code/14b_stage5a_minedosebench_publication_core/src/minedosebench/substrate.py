from __future__ import annotations
import numpy as np, pandas as pd, geopandas as gpd
from .io import require
from .splits import largest_component, spatial_axis_roles, exact_block_map

def build_90m(blocks:gpd.GeoDataFrame, edges:pd.DataFrame, wide:pd.DataFrame, ctx:pd.DataFrame):
    b=blocks.merge(wide,on=['block_id','MineID','MineN'],how='left',validate='1:1').merge(ctx,on='block_id',how='left',validate='1:1')
    active=(b.block_year_eligible_2023==True)&(b.block_year_eligible_2024==True)&b.pv_median_2023.notna()&b.pv_median_2024.notna()&b.ue_median_2024.notna()
    numeric_ctx=[c for c in ctx.columns if c!='block_id']
    active &= b[numeric_ctx].notna().all(axis=1)
    b['benchmark_baseline_eligible']=active
    out=[];split_rows=[]; exact=[]; axes=[]
    for mid,g in b.groupby('MineID',sort=True):
        ge=g[g.benchmark_baseline_eligible].copy()
        ee=edges[(edges.MineID.astype(str)==str(mid))].copy()
        mask,labels,counts=largest_component(ge,ee)
        ge['primary_component']=mask
        sec=ge[~mask].copy(); sec['benchmark_role']='secondary_support_stratum'; sec['spatial_axis_score']=np.nan;sec['spatial_axis_rank']=np.nan
        main=ge[mask].copy(); main2,axis=spatial_axis_roles(main,ee); axes.append((mid,*axis))
        ex=exact_block_map(main2,ee); ex['MineID']=mid; exact.append(ex)
        out.extend([main2,sec])
    activeg=pd.concat(out,ignore_index=True); full=b.merge(activeg[['block_id','primary_component','benchmark_role','spatial_axis_score','spatial_axis_rank']],on='block_id',how='left',validate='1:1')
    full['benchmark_role']=full.benchmark_role.fillna('baseline_ineligible')
    exact=pd.concat(exact,ignore_index=True) if exact else pd.DataFrame()
    axis=pd.DataFrame(axes,columns=['MineID','axis_x','axis_y'])
    return gpd.GeoDataFrame(full,geometry='geometry',crs=blocks.crs),exact,axis

def aggregate_180m(s90:gpd.GeoDataFrame, e90:pd.DataFrame):
    # Anchored 2x2 aggregation from the complete frozen 90 m child inventory.
    # A 180 m benchmark cell is baseline-eligible only when every available 90 m child is baseline-eligible;
    # this prevents silent omission of poor-quality children when the support changes.
    x=s90.copy(); x['grid_row180']=(x.grid_row.astype(int)//2);x['grid_col180']=(x.grid_col.astype(int)//2)
    key=['MineID','MineN','grid_row180','grid_col180']
    numeric=[c for c in ['pv_median_2023','pv_median_2024','ue_median_2024','water_fraction_2024','unclear_fraction_2024',
      'rainfall_2023_mm','rainfall_2024_mm','et_short_crop_2023_mm','et_short_crop_2024_mm','soil_organic_carbon_pct_0_15cm','soil_ph_cacl2_0_15cm','soil_clay_pct_0_15cm','soil_bulk_density_g_cm3_0_15cm','soil_available_water_capacity_mm_0_100cm','terrain_elevation_m','terrain_slope_deg','terrain_aspect_deg','terrain_roughness'] if c in x]
    records=[]
    for k,g in x.groupby(key,sort=True):
        ww=g.mine_footprint_area_m2.to_numpy(float); rec=dict(zip(key,k)); rec['block_id']=f'MDB180_{k[0]}_{int(k[2])}_{int(k[3])}'
        rec['available_child_count']=int(len(g)); rec['eligible_child_count']=int(g.benchmark_baseline_eligible.astype(bool).sum()); rec['eligible_child_fraction']=float(rec['eligible_child_count']/rec['available_child_count'])
        rec['benchmark_baseline_eligible']=bool(rec['eligible_child_count']==rec['available_child_count'] and rec['available_child_count']>0)
        rec['mine_footprint_area_m2']=float(ww.sum()); rec['centroid_x']=float(np.average(g.centroid_x,weights=ww));rec['centroid_y']=float(np.average(g.centroid_y,weights=ww)); rec['geometry']=g.geometry.union_all()
        if rec['benchmark_baseline_eligible']:
            for c in numeric: rec[c]=float(np.average(g[c].astype(float),weights=ww))
        else:
            for c in numeric: rec[c]=np.nan
        records.append(rec)
    b=gpd.GeoDataFrame(records,geometry='geometry',crs=s90.crs)
    # Queen edges from anchored 180 m row/col coordinates, within mine only.
    lookup={(r.MineID,int(r.grid_row180),int(r.grid_col180)):r.block_id for r in b.itertuples()}; er=[]
    for r in b.itertuples():
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                q=lookup.get((r.MineID,int(r.grid_row180)+dr,int(r.grid_col180)+dc))
                if q and str(r.block_id)<str(q): er.append((r.MineID,r.block_id,q,'queen'))
    edges=pd.DataFrame(er,columns=['MineID','source_block_id','target_block_id','contiguity'])
    out=[];exact=[];axes=[]
    for mid,gall in b.groupby('MineID',sort=True):
        ge=gall[gall.benchmark_baseline_eligible].copy(); ee=edges[edges.MineID==mid]
        require(len(ge)>0,f'No quality-consistent 180m cells for mine {mid}')
        mask,_,_=largest_component(ge,ee); ge=ge.copy();ge['primary_component']=mask
        sec=ge[~mask].copy();sec['benchmark_role']='secondary_support_stratum';sec['spatial_axis_score']=np.nan;sec['spatial_axis_rank']=np.nan
        main,axis=spatial_axis_roles(ge[mask].copy(),ee);axes.append((mid,*axis));ex=exact_block_map(main,ee);ex['MineID']=mid;exact.append(ex);out.extend([main,sec])
    activeg=pd.concat(out,ignore_index=True)
    full=b.merge(activeg[['block_id','primary_component','benchmark_role','spatial_axis_score','spatial_axis_rank']],on='block_id',how='left',validate='1:1')
    full['benchmark_role']=full.benchmark_role.fillna('baseline_ineligible')
    return gpd.GeoDataFrame(full,geometry='geometry',crs=s90.crs),edges,pd.concat(exact,ignore_index=True),pd.DataFrame(axes,columns=['MineID','axis_x','axis_y'])
