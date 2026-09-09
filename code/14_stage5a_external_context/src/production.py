from __future__ import annotations
import json, os, shutil, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from .common import require, read_json, write_json, sha256_file, deterministic_zip, utc_now
from .authority import verify_base, load_census, GOVERNING, THRESHOLD, BASE_AGG, CENSUS_ZIP_SHA, CENSUS_AGG

NUMERIC_COLS=['rainfall_2023_mm','rainfall_2024_mm','et_short_crop_2023_mm','et_short_crop_2024_mm','soil_organic_carbon_pct_0_15cm','soil_ph_cacl2_0_15cm','soil_clay_pct_0_15cm','soil_bulk_density_g_cm3_0_15cm','soil_available_water_capacity_mm_0_100cm','terrain_elevation_m','terrain_slope_deg','terrain_aspect_deg','terrain_roughness']
CLIMATE=[('rainfall_2023_mm','silo_rainfall_2023','SILO_daily_rain_2023'),('rainfall_2024_mm','silo_rainfall_2024','SILO_daily_rain_2024'),('et_short_crop_2023_mm','silo_et_short_crop_2023','SILO_et_short_crop_2023'),('et_short_crop_2024_mm','silo_et_short_crop_2024','SILO_et_short_crop_2024')]
SOIL=[
('SOC_0_5','tern_soc_000_005','SOC','000_005'),('SOC_5_15','tern_soc_005_015','SOC','005_015'),
('CLY_0_5','tern_clay_000_005','CLY','000_005'),('CLY_5_15','tern_clay_005_015','CLY','005_015'),
('BDW_0_5','tern_bulk_density_000_005','BDW','000_005'),('BDW_5_15','tern_bulk_density_005_015','BDW','005_015'),
('PHC_0_5','tern_ph_cacl2_000_005','PHC','000_005'),('PHC_5_15','tern_ph_cacl2_005_015','PHC','005_015'),
('AWC_0_5','tern_awc_000_005','AWC','000_005'),('AWC_5_15','tern_awc_005_015','AWC','005_015'),('AWC_15_30','tern_awc_015_030','AWC','015_030'),('AWC_30_60','tern_awc_030_060','AWC','030_060'),('AWC_60_100','tern_awc_060_100','AWC','060_100')]
RAW_RANGES={'SOC_0_5':(0,50),'SOC_5_15':(0,50),'CLY_0_5':(0,100),'CLY_5_15':(0,100),'BDW_0_5':(.2,3),'BDW_5_15':(.2,3),'PHC_0_5':(2,10.5),'PHC_5_15':(2,10.5),'AWC_0_5':(0,100),'AWC_5_15':(0,100),'AWC_15_30':(0,100),'AWC_30_60':(0,100),'AWC_60_100':(0,100)}


def _source_row(source_manifest, family, source_id):
    q=[r for r in source_manifest['source_rows'] if r.get('source_family')==family and r.get('source_id')==source_id]
    require(len(q)==1,f'Expected exactly one census source manifest row for {family}/{source_id}, found {len(q)}')
    return q[0]

def _locked_paths(cache: Path, source_manifest):
    result={}
    for outcol,layer,sid in CLIMATE:
        r=_source_row(source_manifest,'SILO',sid);p=cache/'silo'/'annual_aoi'/Path(r['derived_aoi_raster']).name
        require(p.exists(),f'Missing census-locked SILO AOI raster: {p}')
        require(sha256_file(p)==r['derived_aoi_sha256'],f'Census-locked SILO AOI hash mismatch: {p.name}')
        result[layer]=p
    for name,layer,pid,depth in SOIL:
        rows=[r for r in source_manifest['source_rows'] if r.get('source_family')=='TERN_SLGA_R2' and r.get('product_id')==pid and r.get('depth_cm')==depth]
        require(len(rows)==1,f'Expected one census-locked TERN source for {pid}/{depth}')
        r=rows[0];p=cache/'tern'/'aoi_crops'/Path(r['aoi_crop']).name
        require(p.exists(),f'Missing census-locked TERN AOI crop: {p}')
        require(sha256_file(p)==r['aoi_crop_sha256'],f'Census-locked TERN AOI hash mismatch: {p.name}')
        meta=cache/'tern'/'metadata'/(r['source_id']+'.json')
        require(meta.exists() and sha256_file(meta)==r['metadata_sha256'],f'Census-locked TERN metadata hash mismatch: {meta.name}')
        result[layer]=p
    raw=_source_row(source_manifest,'Geoscience_Australia','GA_DEM_SRTM_1Second_2024')
    rawp=cache/'ga'/'raw_dem'/'GA_DEM_SRTM_1Second_2024_AOI.tif'
    require(rawp.exists() and sha256_file(rawp)==raw['aoi_crop_sha256'],'Census-locked GA raw DEM missing/hash mismatch')
    der=_source_row(source_manifest,'Geoscience_Australia','GA_TERRAIN_DERIVED')['details']
    ddir=cache/'ga'/'terrain_derived'; names={'elevation':'terrain_elevation_epsg9473_30m.tif','slope':'terrain_slope_deg.tif','aspect':'terrain_aspect_deg.tif','roughness':'terrain_roughness_3x3_std_m.tif'}
    result['ga_raw_dem']=rawp
    for k,n in names.items():
        p=ddir/n; require(p.exists() and sha256_file(p)==der['derived_sha256'][n],f'Census-locked GA terrain raster missing/hash mismatch: {n}'); result['ga_terrain_'+k]=p
    return result

def _subset_blocks(base, census, cache):
    sys.path.insert(0,str(base/'src'))
    from geodose_context.provenance import verify_and_extract_prep
    from geodose_context.pipeline import required_blocks
    prep,req,g,rep=verify_and_extract_prep(base,cache/'_final_context_production_v1_0_1'/'prep_work')
    allb=required_blocks(g,req)
    keep=census['complete'].copy(); keep['block_id']=keep.block_id.astype(str)
    ids=set(keep.block_id)
    b=allb[allb.block_id.astype(str).isin(ids)].copy().sort_values('block_id',kind='mergesort').reset_index(drop=True)
    require(len(b)==23618 and set(b.block_id.astype(str))==ids,'Frozen 23,618 block geometry subset mismatch')
    km=keep.set_index('block_id')[['MineID','MineN']]
    geom_ids=b.block_id.astype(str)
    require(all(str(km.loc[i,'MineID'])==str(b.loc[j,'MineID']) and str(km.loc[i,'MineN'])==str(b.loc[j,'MineN']) for j,i in enumerate(geom_ids)),'Census/base mine identity mismatch on retained substrate')
    require(str(b.crs).upper()=='EPSG:9473','Retained substrate CRS is not EPSG:9473')
    return b,req,allb

def _checkpoint_extract(base, blocks, raster, var, layer, cache, expected_support, min_valid=THRESHOLD):
    sys.path.insert(0,str(base/'src'))
    from geodose_context.exact_area import aggregate_scalar
    prod=cache/'_final_context_production_v1_0_1'; ext=prod/'extraction';ext.mkdir(parents=True,exist_ok=True)
    p=ext/f'{var}.csv.gz';m=ext/f'{var}.json';source_sha=sha256_file(raster);bidsha=hashlib.sha256('\n'.join(blocks.block_id.astype(str)).encode()).hexdigest()
    if p.exists() and m.exists():
        mm=read_json(m)
        if mm.get('source_raster_sha256')==source_sha and mm.get('block_id_sha256')==bidsha and mm.get('threshold')==min_valid and mm.get('layer_id')==layer:
            d=pd.read_csv(p,dtype={'block_id':str})
            if len(d)==len(blocks) and d.block_id.tolist()==blocks.block_id.astype(str).tolist():return d,True
    vals,frac,_=aggregate_scalar(blocks,raster,prod/'weights',min_valid)
    d=pd.DataFrame({'block_id':blocks.block_id.astype(str),var:vals,f'{var}__valid_area_fraction':frac})
    exp=expected_support.set_index('block_id').loc[d.block_id,f'{layer}__valid_area_fraction'].to_numpy(float)
    diff=np.max(np.abs(frac-exp));require(diff<=5e-12,f'{layer} valid-area fractions do not reconcile to frozen census; max abs diff={diff}')
    d.to_csv(p,index=False,compression={'method':'gzip','compresslevel':9,'mtime':0})
    write_json(m,{'variable':var,'layer_id':layer,'source_raster_sha256':source_sha,'block_id_sha256':bidsha,'threshold':min_valid,'support_reconciliation_max_abs_diff':float(diff),'rows':len(d)})
    return d,False

def _range_raw(name,x):
    x=np.asarray(x,float);require(np.isfinite(x).all(),f'{name} contains nonfinite values on retained substrate');lo,hi=RAW_RANGES[name];require(x.min()>=lo and x.max()<=hi,f'{name} outside native product range [{lo},{hi}]: {x.min()}..{x.max()}')

def _attrition_outputs(census, req, out):
    union=census['union'].copy(); req2=req[['block_id','MineID','MineN','centroid_x_epsg9473','centroid_y_epsg9473','centroid_lon_wgs84','centroid_lat_wgs84','required_for_primary_freeze']].copy();req2=req2[req2.required_for_primary_freeze.astype(bool)].copy();req2['block_id']=req2.block_id.astype(str)
    a=union.merge(req2,on=['block_id','MineID','MineN'],how='left',validate='1:1');require(a[['centroid_x_epsg9473','centroid_y_epsg9473']].notna().all().all(),'Pre-outcome centroid audit merge failed')
    a['substrate_status']=np.where(a.context_complete_at_frozen_0_99.astype(bool),'retained_context_complete','excluded_support_gap')
    rows=[]
    for (mine,status),g in a.groupby(['MineN','substrate_status'],sort=True):
        r={'MineN':mine,'substrate_status':status,'n_blocks':int(len(g))}
        for c in ['centroid_x_epsg9473','centroid_y_epsg9473','centroid_lon_wgs84','centroid_lat_wgs84']:
            x=g[c].astype(float);r.update({f'{c}__min':float(x.min()),f'{c}__mean':float(x.mean()),f'{c}__max':float(x.max())})
        rows.append(r)
    pd.DataFrame(rows).to_csv(out/'CONTEXT_ATTRITION_PREOUTCOME_SPATIAL_AUDIT.csv',index=False)
    bm=[]
    for mine,g in union.groupby('MineN',sort=True):
        n=len(g);k=int(g.context_complete_at_frozen_0_99.astype(bool).sum());e=n-k
        bm.append({'MineN':mine,'original_required_blocks':n,'retained_context_complete_blocks':k,'excluded_support_gap_blocks':e,'retention_fraction':k/n,'exclusion_fraction':e/n})
    pd.DataFrame(bm).to_csv(out/'CONTEXT_ATTRITION_BY_MINE.csv',index=False)
    ex=census['excluded'].copy();ex['failed_governing_layers']=ex.failed_governing_layers.fillna('')
    fp=ex.groupby(['MineN','failed_governing_layers'],dropna=False).size().reset_index(name='excluded_blocks').sort_values(['excluded_blocks','MineN'],ascending=[False,True])
    fp.to_csv(out/'CONTEXT_ATTRITION_FAILURE_PATTERNS.csv',index=False)
    write_json(out/'CONTEXT_ATTRITION_AUDIT.json',{'original_required_blocks':23710,'retained_context_complete_blocks':23618,'excluded_support_gap_blocks':92,'retention_fraction':23618/23710,'exclusion_fraction':92/23710,'decision_basis':'Prospective external-source support only, before benchmark outcomes or M2-M6 performance; exact >=0.99 rule on all 18 governing layers; no imputation or threshold change.','allowed_preoutcome_audit_fields':['MineN','centroid_x_epsg9473','centroid_y_epsg9473','centroid_lon_wgs84','centroid_lat_wgs84']})

def _range_qa(ctx,schema,out):
    rows=[]
    for c in NUMERIC_COLS:
        x=ctx[c].to_numpy(float);lo,hi=map(float,schema['physical_ranges'][c]);q=np.quantile(x,[.01,.05,.5,.95,.99])
        rows.append({'variable':c,'n':len(x),'minimum':float(x.min()),'q01':float(q[0]),'q05':float(q[1]),'mean':float(x.mean()),'std_population':float(x.std(ddof=0)),'median':float(q[2]),'q95':float(q[3]),'q99':float(q[4]),'maximum':float(x.max()),'frozen_physical_min':lo,'frozen_physical_max':hi,'all_in_range':bool(((x>=lo)&(x<=hi)).all())})
    pd.DataFrame(rows).to_csv(out/'CONTEXT_RANGE_QA.csv',index=False)
    br=[]
    tmp=ctx.merge(pd.read_csv(out/'FROZEN_CONTEXT_COMPLETE_SUBSTRATE.csv',dtype={'block_id':str})[['block_id','MineN']],on='block_id',validate='1:1')
    for mine,g in tmp.groupby('MineN',sort=True):
        for c in NUMERIC_COLS:
            x=g[c].to_numpy(float);br.append({'MineN':mine,'variable':c,'n':len(x),'minimum':float(x.min()),'mean':float(x.mean()),'median':float(np.median(x)),'maximum':float(x.max())})
    pd.DataFrame(br).to_csv(out/'CONTEXT_RANGE_SUMMARY_BY_MINE.csv',index=False)

def _snapshot(cache, census, out):
    """Create a compact snapshot of *only* source files explicitly frozen by the census.

    This avoids accidentally packaging unrelated cache/work files while preserving every
    raster/metadata byte used for final context extraction.
    """
    snap=out/'_SOURCE_SNAPSHOT'
    if snap.exists():shutil.rmtree(snap)
    snap.mkdir()
    files=[]; selected=[]
    sm=census['source_manifest']
    for r in sm['source_rows']:
        fam=r.get('source_family'); sid=r.get('source_id')
        if fam=='SILO':
            selected.append(cache/'silo'/'annual_aoi'/Path(r['derived_aoi_raster']).name)
        elif fam=='TERN_SLGA_R2':
            selected.append(cache/'tern'/'aoi_crops'/Path(r['aoi_crop']).name)
            selected.append(cache/'tern'/'metadata'/(sid+'.json'))
        elif fam=='Geoscience_Australia' and sid=='GA_DEM_SRTM_1Second_2024':
            selected.append(cache/'ga'/'raw_dem'/'GA_DEM_SRTM_1Second_2024_AOI.tif')
        elif fam=='Geoscience_Australia' and sid=='GA_TERRAIN_DERIVED':
            for n in sorted(r['details']['derived_sha256']): selected.append(cache/'ga'/'terrain_derived'/n)
    # De-duplicate while preserving deterministic path order.
    selected=sorted(set(Path(x) for x in selected),key=lambda x:x.as_posix())
    require(len(selected)==35,f'Expected 35 exact frozen source snapshot files (4 SILO + 26 TERN crop/meta + 1 GA raw + 4 GA derived), found {len(selected)}')
    for p in selected:
        require(p.exists(),f'Frozen snapshot source missing: {p}')
        rel=p.relative_to(cache);dst=snap/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        files.append({'file':rel.as_posix(),'size':p.stat().st_size,'sha256':sha256_file(p)})
    write_json(snap/'SNAPSHOT_MANIFEST.json',{'files':files,'count':len(files),'source_authority':'CONTEXT_SUPPORT_CENSUS_OUTPUTS v1.0.1','census_output_aggregate_sha256':CENSUS_AGG,'policy':'EXACT_CENSUS_FROZEN_SOURCE_FILES_ONLY'})
    z=out/'GeoDose_Stage5A_CONTEXT_SOURCE_SNAPSHOT.zip';deterministic_zip(snap,z);shutil.rmtree(snap);return z

def run(package_root: Path, base: Path, output_dir: Path|None=None):
    package_root=Path(package_root).resolve();base=Path(base).resolve();verify_base(base)
    out=Path(output_dir).resolve() if output_dir else package_root/'FINAL_CONTEXT_PRODUCTION_OUTPUTS';out.mkdir(parents=True,exist_ok=True)
    cache=base/'CONTEXT_SOURCE_CACHE';require(cache.exists(),'Existing CONTEXT_SOURCE_CACHE missing. Do not redownload; restore the census cache.')
    census=load_census(package_root,out/'_census_authority_work')
    sys.path.insert(0,str(base/'src'))
    from geodose_context.pipeline import load_contract
    contract,sources,schema,local=load_contract(base)
    require(float(contract['min_valid_area_fraction'])==THRESHOLD,'Base acquisition valid-area threshold changed')
    blocks,req,allb=_subset_blocks(base,census,cache)
    locked=_locked_paths(cache,census['source_manifest'])
    print('PASS frozen_authorities original=23710 retained=23618 excluded=92 threshold=0.99')
    print('PASS exact_census_locked_source_bytes 4 SILO + 13 TERN + GA raw/derived')
    support=census['support'].copy();support['block_id']=support.block_id.astype(str);support=support[support.block_id.isin(set(blocks.block_id.astype(str)))].copy().sort_values('block_id').reset_index(drop=True)
    require(len(support)==23618,'Retained support census subset mismatch')
    parts=[];valid_cols=[];recon=[];step=0
    for outcol,layer,sid in CLIMATE:
        step+=1;print(f'[{step:02d}/18] {layer}: exact census-locked AOI raster -> retained-block area-weighted value')
        d,reused=_checkpoint_extract(base,blocks,locked[layer],outcol,layer,cache,support);parts.append(d[['block_id',outcol]]);valid_cols.append(d[['block_id',f'{outcol}__valid_area_fraction']])
        exp=support.set_index('block_id').loc[d.block_id,f'{layer}__valid_area_fraction'].to_numpy(float);got=d[f'{outcol}__valid_area_fraction'].to_numpy(float);recon.append({'governing_layer':layer,'max_abs_valid_area_fraction_difference':float(np.max(np.abs(got-exp))),'pass':bool(np.max(np.abs(got-exp))<=5e-12),'checkpoint_reused':reused})
    raw={};raw_df=pd.DataFrame({'block_id':blocks.block_id.astype(str)})
    for name,layer,pid,depth in SOIL:
        step+=1;print(f'[{step:02d}/18] {layer}: exact census-locked TERN crop -> retained-block area-weighted value')
        d,reused=_checkpoint_extract(base,blocks,locked[layer],name,layer,cache,support);_range_raw(name,d[name]);raw[name]=d[name].to_numpy(float);raw_df[name]=raw[name];valid_cols.append(d[['block_id',f'{name}__valid_area_fraction']])
        exp=support.set_index('block_id').loc[d.block_id,f'{layer}__valid_area_fraction'].to_numpy(float);got=d[f'{name}__valid_area_fraction'].to_numpy(float);recon.append({'governing_layer':layer,'max_abs_valid_area_fraction_difference':float(np.max(np.abs(got-exp))),'pass':bool(np.max(np.abs(got-exp))<=5e-12),'checkpoint_reused':reused})
    soil=pd.DataFrame({'block_id':blocks.block_id.astype(str)})
    soil['soil_organic_carbon_pct_0_15cm']=(5*raw['SOC_0_5']+10*raw['SOC_5_15'])/15
    soil['soil_ph_cacl2_0_15cm']=(5*raw['PHC_0_5']+10*raw['PHC_5_15'])/15
    soil['soil_clay_pct_0_15cm']=(5*raw['CLY_0_5']+10*raw['CLY_5_15'])/15
    soil['soil_bulk_density_g_cm3_0_15cm']=(5*raw['BDW_0_5']+10*raw['BDW_5_15'])/15
    soil['soil_available_water_capacity_mm_0_100cm']=(raw['AWC_0_5']/100)*50+(raw['AWC_5_15']/100)*100+(raw['AWC_15_30']/100)*150+(raw['AWC_30_60']/100)*300+(raw['AWC_60_100']/100)*400
    parts.append(soil)
    step+=1;print(f'[{step:02d}/18] ga_terrain_common: exact census-locked derived terrain rasters -> retained-block area-weighted values')
    from geodose_context.exact_area import aggregate_terrain
    tres,_=aggregate_terrain(blocks,locked['ga_terrain_elevation'],locked['ga_terrain_slope'],locked['ga_terrain_aspect'],locked['ga_terrain_roughness'],cache/'_final_context_production_v1_0_1'/'weights',THRESHOLD,float(contract['terrain']['flat_slope_threshold_deg']))
    terr=pd.DataFrame({'block_id':blocks.block_id.astype(str),'terrain_elevation_m':tres['elevation'],'terrain_slope_deg':tres['slope'],'terrain_aspect_deg':tres['aspect'],'terrain_roughness':tres['roughness']});parts.append(terr)
    exp=support.set_index('block_id').loc[terr.block_id,'ga_terrain_common__valid_area_fraction'].to_numpy(float);got=np.asarray(tres['valid_fraction'],float);diff=float(np.max(np.abs(got-exp)));require(diff<=5e-12,f'GA terrain common support does not reconcile to census: {diff}');recon.append({'governing_layer':'ga_terrain_common','max_abs_valid_area_fraction_difference':diff,'pass':True,'checkpoint_reused':False})
    # Freeze registries first.
    frozen=census['complete'].sort_values('block_id',kind='mergesort').reset_index(drop=True).copy();frozen['frozen_context_complete_at_0_99']=True;frozen['freeze_authority']='Stage5A External Context Support Census v1.0.1';frozen['freeze_authority_output_aggregate_sha256']=CENSUS_AGG;frozen.to_csv(out/'FROZEN_CONTEXT_COMPLETE_SUBSTRATE.csv',index=False)
    exclusions=census['excluded'].sort_values('block_id',kind='mergesort').reset_index(drop=True).copy();exclusions['frozen_exclusion_reason']='external_context_support_below_0.99_on_one_or_more_governing_layers';exclusions['freeze_authority']='Stage5A External Context Support Census v1.0.1';exclusions.to_csv(out/'FROZEN_CONTEXT_SUPPORT_EXCLUSIONS.csv',index=False)
    ctx=pd.DataFrame({'block_id':blocks.block_id.astype(str)})
    for d in parts:ctx=ctx.merge(d,on='block_id',how='left',validate='1:1')
    ctx=ctx[['block_id']+NUMERIC_COLS].sort_values('block_id',kind='mergesort').reset_index(drop=True)
    require(len(ctx)==23618 and ctx.block_id.is_unique,'Final context row identity failure');arr=ctx[NUMERIC_COLS].to_numpy(float);require(np.isfinite(arr).all(),'Final context contains NaN/infinity')
    for c,(lo,hi) in schema['physical_ranges'].items():
        x=ctx[c].to_numpy(float);require(((x>=lo)&(x<=hi)).all(),f'{c} violates frozen physical range [{lo},{hi}]')
    ctx.to_csv(out/'mine_context_covariates.csv',index=False)
    raw_df.to_csv(out/'SOIL_DEPTH_COMPONENT_AUDIT.csv.gz',index=False,compression={'method':'gzip','compresslevel':9,'mtime':0})
    # Valid-area evidence comes from the independently completed census; restrict exactly to frozen retained IDs.
    vcols=['block_id']+[f'{x}__valid_area_fraction' for x in GOVERNING]
    sva=support[vcols].copy();require(sva[[c for c in vcols if c!='block_id']].min().min()>=THRESHOLD,'Frozen retained support evidence below 0.99');sva.to_csv(out/'CONTEXT_VALID_AREA_QA.csv.gz',index=False,compression={'method':'gzip','compresslevel':9,'mtime':0})
    pd.DataFrame(recon).to_csv(out/'CONTEXT_SUPPORT_RECONCILIATION.csv',index=False);require(all(r['pass'] for r in recon) and len(recon)==18,'Support reconciliation did not pass all 18 governing layers')
    # Formula audit: recompute from raw depth components independently of final table construction.
    formula_checks={
      'soil_organic_carbon_pct_0_15cm':(5*raw_df.SOC_0_5.to_numpy()+10*raw_df.SOC_5_15.to_numpy())/15,
      'soil_ph_cacl2_0_15cm':(5*raw_df.PHC_0_5.to_numpy()+10*raw_df.PHC_5_15.to_numpy())/15,
      'soil_clay_pct_0_15cm':(5*raw_df.CLY_0_5.to_numpy()+10*raw_df.CLY_5_15.to_numpy())/15,
      'soil_bulk_density_g_cm3_0_15cm':(5*raw_df.BDW_0_5.to_numpy()+10*raw_df.BDW_5_15.to_numpy())/15,
      'soil_available_water_capacity_mm_0_100cm':(raw_df.AWC_0_5.to_numpy()/100)*50+(raw_df.AWC_5_15.to_numpy()/100)*100+(raw_df.AWC_15_30.to_numpy()/100)*150+(raw_df.AWC_30_60.to_numpy()/100)*300+(raw_df.AWC_60_100.to_numpy()/100)*400}
    fa=[]
    for c,x in formula_checks.items():fa.append({'derived_variable':c,'max_abs_recompute_residual':float(np.max(np.abs(ctx[c].to_numpy()-x))),'pass':bool(np.max(np.abs(ctx[c].to_numpy()-x))<=1e-12)})
    pd.DataFrame(fa).to_csv(out/'CONTEXT_DERIVATION_FORMULA_AUDIT.csv',index=False);require(all(r['pass'] for r in fa),'Depth-harmonization formula audit failed')
    _attrition_outputs(census,req,out);_range_qa(ctx,schema,out)
    # Freeze provenance based on already-audited census source identities; no live-source drift allowed at production.
    prov={'production_stage':'GeoDose-CP Stage5A External Context Production','production_version':'1.0.1-stage5a-external-context-production-freeze-candidate','source_policy':'EXACT_CENSUS_BYTES_ONLY_NO_NETWORK_DRIFT','base_acquisition_aggregate_sha256':BASE_AGG,'support_census_zip_sha256':CENSUS_ZIP_SHA,'support_census_output_aggregate_sha256':CENSUS_AGG,'frozen_valid_area_threshold':THRESHOLD,'source_rows':census['source_manifest']['source_rows'],'credential_values_recorded':False,'local_source_hash_validation_passed':True,'generated_utc':utc_now()};write_json(out/'SOURCE_PROVENANCE_MANIFEST.json',prov)
    meta={'scientific_context_complete':True,'spatial_aggregation_contract':'AREA_WEIGHTED_BLOCK_POLYGON','context_substrate':{'original_required_blocks':23710,'frozen_context_complete_blocks':23618,'prospectively_excluded_for_external_support':92,'frozen_valid_area_threshold':0.99,'support_census_version':'1.0.1','support_census_output_aggregate_sha256':CENSUS_AGG,'selection_timing':'before MineDoseBench outcomes and before M2-M6 performance','imputation_performed':False,'threshold_changed':False},'climate':{'source':'Queensland Government SILO gridded climate','variables':['daily_rain','et_short_crop'],'years':[2023,2024],'temporal_aggregation':'annual sums; all daily grids required valid per source cell','spatial_aggregation':'AREA_WEIGHTED_BLOCK_POLYGON','source_identity':'Exact census-locked annual AOI raster hashes in SOURCE_PROVENANCE_MANIFEST.json','license_or_terms':sources['silo']['license'],'citation':sources['silo']['citation']},'soil':{'source':'TERN Soil and Landscape Grid of Australia Release 2 estimated-value products','depth_harmonization':'0-15 cm=(5*x_0_5+10*x_5_15)/15; AWC 0-100 cm=sum((AWC_percent/100)*layer_thickness_mm) for 0-5,5-15,15-30,30-60,60-100 cm','spatial_aggregation':'AREA_WEIGHTED_BLOCK_POLYGON','source_identity':'Exact census-locked COG crop and metadata hashes in SOURCE_PROVENANCE_MANIFEST.json','license':sources['tern']['license'],'product_specific_citations':'Recorded exactly in SOURCE_PROVENANCE_MANIFEST.json'},'terrain':{'source':'Geoscience Australia DEM_SRTM_1Second_2024, dataset PID ga/72759','derivation':'30 m EPSG:9473 bilinear DEM; Horn 3x3 slope; compass downslope aspect with circular block mean and 0 deg convention for flat/undefined; 3x3 population-SD elevation roughness','spatial_aggregation':'AREA_WEIGHTED_BLOCK_POLYGON','source_identity':'Exact census-locked GA raw and derived raster hashes in SOURCE_PROVENANCE_MANIFEST.json','license':sources['ga_terrain']['license'],'dataset_pid':sources['ga_terrain']['dataset_pid'],'citation':sources['ga_terrain']['title']},'extraction_software':{'base_acquisition_source_aggregate_sha256':BASE_AGG,'production_version':'1.0.1-stage5a-external-context-production-freeze-candidate','produced_utc':utc_now()},'claim_boundary':'External context production only. No MineDoseBench outcome, M2-M6 performance, threshold tuning, imputation, or automatic Stage5A benchmark freeze.'};write_json(out/'context_metadata.json',meta)
    snapshot=_snapshot(cache,census,out)
    # Copy frozen support evidence from authority for reviewer-traceable provenance.
    shutil.copy2(census['root']/'SOURCE_LAYER_SUPPORT_SUMMARY.csv',out/'FROZEN_SOURCE_LAYER_SUPPORT_SUMMARY.csv')
    # Explicit handoff folder. It is NOT compatible with the unchanged v1.1.0 MineDoseBench generator, which still requires 23,710 rows.
    ready=out/'READY_FOR_STAGE5A_REVISED_FREEZE';ready.mkdir(exist_ok=True)
    for n in ['mine_context_covariates.csv','context_metadata.json','FROZEN_CONTEXT_COMPLETE_SUBSTRATE.csv','FROZEN_CONTEXT_SUPPORT_EXCLUSIONS.csv']:
        shutil.copy2(out/n,ready/n)
    (ready/'IMPORTANT_NEXT_STEP.txt').write_text('These files freeze the prospectively context-complete 23,618-block substrate. Do NOT run the unchanged MineDoseBench Generator v1.1.0, because that version still requires all 23,710 original rows. The next stage must be a prospectively revised MineDoseBench freeze package that accepts this frozen substrate without examining outcomes or M2-M6 performance.\n',encoding='utf-8')
    report={'status':'FINAL_CONTEXT_PRODUCTION_COMPLETE_PENDING_VERIFICATION','original_required_blocks':23710,'frozen_context_complete_blocks':23618,'frozen_support_exclusions':92,'required_numeric_columns':13,'frozen_valid_area_threshold':0.99,'source_policy':'EXACT_CENSUS_BYTES_ONLY_NO_NETWORK_DRIFT','source_snapshot_sha256':sha256_file(snapshot),'benchmark_outcomes_read':False,'M2_M6_results_read':False,'thresholds_retuned':False,'imputation_performed':False,'unsupported_blocks_recovered':False,'automatic_minedosebench_freeze':False,'generated_utc':utc_now()};write_json(out/'CONTEXT_PRODUCTION_REPORT.json',report)
    return out
