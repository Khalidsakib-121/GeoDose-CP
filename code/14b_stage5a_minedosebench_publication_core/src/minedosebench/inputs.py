from __future__ import annotations
import json, zipfile, tempfile
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from .io import require, sha256_file, safe_extract_zip

EXPECTED={
 'stage1':'24be83a4bd61db20fcbb02afaa1e9623029393909906195254b1974d318930c8',
 'stage2a':'6f41e47ed97e88ecd90ad67282908b1b55b993f0eed3f317cea48677c29dd4d4',
 'stage2b':'329a21f3b94d88f8f0b0b3327dbe250eb1df0911f73a6309c6c80c8417bd73a8',
 'stage3a':'9a46e21f4f488447f0829f079c8f19a345e6ac1748bc6907e20917bbb096d406',
 'stage4_closure':'d4d67252353f042723271ae468208bc05c0f6c67e4c2b3f7a106ee4338f60d36',
}

def verify_embedded_inputs(root: Path):
    paths={
      'stage1':root/'inputs/G3_STAGE1_MIN_AUDIT.zip',
      'stage2a':root/'inputs/G3_STAGE2_BLOCKS.zip',
      'stage2b':root/'inputs/GeoDose_Stage2B_OUTPUTS_FINAL.zip',
      'stage3a':root/'inputs/GeoDose_Stage3A_OUTPUTS.zip',
      'stage4_closure':root/'inputs/GeoDose_Stage4_FINAL_CLOSURE_OUTPUTS.zip',
    }
    for k,p in paths.items(): require(p.exists(),f'Missing embedded frozen input: {p}'); require(sha256_file(p)==EXPECTED[k],f'Frozen input hash mismatch: {k}')
    with zipfile.ZipFile(paths['stage4_closure']) as z:
        cert=json.loads(z.read('STAGE4_FINAL_FREEZE_CERTIFICATE.json'))
    require(cert.get('status')=='ACCEPTED_AND_FROZEN','Stage4 is not accepted/frozen')
    require(cert.get('production_rerun_required') is False,'Stage4 closure contract mismatch')
    return paths,cert

def load_stage2(root: Path):
    paths,_=verify_embedded_inputs(root)
    td=tempfile.TemporaryDirectory(prefix='mdb_inputs_'); d=Path(td.name)
    safe_extract_zip(paths['stage2a'],d/'s2a'); safe_extract_zip(paths['stage2b'],d/'s2b'); safe_extract_zip(paths['stage3a'],d/'s3a')
    gpkg=d/'s2a/G3_STAGE2_BLOCKS/mine_blocks_90m.gpkg'
    blocks=gpd.read_file(gpkg,layer='blocks_90m')
    edges=pd.read_csv(d/'s2a/G3_STAGE2_BLOCKS/block_graph_edges.csv')
    dea=pd.read_csv(d/'s2b/dea_block_year_prescreen.csv.gz')
    roles=pd.read_csv(d/'s3a/outputs_stage3a/mine_role_registry.csv')
    require(str(blocks.crs).upper().endswith('9473'),'Stage2A CRS is not EPSG:9473')
    require(len(blocks)==27042,'Stage2A block count changed')
    return td,blocks,edges,dea,roles

def wide_dea(dea: pd.DataFrame)->pd.DataFrame:
    cols=['block_id','MineID','MineN','year','pv_median','ue_median','water_fraction','unclear_fraction','valid_observation_count','valid_pixel_fraction','block_year_eligible','all_years_eligible','high_ue_fraction_25']
    require(set(cols).issubset(dea.columns),'Stage2B schema missing required columns')
    d=dea[cols].copy(); d['year']=d.year.astype(int)
    pieces=[]
    for y in [2023,2024,2025]:
        q=d[d.year==y].drop(columns=['year']).copy(); q=q.rename(columns={c:f'{c}_{y}' for c in q.columns if c not in ['block_id','MineID','MineN']}); pieces.append(q)
    out=pieces[0]
    for q in pieces[1:]: out=out.merge(q,on=['block_id','MineID','MineN'],how='outer',validate='1:1')
    return out
