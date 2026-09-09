from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import pandas as pd
from .common import require, sha256_file, read_json, safe_extract_zip, acquisition_manifest_aggregate

BASE_AGG='7a99a75d2958f2bbff6fa5f91c2108d881d0a42295090a2cfadabb29e562b181'
CENSUS_ZIP_SHA='2facc83842bd2ec8e30a2763c7040cb5a6e7446a0292d52d1eff0a1ea9c5152e'
CENSUS_AGG='053e09d0ab54fad015c5c2787029ff8bbb1b28195b31115f7ae76cfe6cd68177'
THRESHOLD=0.99
GOVERNING=[
'silo_rainfall_2023','silo_rainfall_2024','silo_et_short_crop_2023','silo_et_short_crop_2024',
'tern_soc_000_005','tern_soc_005_015','tern_clay_000_005','tern_clay_005_015',
'tern_bulk_density_000_005','tern_bulk_density_005_015','tern_ph_cacl2_000_005','tern_ph_cacl2_005_015',
'tern_awc_000_005','tern_awc_005_015','tern_awc_015_030','tern_awc_030_060','tern_awc_060_100','ga_terrain_common']

def verify_base(base: Path):
    base=Path(base);mp=base/'PACKAGE_MANIFEST.json';require(mp.exists(),f'Missing base PACKAGE_MANIFEST.json: {base}')
    m=read_json(mp)
    require(m.get('package')=='GeoDose_Stage5A_External_Context_Acquisition_v1_0_1_FREEZE',f"Wrong base acquisition package: {m.get('package')}")
    require(m.get('aggregate_sha256')==BASE_AGG,'Base acquisition aggregate identity mismatch')
    rows=m.get('immutable_files',[]);require(len(rows)==37 and m.get('immutable_file_count')==37,'Base acquisition immutable-file count mismatch')
    bad=[]
    for r in rows:
        p=base/r['file']
        if not p.exists() or p.stat().st_size!=int(r['size']) or sha256_file(p)!=r['sha256']:bad.append(r['file'])
    require(not bad,f'Base acquisition source freeze mismatch: {bad[:8]}')
    # IMPORTANT: Acquisition v1.0.1 uses a TSV+size+final-newline aggregate contract.
    # Production v1.0.0 incorrectly reused a different package aggregate function here.
    # Recompute with the *base package's own* exact contract and compare both the
    # declared identity and the accepted frozen aggregate.
    calc=acquisition_manifest_aggregate(rows)
    require(calc==m.get('aggregate_sha256'),f'Base acquisition manifest aggregate recomputation mismatch: {calc}')
    require(calc==BASE_AGG,'Base acquisition aggregate does not equal accepted freeze')
    return m

def load_census(package_root: Path, work: Path):
    z=Path(package_root)/'inputs'/'CONTEXT_SUPPORT_CENSUS_OUTPUTS_ACCEPTED.zip'
    require(z.exists(),'Embedded accepted support census ZIP missing')
    require(sha256_file(z)==CENSUS_ZIP_SHA,'Embedded support census ZIP SHA-256 mismatch')
    if work.exists():shutil.rmtree(work)
    safe_extract_zip(z,work)
    om=read_json(work/'OUTPUT_FILE_MANIFEST.json');rows=om.get('files',[])
    require(om.get('aggregate_sha256')==CENSUS_AGG,'Census internal aggregate declaration mismatch')
    calc=hashlib.sha256('\n'.join(f"{r['file']}\t{r['size']}\t{r['sha256']}" for r in rows).encode()).hexdigest()
    require(calc==CENSUS_AGG,'Census internal aggregate recomputation mismatch')
    for r in rows:
        p=work/r['file'];require(p.exists() and p.stat().st_size==r['size'] and sha256_file(p)==r['sha256'],f'Census output integrity mismatch: {r["file"]}')
    rep=read_json(work/'SUPPORT_CENSUS_REPORT.json');ver=read_json(work/'SUPPORT_CENSUS_VERIFICATION.json')
    require(ver.get('status')=='VERIFIED_SUPPORT_CENSUS_OUTPUTS','Census is not independently verified output')
    require(all(ver.get('checks',{}).values()),'One or more support-census verification checks failed')
    require(rep.get('required_blocks')==23710 and rep.get('context_complete_blocks')==23618 and rep.get('unsupported_union_blocks')==92,'Frozen census counts mismatch')
    require(float(rep.get('frozen_valid_area_threshold'))==THRESHOLD,'Frozen census valid-area threshold changed')
    require(rep.get('governing_layer_count')==18 and rep.get('governing_layers')==GOVERNING,'Frozen governing-layer registry changed')
    b=rep.get('scientific_boundary',{})
    for k in ['benchmark_outcomes_read','M2_M6_results_read','threshold_retuned','valid_area_threshold_changed','imputation_performed','blocks_dropped_or_rewritten','candidate_block_list_applied_to_stage5a','stage5a_freeze_authorized']:
        require(b.get(k) is False,f'Census scientific boundary not prospective: {k}')
    complete=pd.read_csv(work/'CONTEXT_COMPLETE_CANDIDATE_BLOCKS.csv',dtype={'block_id':str})
    excluded=pd.read_csv(work/'CONTEXT_UNSUPPORTED_BLOCKS.csv',dtype={'block_id':str})
    union=pd.read_csv(work/'CONTEXT_SUPPORT_UNION_BY_BLOCK.csv.gz',dtype={'block_id':str})
    support=pd.read_csv(work/'BLOCK_LEVEL_CONTEXT_SUPPORT_CENSUS.csv.gz',dtype={'block_id':str})
    require(len(complete)==23618 and complete.block_id.is_unique,'Context-complete candidate registry mismatch')
    require(len(excluded)==92 and excluded.block_id.is_unique,'Support exclusion registry mismatch')
    require(len(union)==23710 and union.block_id.is_unique and len(support)==23710 and support.block_id.is_unique,'Census block registry mismatch')
    c=set(complete.block_id);e=set(excluded.block_id);u=set(union.block_id)
    require(not(c&e) and c|e==u,'Context-complete/excluded partition is not exact')
    require(set(union.loc[union.context_complete_at_frozen_0_99.astype(bool),'block_id'])==c,'Union complete flag disagrees with candidate registry')
    require(set(union.loc[~union.context_complete_at_frozen_0_99.astype(bool),'block_id'])==e,'Union exclusion flag disagrees with exclusion registry')
    source=read_json(work/'SOURCE_SUPPORT_CENSUS_MANIFEST.json')
    require(source.get('base_package_aggregate_sha256')==BASE_AGG,'Census source manifest base aggregate mismatch')
    require(float(source.get('frozen_valid_area_threshold'))==THRESHOLD,'Census source manifest threshold mismatch')
    require(source.get('governing_layer_ids')==GOVERNING,'Census source governing layer IDs mismatch')
    require(source.get('credential_values_recorded') is False,'Credential serialization detected in census manifest')
    return {'root':work,'report':rep,'verification':ver,'complete':complete,'excluded':excluded,'union':union,'support':support,'source_manifest':source}
