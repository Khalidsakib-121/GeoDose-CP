from __future__ import annotations
from pathlib import Path
import json, hashlib, zipfile
import pandas as pd
from .common import require,sha256_file,read_json,safe_extract_zip,aggregate_manifest

def _manifest_aggregate(rows):
    # Exact Stage5A v1.2 release rule from accepted package_minedosebench.py:
    # SHA256(concat(file + decimal_size + sha256)) in manifest row order.
    return hashlib.sha256(''.join(str(r['file'])+str(int(r.get('size',r.get('bytes'))))+str(r['sha256']) for r in rows).encode()).hexdigest()

def verify_authorities(root:Path,work:Path):
    exp=read_json(root/'configs/expected_authorities.json')
    rel=root/'inputs/GeoDose_MineDoseBench_v1_2_0_RELEASE_CANDIDATE.zip'
    audit=root/'inputs/Stage5A_MineDoseBench_v1_2_0_Independent_Freeze_Audit.md'
    require(sha256_file(rel)==exp['stage5a_release_zip_sha256'],'Stage5A release ZIP SHA-256 mismatch')
    require(sha256_file(audit)==exp['stage5a_audit_sha256'],'Stage5A independent audit SHA-256 mismatch')
    for name,h in exp['vendor_minedosebench_hashes'].items():
        require(sha256_file(root/'vendor/minedosebench'/name)==h,f'Frozen MineDoseBench vendor source drift: {name}')
    for name,h in exp.get('vendor_d3_exact_hashes',{}).items():
        require(sha256_file(root/'vendor/d3_exact'/name)==h,f'Accepted D3 exact-comparator vendor source drift: {name}')
    for name,h in exp.get('stage5a_contract_hashes',{}).items():
        require(sha256_file(root/'authorities'/name)==h,f'Frozen Stage5A evaluation/math contract drift: {name}')
    dest=work/'stage5a_release'
    marker=dest/'.verified_stage5a'
    if marker.exists() and marker.read_text().strip()==exp['stage5a_release_zip_sha256']:
        pass
    else:
        if dest.exists():
            import shutil; shutil.rmtree(dest)
        safe_extract_zip(rel,dest)
        man=json.loads((dest/'MINEDOSEBENCH_RELEASE_MANIFEST.json').read_text())
        rows=man.get('files',man.get('manifest',[]))
        require(len(rows)==40,f'Stage5A release manifest expected 40 files, found {len(rows)}')
        for r in rows:
            p=dest/r['file']; require(p.exists(),f'Stage5A release manifest missing {r["file"]}')
            require(p.stat().st_size==int(r.get('bytes',r.get('size'))),f'Stage5A release size mismatch {r["file"]}')
            require(sha256_file(p)==r['sha256'],f'Stage5A release hash mismatch {r["file"]}')
        declared=man.get('aggregate_sha256') or man.get('manifest_aggregate_sha256')
        calc=_manifest_aggregate(rows)
        require(calc==exp['stage5a_release_manifest_aggregate'],f'Stage5A release aggregate mismatch: {calc}')
        if declared: require(declared==calc,'Stage5A declared release aggregate mismatch')
        marker.write_text(exp['stage5a_release_zip_sha256'])
    # Frozen benchmark contract checks.
    rep=json.loads((dest/'MINEDOSEBENCH_FREEZE_REPORT.json').read_text())
    require(rep['frozen_context_complete_blocks']==23618 and rep['frozen_context_support_exclusions']==92,'Stage5A frozen substrate counts changed')
    require(abs(float(rep['frozen_valid_area_threshold'])-.99)<1e-15,'Stage5A valid-area threshold changed')
    sc=pd.read_csv(dest/'minedosebench_scenario_registry.csv'); seeds=pd.read_csv(dest/'minedosebench_seed_registry.csv'); targets=pd.read_csv(dest/'minedosebench_primary_target_registry.csv')
    require(len(sc)==27 and sc.case_id.nunique()==27,'Stage5A scenario registry is not 27 unique cases')
    require(len(seeds)==540,'Stage5A seed registry is not 540 rows')
    require(len(targets)==13500,'Stage5A target registry is not 13,500 rows')
    return dest,exp
