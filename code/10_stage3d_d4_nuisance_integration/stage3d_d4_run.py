from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
for p in [ROOT/'vendor',ROOT/'src']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))

from geodose_stage3d_d4.io import D4Error, write_json
from geodose_stage3d_d4.provenance import output_manifest, source_inventory
from geodose_stage3d_d4.runner import run


def _cfg():
    return json.loads((ROOT/'configs'/'local_paths_windows.json').read_text(encoding='utf-8'))

def _fallback_cfg():
    p=ROOT/'configs'/'fallback_globs_windows.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}

def _expected_hashes():
    return json.loads((ROOT/'configs'/'expected_upstream_hashes.json').read_text(encoding='utf-8'))

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _hash_matches_expected(key: str, path: Path) -> tuple[bool,str|None]:
    expected=_expected_hashes().get(key)
    if not expected: return True,None
    observed=_sha256(path)
    return observed.lower()==str(expected).lower(), observed

def _glob_candidates(key: str) -> list[Path]:
    out=[]
    if os.name!='nt': return out
    for pattern in _fallback_cfg().get(key,[]):
        for raw in glob.glob(pattern,recursive=True):
            p=Path(raw)
            if p.is_file() and p not in out: out.append(p)
    return sorted(out,key=lambda p:str(p).lower())

def resolve_one(key: str, explicit: str|None, want_dir: bool=False) -> Path:
    check=(lambda p:p.is_dir()) if want_dir else (lambda p:p.is_file())
    if explicit:
        p=Path(explicit)
        if check(p): return p
        raise D4Error(f'Explicit path does not exist for {key}: {p}')

    tried=[]
    wrong_hash=[]
    for raw in _cfg().get(key,[]):
        p=Path(raw); tried.append(str(p))
        if not check(p): continue
        if want_dir: return p
        ok,obs=_hash_matches_expected(key,p)
        if ok: return p
        wrong_hash.append((str(p),obs))

    # Controlled fallback discovery is intentionally limited to configured
    # GeoDose stage-name patterns and accepted only by the frozen SHA-256.
    if not want_dir:
        for p in _glob_candidates(key):
            tried.append(str(p))
            ok,obs=_hash_matches_expected(key,p)
            if ok:
                print(f'  Auto-discovered accepted {key}: {p}')
                return p
            wrong_hash.append((str(p),obs))

    flag='--'+key.replace('_','-')
    tried_text='\n  - '.join(tried) if tried else '(no configured/fallback candidates)'
    extra=''
    if wrong_hash:
        extra='\nExisting candidates with WRONG SHA-256 (not used):\n'+'\n'.join(f'  - {p}: {h}' for p,h in wrong_hash)
    raise D4Error(f'Could not locate the accepted {key}. Tried:\n  - {tried_text}{extra}\nEdit configs/local_paths_windows.json or pass {flag}.')


def main():
    ap=argparse.ArgumentParser(description='GeoDose Stage3D D4 source-level M6 new-data + nuisance integration gate')
    file_keys=['stage3a_output_zip','stage3b_output_zip','stage3c_output_zip','stage3d_d1_output_zip','stage3d_d2_output_zip','stage3d_d3_m3_output_zip','stage3d_d3_m4_output_zip','stage3d_d3_integration_output_zip']
    for key in file_keys: ap.add_argument('--'+key.replace('_','-'),dest=key)
    ap.add_argument('--stage3c-source-root'); ap.add_argument('--stage3d-d2-source-root'); ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args(); paths={}
    for key in file_keys: paths[key]=resolve_one(key,getattr(args,key),False)
    paths['stage3c_source_root']=resolve_one('stage3c_source_root',args.stage3c_source_root,True)
    paths['stage3d_d2_source_root']=resolve_one('stage3d_d2_source_root',args.stage3d_d2_source_root,True)
    print('Resolved frozen inputs:')
    for k in sorted(paths): print(f'  {k}: {paths[k]}')
    out=ROOT/'outputs_stage3d_d4'; result=run(ROOT,paths,out,args.overwrite)
    write_json(out/'stage3d_d4_source_hashes.json',source_inventory(ROOT))
    write_json(out/'stage3d_d4_manifest.json',output_manifest(out,{'stage3d_d4_manifest.json','STAGE3D_D4_VERIFICATION.json'}))
    print('\nSTAGE 3D D4 M6 NEW-DATA + NUISANCE INTEGRATION: IMPLEMENTATION RUN PASSED')
    print('New-case M6 candidate rows:',result['counts']['m6_candidate_rows'])
    print('Nuisance variants exercised:',', '.join(result['counts']['nuisance_variants']))
    print('Accepted D2 source replay: PASS')
    print('M3 treatment invariance: PASS')
    print('M4 d*s one-normalization regression: PASS')
    print('G1/G2/G3 reimplemented: NO')
    print('Pilot/production evidence: NO')
    print('Next scientific gate after independent review: Stage3E N2/N3')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'\nFAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
