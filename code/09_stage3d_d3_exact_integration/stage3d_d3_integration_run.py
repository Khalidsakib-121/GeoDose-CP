from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
for p in [ROOT/'vendor', ROOT/'src']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))

from geodose_stage3d_d3_integration.io import IntegrationError, write_json
from geodose_stage3d_d3_integration.provenance import source_inventory, output_manifest
from geodose_stage3d_d3_integration.runner import run


def load_cfg(): return json.loads((ROOT/'configs'/'local_paths_windows.json').read_text(encoding='utf-8'))

def resolve_one(key, explicit, want_dir=False):
    check = (lambda p: p.is_dir()) if want_dir else (lambda p: p.is_file())
    if explicit:
        p=Path(explicit)
        if check(p):
            return p
        raise IntegrationError(f'Explicit path does not exist for {key}: {p}')
    tried=[]
    for raw in load_cfg().get(key,[]):
        p=Path(raw)
        tried.append(str(p))
        if check(p):
            return p
    flag='--'+key.replace("_","-")
    tried_text='\n  - '.join(tried) if tried else '(no configured candidates)'
    raise IntegrationError(
        f'Could not locate {key}. Tried:\n  - {tried_text}\n'
        f'Edit configs/local_paths_windows.json or pass {flag}.'
    )

def main():
    ap=argparse.ArgumentParser()
    for key in ['stage3a_output_zip','stage3b_output_zip','stage3c_output_zip','stage3d_d1_output_zip','stage3d_d2_output_zip','stage3d_d3_m3_output_zip','stage3d_d3_m4_output_zip']:
        ap.add_argument('--'+key.replace('_','-'),dest=key)
    ap.add_argument('--stage3c-source-root'); ap.add_argument('--stage3d-d2-source-root')
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    paths={}
    for key in ['stage3a_output_zip','stage3b_output_zip','stage3c_output_zip','stage3d_d1_output_zip','stage3d_d2_output_zip','stage3d_d3_m3_output_zip','stage3d_d3_m4_output_zip']:
        paths[key]=resolve_one(key,getattr(args,key),False)
    paths['stage3c_source_root']=resolve_one('stage3c_source_root',args.stage3c_source_root,True)
    paths['stage3d_d2_source_root']=resolve_one('stage3d_d2_source_root',args.stage3d_d2_source_root,True)
    print('Resolved frozen inputs:')
    for key in sorted(paths):
        print(f'  {key}: {paths[key]}')
    out=ROOT/'outputs_stage3d_d3_integration'
    result=run(ROOT,paths,out,args.overwrite)
    write_json(out/'stage3d_d3_integration_source_hashes.json',source_inventory(ROOT))
    write_json(out/'stage3d_d3_integration_manifest.json',output_manifest(out,{'stage3d_d3_integration_manifest.json','STAGE3D_D3_INTEGRATION_VERIFICATION.json'}))
    print('\nSTAGE 3D D3 M1-M6 EXACT INTEGRATION: IMPLEMENTATION RUN PASSED')
    print('Stage3C baseline replay groups:',result['replay_summary']['groups_replayed'])
    print('S4 common-interface candidates:',result['meta']['candidate_count'])
    print('Principal methods: M1 M2 M3 M4 M5 M6')
    print('R1/R2/R4: PASS; R3: DEFERRED TO STAGE3E')
    print('M6 adapter: accepted D2 exact-reference output; new-data pilot M6 engine: NOT YET')
    print('Pilot/production evidence: NO')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'\nFAILED: {type(exc).__name__}: {exc}',file=sys.stderr); raise
