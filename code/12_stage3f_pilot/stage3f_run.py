from __future__ import annotations
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from geodose_stage3f.io import Stage3FError, require
from geodose_stage3f.runner import run

KEYS=['stage3a_output_zip','stage3b_output_zip','stage3b_source_root','stage3c_source_root','stage3d_d2_source_root','stage3d_d4_source_root','stage3d_d4_output_zip','stage3e_source_root','stage3e_output_zip']

def _paths(args):
    resolved=ROOT/'configs/resolved_paths_windows.json'
    cfg=json.loads(resolved.read_text()) if resolved.is_file() else json.loads((ROOT/'configs/local_paths_windows.json').read_text())
    out={}
    for k in KEYS:
        v=getattr(args,k,None)
        if v: out[k]=Path(v)
        else:
            x=cfg.get(k)
            if isinstance(x,list):
                found=[Path(p) for p in x if Path(p).exists()]; require(len(found)>=1,f'PATH_NOT_RESOLVED:{k}'); out[k]=found[0]
            else: out[k]=Path(x)
    return out

def main():
    ap=argparse.ArgumentParser(description='GeoDose Stage3F preregistered S1-S10 20-rep pilot + outcome-blind operational threshold freeze')
    for k in KEYS: ap.add_argument('--'+k.replace('_','-'),dest=k)
    ap.add_argument('--overwrite',action='store_true')
    a=ap.parse_args(); paths=_paths(a)
    print('Resolved frozen inputs:')
    for k in KEYS: print(f'  {k}: {paths[k]}')
    result=run(ROOT,paths,ROOT/'outputs_stage3f',overwrite=a.overwrite)
    print('\nSTAGE3F PILOT RUN COMPLETE')
    print('Pilot scale ready for independent review:',result['pilot_scale_ready'])
    print('Frozen threshold candidate:',result['thresholds'])
    print('Scientific query rows:',result['results'])
    print('Support rows:',result['support_rows'])
    print('Runtime seconds:',round(result['runtime_seconds'],3))
    print('Production remains BLOCKED until this output is independently reviewed.')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'\nFAILED: {type(exc).__name__}: {exc}')
        traceback.print_exc()
        print('\nFAILED. Stop here. Do not change frozen upstream stages and do not start production.')
        sys.exit(2)
