from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from geodose_stage3e.io import Stage3EError, sha256_file
from geodose_stage3e.runner import run

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--overwrite',action='store_true')
 for k in ['stage3a_output_zip','stage3b_output_zip','stage3d_d2_output_zip','stage3d_d2_source_root','stage3d_d4_output_zip','stage3d_d4_source_root']:
  ap.add_argument('--'+k.replace('_','-'),dest=k)
 args=ap.parse_args(); cfg=json.loads((ROOT/'configs'/'local_paths_windows.json').read_text()); fb=json.loads((ROOT/'configs'/'fallback_globs_windows.json').read_text()); exp=json.loads((ROOT/'configs'/'expected_upstream_hashes.json').read_text())
 def file_one(key,explicit):
  candidates=[]
  if explicit: candidates.append(Path(explicit))
  candidates += [Path(x) for x in cfg.get(key,[])]
  candidates += [Path(x) for pat in fb.get(key,[]) for x in sorted(glob.glob(pat,recursive=True),key=str.lower)]
  wrong=[]
  for p in candidates:
   if p.is_file():
    h=sha256_file(p)
    if h.lower()==exp[key].lower(): return p
    wrong.append((str(p),h))
  raise Stage3EError(f'Could not locate accepted {key}; wrong-hash candidates={wrong}; tried={[str(x) for x in candidates]}')
 def dir_one(key,explicit):
  for p in ([Path(explicit)] if explicit else [])+[Path(x) for x in cfg.get(key,[])]:
   if p.is_dir(): return p
  raise Stage3EError(f'Could not locate {key}')
 paths={}
 for k in ['stage3a_output_zip','stage3b_output_zip','stage3d_d2_output_zip','stage3d_d4_output_zip']: paths[k]=file_one(k,getattr(args,k))
 paths['stage3d_d2_source_root']=dir_one('stage3d_d2_source_root',args.stage3d_d2_source_root)
 paths['stage3d_d4_source_root']=dir_one('stage3d_d4_source_root',args.stage3d_d4_source_root)
 print('Resolved frozen inputs:')
 for k,p in paths.items(): print(f'  {k}: {p}')
 result=run(ROOT,paths,ROOT/'outputs_stage3e',args.overwrite)
 print('\nSTAGE 3E N2/N3 SCALABLE CERTIFICATION RUN COMPLETE')
 print('Pilot readiness:',result['pilot_readiness']['ready_for_preregistered_20rep_pilot'])
 print('Primary neighborhood m:',result['pilot_readiness']['primary_neighborhood_size'])
 print('Estimated nuisance certified:',result['pilot_readiness']['estimated_nuisance_certified'])
 print('Production run:',False)
if __name__=='__main__':
 try: main()
 except Exception as exc:
  print(f'\nFAILED: {type(exc).__name__}: {exc}',file=sys.stderr); raise
