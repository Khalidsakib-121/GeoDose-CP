from __future__ import annotations
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from geodose_stage4.io import require
from geodose_stage4.runner import run
KEYS=['stage3a_output_zip','stage3b_output_zip','stage3b_source_root','stage3c_source_root','stage3d_d2_source_root','stage3d_d4_source_root','stage3d_d4_output_zip','stage3e_source_root','stage3e_output_zip','stage3f_output_zip']
def paths(args):
 p=ROOT/'configs/resolved_paths_windows.json';cfg=json.loads(p.read_text()) if p.is_file() else json.loads((ROOT/'configs/local_paths_windows.json').read_text());out={}
 for k in KEYS:
  v=getattr(args,k,None)
  if v:out[k]=Path(v);continue
  x=cfg.get(k);cand=[Path(q) for q in x] if isinstance(x,list) else [Path(x)];found=[q for q in cand if q.exists()];require(found,f'PATH_NOT_RESOLVED:{k}');out[k]=found[0]
 return out
def main():
 ap=argparse.ArgumentParser(description='GeoDose Stage4 full controlled production + prospectively frozen F06 + ablation/efficiency')
 for k in KEYS:ap.add_argument('--'+k.replace('_','-'),dest=k)
 ap.add_argument('--overwrite',action='store_true',help='Clear outputs/checkpoints and restart. Default is interruption-safe resume.')
 ap.add_argument('--qa-smoke',action='store_true',help=argparse.SUPPRESS)
 a=ap.parse_args();ps=paths(a);print('Resolved frozen inputs:');[print(f'  {k}: {ps[k]}') for k in KEYS]
 r=run(ROOT,ps,ROOT/'outputs_stage4',overwrite=a.overwrite,qa_smoke=a.qa_smoke)
 print('\nSTAGE4 RUN COMPLETE');print('QA smoke:',r['qa_smoke']);print('Scientific query rows:',r['result_rows']);print('Support rows:',r['support_rows']);print('False support:',r['false_support']);print('Computational failures:',r['computational_failures']);print('Runtime seconds:',round(r['runtime_seconds'],3));print('Do not start MineDoseBench until the verified Windows output is independently reviewed.')
if __name__=='__main__':
 try:main()
 except Exception as exc:print(f'\nFAILED: {type(exc).__name__}: {exc}');traceback.print_exc();print('\nFAILED. Frozen upstream stages and Stage3F thresholds must not be changed. Resume after correcting only Stage4.');sys.exit(2)
