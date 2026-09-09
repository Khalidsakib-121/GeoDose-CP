from __future__ import annotations
"""Stdlib-only Windows provenance preflight for Stage3E, before venv creation."""
import glob, hashlib, json, os, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class PreflightError(RuntimeError): pass

def sha256_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def loadj(p):
 o=json.loads(Path(p).read_text(encoding='utf-8'))
 if not isinstance(o,dict): raise PreflightError(f'Expected JSON object: {p}')
 return o
def zipcheck(p):
 with zipfile.ZipFile(p) as z:
  bad=z.testzip()
  if bad: raise PreflightError(f'ZIP CRC failure {p}: {bad}')
  return len(z.namelist())
def readjzip(p,suffix):
 with zipfile.ZipFile(p) as z:
  m=[n for n in z.namelist() if n.replace('\\','/').endswith(suffix)]
  if len(m)!=1: raise PreflightError(f'Expected one {suffix} in {p}; found {len(m)}')
  return json.loads(z.read(m[0]).decode('utf-8'))
def resolve_file(key,cfg,fb,expected):
 tried=[]; wrong=[]
 for raw in cfg.get(key,[]):
  p=Path(str(raw)); tried.append(str(p))
  if p.is_file():
   h=sha256_file(p)
   if h.lower()==expected[key].lower(): return p
   wrong.append((str(p),h))
 for pattern in fb.get(key,[]):
  for raw in sorted(glob.glob(str(pattern),recursive=True),key=str.lower):
   p=Path(raw)
   if str(p) in tried or not p.is_file(): continue
   tried.append(str(p)); h=sha256_file(p)
   if h.lower()==expected[key].lower(): return p
   wrong.append((str(p),h))
 msg='\n'.join('  - '+x for x in tried) or '  - (none)'
 if wrong: msg+='\nWrong-hash existing candidates rejected:\n'+'\n'.join(f'  - {p}: {h}' for p,h in wrong)
 raise PreflightError(f'Could not locate accepted {key}. Tried:\n{msg}')
def resolve_dir(key,cfg):
 tried=[]
 for raw in cfg.get(key,[]):
  p=Path(str(raw)); tried.append(str(p))
  if p.is_dir(): return p
 raise PreflightError(f'Could not locate {key}. Tried:\n'+'\n'.join('  - '+x for x in tried))
def verify_tree(root,files,label='source'):
 bad=[]; miss=[]
 for rel,exp in sorted(files.items()):
  p=root/Path(rel)
  if not p.is_file(): miss.append(rel); continue
  h=sha256_file(p)
  if h.lower()!=str(exp).lower(): bad.append((rel,exp,h))
 if miss or bad: raise PreflightError(f'{label} mismatch missing={miss} mismatched={bad}')
 return len(files)
def main():
 if os.name!='nt': print('STAGE3E WINDOWS PREFLIGHT: non-Windows host; skipped operational path resolution.'); return 0
 cfg=loadj(ROOT/'configs'/'local_paths_windows.json'); fb=loadj(ROOT/'configs'/'fallback_globs_windows.json'); exp=loadj(ROOT/'configs'/'expected_upstream_hashes.json')
 keys=['stage3a_output_zip','stage3b_output_zip','stage3d_d2_output_zip','stage3d_d4_output_zip']
 if set(keys)!=set(exp): raise PreflightError('Frozen upstream registry keys differ from Stage3E contract')
 resolved={}
 for k in keys:
  resolved[k]=resolve_file(k,cfg,fb,exp); print(f'  PASS {k}: {resolved[k]} [{zipcheck(resolved[k])} ZIP members]')
 d2root=resolve_dir('stage3d_d2_source_root',cfg); inv=readjzip(resolved['stage3d_d2_output_zip'],'stage3d_d2_source_hashes.json')
 if not isinstance(inv.get('files'),dict): raise PreflightError('Unexpected D2 source inventory schema')
 n=verify_tree(d2root,inv['files'],'D2 source'); print(f'  PASS stage3d_d2_source_root: {n}/{n} files')
 d4v=readjzip(resolved['stage3d_d4_output_zip'],'STAGE3D_D4_VERIFICATION.json')
 if d4v.get('status')!='verified_complete' or int(d4v.get('failed_checks',-1))!=0: raise PreflightError('Accepted D4 output verification is not complete')
 print('  PASS accepted D4 verification: verified_complete, failed_checks=0')
 d4root=resolve_dir('stage3d_d4_source_root',cfg); d4inv=readjzip(resolved['stage3d_d4_output_zip'],'stage3d_d4_source_hashes.json')
 d4files=d4inv.get('files')
 if isinstance(d4files,list):
  if not all(isinstance(x,dict) and 'path' in x and 'sha256' in x for x in d4files): raise PreflightError('Unexpected D4 source inventory list schema')
  d4expected={str(x['path']):str(x['sha256']) for x in d4files}
 elif isinstance(d4files,dict): d4expected={str(k):str(v) for k,v in d4files.items()}
 else: raise PreflightError('Unexpected D4 source inventory schema')
 n4=verify_tree(d4root,d4expected,'D4 source'); print(f'  PASS stage3d_d4_source_root: {n4}/{n4} files')
 print('STAGE3E WINDOWS INPUT PREFLIGHT PASSED before virtual-environment creation.')
 return 0
if __name__=='__main__':
 try: raise SystemExit(main())
 except Exception as exc:
  print(f'STAGE3E WINDOWS INPUT PREFLIGHT FAILED: {type(exc).__name__}: {exc}',file=sys.stderr); raise SystemExit(1)
