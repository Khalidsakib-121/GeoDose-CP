from __future__ import annotations
import argparse, hashlib, json, sys, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def find_one(key, cfg, expected=None, is_root=False):
 tried=[]
 for raw in cfg.get(key,[]):
  p=Path(raw); tried.append(str(p))
  if (p.is_dir() if is_root else p.is_file()):
   if expected is None or is_root or sha(p)==expected: return p
 # controlled recursive search on D:\ for moved accepted packages
 patterns={
  'stage3a_output_zip':'GeoDose_Stage3A_OUTPUTS.zip','stage3b_output_zip':'GeoDose_Stage3B_OUTPUTS.zip',
  'stage3d_d4_output_zip':'GeoDose_Stage3D_D4_M6_NEWDATA_NUISANCE_OUTPUTS.zip',
  'stage3e_output_zip':'GeoDose_Stage3E_N2_N3_SCALABLE_CERTIFICATION_OUTPUTS.zip'}
 if not is_root and key in patterns:
  base=Path('D:/')
  if base.exists():
   for p in base.glob('GeoDose*/**/'+patterns[key]):
    tried.append(str(p))
    try:
     if p.is_file() and (expected is None or sha(p)==expected): return p
    except OSError: pass
 raise RuntimeError(f'Could not locate accepted {key}. Tried: '+ ' | '.join(tried[:12]))

def source_matches(root, lock):
 if not root.is_dir(): return False
 for rel,exp in lock['files'].items():
  q=root/rel
  try:
   if not q.is_file() or sha(q)!=exp: return False
  except OSError:
   return False
 return True

def find_source_root(key,cfg,lock):
 tried=[]
 for raw in cfg.get(key,[]):
  p=Path(raw); tried.append(str(p))
  if source_matches(p,lock): return p
 # Hash-validated fallback if an accepted package was extracted one level differently.
 # Search only GeoDose-prefixed trees on D:\ and identify a root from a frozen marker file.
 base=Path('D:/')
 if base.exists() and lock.get('files'):
  marker=sorted(lock['files'])[0]; parts=Path(marker).parts
  try:
   candidates=base.glob('GeoDose*/**/'+marker.replace('\\','/'))
   for q in candidates:
    try:
     r=q
     for _ in parts: r=r.parent
     tried.append(str(r))
     if source_matches(r,lock): return r
    except OSError: pass
  except OSError: pass
 raise RuntimeError(f'Could not locate accepted {key} source tree with frozen byte hashes. Tried: '+ ' | '.join(tried[:12]))

def main():
 cfg=json.loads((ROOT/'configs/local_paths_windows.json').read_text())
 hashes=json.loads((ROOT/'configs/expected_upstream_hashes.json').read_text())
 locks=json.loads((ROOT/'configs/source_locks.json').read_text())
 resolved={}
 print('[PRECHECK] Verifying accepted archives and exact source trees before environment setup...')
 for key,expected in hashes.items():
  p=find_one(key,cfg,expected,False); got=sha(p)
  if got!=expected: raise RuntimeError(f'{key} SHA mismatch')
  with zipfile.ZipFile(p) as z:
   bad=z.testzip()
   if bad: raise RuntimeError(f'{key} ZIP CRC failed at {bad}')
   n=len(z.namelist())
  resolved[key]=str(p); print(f'  PASS {key}: {p} [{n} ZIP members]')
 for key,lock in locks.items():
  p=find_source_root(key,cfg,lock); bad=[]
  for rel,exp in lock['files'].items():
   q=p/rel
   if not q.is_file() or sha(q)!=exp: bad.append(rel)
  if bad: raise RuntimeError(f'{key} source mismatch: {bad[:8]}')
  resolved[key]=str(p); print(f"  PASS {key}: {len(lock['files'])}/{len(lock['files'])} files")
 (ROOT/'configs/resolved_paths_windows.json').write_text(json.dumps(resolved,indent=2)+'\n')
 print('STAGE3F WINDOWS INPUT PREFLIGHT PASSED before virtual-environment creation.')
if __name__=='__main__':
 try: main()
 except Exception as exc:
  print('\nPRECHECK FAILED:',exc); sys.exit(2)
