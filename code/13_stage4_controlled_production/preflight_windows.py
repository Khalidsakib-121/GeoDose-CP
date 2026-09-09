from __future__ import annotations
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def find_file(key,cfg,expected):
 tried=[]
 for raw in cfg.get(key,[]):
  p=Path(raw);tried.append(str(p))
  if p.is_file() and sha(p)==expected:return p
 names={'stage3a_output_zip':'GeoDose_Stage3A_OUTPUTS.zip','stage3b_output_zip':'GeoDose_Stage3B_OUTPUTS.zip','stage3d_d4_output_zip':'GeoDose_Stage3D_D4_M6_NEWDATA_NUISANCE_OUTPUTS.zip','stage3e_output_zip':'GeoDose_Stage3E_N2_N3_SCALABLE_CERTIFICATION_OUTPUTS.zip','stage3f_output_zip':'GeoDose_Stage3F_PILOT_S1_S10_20REP_THRESHOLD_FREEZE_OUTPUTS.zip'}
 base=Path('D:/')
 if base.exists():
  for p in base.glob('GeoDose*/**/'+names[key]):
   tried.append(str(p))
   try:
    if p.is_file() and sha(p)==expected:return p
   except OSError:pass
 raise RuntimeError(f'Could not locate hash-accepted {key}. Tried: '+' | '.join(tried[:15]))

def source_matches(root,lock):
 if not root.is_dir():return False
 for rel,h in lock['files'].items():
  p=root/rel
  try:
   if not p.is_file() or sha(p)!=h:return False
  except OSError:return False
 return True

def find_source(key,cfg,lock):
 tried=[]
 for raw in cfg.get(key,[]):
  p=Path(raw);tried.append(str(p))
  if source_matches(p,lock):return p
 base=Path('D:/')
 if base.exists() and lock.get('files'):
  marker=sorted(lock['files'])[0];parts=Path(marker).parts
  for q in base.glob('GeoDose*/**/'+marker.replace('\\','/')):
   r=q
   for _ in parts:r=r.parent
   tried.append(str(r))
   if source_matches(r,lock):return r
 raise RuntimeError(f'Could not locate byte-accepted {key}. Tried: '+' | '.join(tried[:15]))

def main():
 cfg=json.loads((ROOT/'configs/local_paths_windows.json').read_text());hashes=json.loads((ROOT/'configs/expected_upstream_hashes.json').read_text());locks=json.loads((ROOT/'configs/source_locks.json').read_text());resolved={}
 print('[PRECHECK] Verifying Stage3F freeze, accepted upstream archives, and exact source trees BEFORE environment setup...')
 for key,expected in hashes.items():
  p=find_file(key,cfg,expected)
  with zipfile.ZipFile(p) as z:
   bad=z.testzip()
   if bad:raise RuntimeError(f'{key} ZIP CRC failed at {bad}')
   n=len(z.namelist())
  resolved[key]=str(p);print(f'  PASS {key}: {p} [{n} ZIP members]')
 for key,lock in locks.items():
  p=find_source(key,cfg,lock);resolved[key]=str(p);print(f"  PASS {key}: {len(lock['files'])}/{len(lock['files'])} files")
 # Validate that Stage3F really froze the exact thresholds Stage4 is forbidden to retune.
 with zipfile.ZipFile(resolved['stage3f_output_zip']) as z:
  names=[n for n in z.namelist() if n.replace('\\','/').endswith('stage3f_operational_thresholds_FROZEN.json')]
  if len(names)!=1:raise RuntimeError('Stage3F frozen threshold member is not unique')
  t=json.loads(z.read(names[0]).decode())
 if (float(t['minimum_ess']),float(t['maximum_normalized_weight']),int(t['minimum_graph_safe_count']))!=(3.0,0.5,20):raise RuntimeError('Stage3F frozen thresholds do not match accepted freeze')
 resolved_path=ROOT/'configs/resolved_paths_windows.json';resolved_path.write_text(json.dumps(resolved,indent=2)+'\n')
 print('STAGE4 WINDOWS INPUT PREFLIGHT PASSED. F06 structural QA will run before any production outcome evaluation.')
if __name__=='__main__':
 try:main()
 except Exception as exc:print('\nPRECHECK FAILED:',exc);sys.exit(2)
