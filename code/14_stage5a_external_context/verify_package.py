from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'PRODUCTION_PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
bad=[]
for r in m['immutable_files']:
 p=ROOT/r['file']
 if not p.exists() or p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:bad.append(r['file'])
if bad:raise SystemExit('PACKAGE MANIFEST FAILURE: '+repr(bad[:10]))
agg=hashlib.sha256(''.join(r['file']+r['sha256'] for r in sorted(m['immutable_files'],key=lambda x:x['file'])).encode()).hexdigest()
if agg!=m['aggregate_sha256']:raise SystemExit('PACKAGE AGGREGATE FAILURE')
print(f"PRODUCTION PACKAGE VERIFIED: {m['immutable_file_count']} immutable files; aggregate {agg}")
print('User-editable file excluded from immutable hash lock: configs/local_paths_windows.json')
