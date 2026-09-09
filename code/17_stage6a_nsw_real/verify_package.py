from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
for r in m['files']:
    p=ROOT/r['path']
    if not p.exists():raise SystemExit('PACKAGE MISSING '+r['path'])
    b=p.read_bytes()
    if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise SystemExit('PACKAGE HASH/SIZE FAIL '+r['path'])
agg=hashlib.sha256('\n'.join(f"{r['path']}|{r['bytes']}|{r['sha256']}" for r in m['files']).encode()).hexdigest()
if agg!=m['aggregate_sha256']:raise SystemExit('PACKAGE AGGREGATE FAIL')
print(f"PACKAGE MANIFEST VERIFIED: {len(m['files'])} immutable files; aggregate {agg}")
