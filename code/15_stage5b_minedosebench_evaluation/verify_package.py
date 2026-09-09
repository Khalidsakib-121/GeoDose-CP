from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parent
man=ROOT/'PACKAGE_MANIFEST.json'
if not man.exists():raise SystemExit('PACKAGE_MANIFEST.json missing')
m=json.loads(man.read_text(encoding='utf-8'));rows=m['files'];h=hashlib.sha256()
for r in sorted(rows,key=lambda x:x['file']):
    p=ROOT/r['file']
    if not p.exists():raise SystemExit(f'PACKAGE FILE MISSING: {r["file"]}')
    if p.stat().st_size!=int(r['bytes']):raise SystemExit(f'PACKAGE SIZE MISMATCH: {r["file"]}')
    z=hashlib.sha256(p.read_bytes()).hexdigest()
    if z!=r['sha256']:raise SystemExit(f'PACKAGE SHA MISMATCH: {r["file"]}')
    h.update(f"{r['file']}\t{int(r['bytes'])}\t{r['sha256']}\n".encode())
agg=h.hexdigest()
if agg!=m['aggregate_sha256']:raise SystemExit(f'PACKAGE AGGREGATE MISMATCH: {agg}')
print(f'PACKAGE MANIFEST VERIFIED: {len(rows)} immutable files; aggregate {agg}')
