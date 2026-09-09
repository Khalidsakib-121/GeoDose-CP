from __future__ import annotations
import json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs_stage3d_d4'; DEST=ROOT/'GeoDose_Stage3D_D4_M6_NEWDATA_NUISANCE_OUTPUTS.zip'
ver=json.loads((OUT/'STAGE3D_D4_VERIFICATION.json').read_text(encoding='utf-8'))
if ver.get('status')!='verified_complete' or int(ver.get('failed_checks',1))!=0:
    raise SystemExit('Refusing to package unverified D4 outputs')
manifest=json.loads((OUT/'stage3d_d4_manifest.json').read_text(encoding='utf-8'))
allowed=[x['name'] for x in manifest['files']]+['stage3d_d4_manifest.json','STAGE3D_D4_VERIFICATION.json']
with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
    for name in sorted(set(allowed)):
        fp=OUT/name
        if not fp.is_file(): raise SystemExit(f'Missing intended output {name}')
        zf.write(fp,arcname=name)
with zipfile.ZipFile(DEST) as zf:
    bad=zf.testzip()
    if bad: raise SystemExit(f'Output ZIP CRC failure: {bad}')
print('PACKAGED',DEST.name,'members',len(set(allowed)))
