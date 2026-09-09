from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs_stage3e'; DEST=ROOT/'GeoDose_Stage3E_N2_N3_SCALABLE_CERTIFICATION_OUTPUTS.zip'
ver=json.loads((OUT/'STAGE3E_VERIFICATION.json').read_text(encoding='utf-8'))
if ver.get('status')!='verified_complete' or int(ver.get('failed_checks',-1))!=0: raise SystemExit('Stage3E outputs are not verified_complete; refusing to package.')
files=sorted([p for p in OUT.iterdir() if p.is_file()])
with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:
        info=zipfile.ZipInfo(p.name,date_time=(2026,8,9,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=(0o644&0xFFFF)<<16
        z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
with zipfile.ZipFile(DEST) as z:
    bad=z.testzip()
    if bad: raise SystemExit(f'Output ZIP integrity failure: {bad}')
h=hashlib.sha256(DEST.read_bytes()).hexdigest()
print('PACKAGED VERIFIED STAGE3E OUTPUTS:',DEST.name)
print('SHA-256:',h)
print('Files:',len(files))
