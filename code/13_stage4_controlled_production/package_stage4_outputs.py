from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs_stage4';DEST=ROOT/'GeoDose_Stage4_PRODUCTION_F06_ABLATION_EFFICIENCY_OUTPUTS.zip'
def main():
    v=json.loads((OUT/'STAGE4_VERIFICATION.json').read_text())
    if v.get('status')!='verified_complete' or int(v.get('failed_checks',0))!=0:raise RuntimeError('Refusing to package unverified Stage4 outputs')
    fixed=(2026,8,9,0,0,0)
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(OUT.iterdir(),key=lambda x:x.name):
            if not p.is_file():continue
            info=zipfile.ZipInfo(p.name,date_time=fixed);info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(DEST) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Output ZIP CRC failure: {bad}')
        n=len(z.namelist())
    h=hashlib.sha256(DEST.read_bytes()).hexdigest();print('VERIFIED OUTPUT ZIP CREATED:',DEST);print('SHA-256:',h);print('Members:',n)
if __name__=='__main__':main()
