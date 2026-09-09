from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs_stage3d_d3_integration'; DEST=ROOT/'GeoDose_Stage3D_D3_M1_M6_EXACT_INTEGRATION_OUTPUTS.zip'
ver=OUT/'STAGE3D_D3_INTEGRATION_VERIFICATION.json'
if not ver.is_file() or 'verified_complete' not in ver.read_text(encoding='utf-8'):
    raise SystemExit('Outputs are not verified_complete')
files=sorted(p for p in OUT.iterdir() if p.is_file())
with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files: z.write(p,p.name)
print('PACKAGED',DEST)
