from pathlib import Path
import sys,hashlib,zipfile
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from geodose_stage6a.common import require,read_json,write_json,sha256_file
def main():
    out=ROOT/'STAGE6A_NSW_REAL_OUTPUTS';v=read_json(out/'STAGE6A_VERIFICATION.json');require(v['status']=='STAGE6A_VERIFIED_COMPLETE','Stage6A not verified')
    zpath=out/'GeoDose_Stage6A_NSW_REAL_DEMONSTRATION_RESULTS.zip';mp=out/'STAGE6A_RELEASE_MANIFEST.json'
    rows=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p not in {zpath,mp}:
            rows.append({'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    agg=hashlib.sha256('\n'.join(f"{r['path']}|{r['bytes']}|{r['sha256']}" for r in rows).encode()).hexdigest();write_json(mp,{'files':rows,'file_count':len(rows),'aggregate_sha256':agg})
    fixed=(2026,8,11,0,0,0)
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted([x for x in out.rglob('*') if x.is_file() and x!=zpath],key=lambda x:x.relative_to(out).as_posix()):
            info=zipfile.ZipInfo(p.relative_to(out).as_posix(),date_time=fixed);info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16;z.writestr(info,p.read_bytes())
    # Re-open independently and reconcile release manifest.
    with zipfile.ZipFile(zpath) as z:
        require(z.testzip() is None,'Release ZIP CRC failure');m=__import__('json').loads(z.read('STAGE6A_RELEASE_MANIFEST.json'))
        for r in m['files']:
            b=z.read(r['path']);require(len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],'Release ZIP manifest mismatch '+r['path'])
    print('Stage6A production result bundle:',zpath);print('SHA-256:',sha256_file(zpath));print('Release manifest:',len(rows),'files; aggregate',agg)
if __name__=='__main__':main()
