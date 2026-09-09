from pathlib import Path
import sys,json,shutil,tempfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from geodose_stage5b.common import require,read_json,sha256_file,file_manifest,aggregate_manifest,write_json,deterministic_zip

def main():
    out=ROOT/'STAGE5B_PRODUCTION_OUTPUTS';ver=read_json(out/'STAGE5B_VERIFICATION.json');require(ver['status']=='STAGE5B_VERIFIED_COMPLETE_PENDING_INDEPENDENT_AUDIT','Stage5B verification not passed')
    rel=out/'GeoDose_Stage5B_MineDoseBench_PRODUCTION_RESULTS.zip'
    with tempfile.TemporaryDirectory(prefix='stage5b_release_') as td:
        st=Path(td)/'release';st.mkdir()
        for p in sorted(x for x in out.rglob('*') if x.is_file() and x.name not in {rel.name,'STAGE5B_RELEASE_MANIFEST.json'}):
            q=st/p.relative_to(out);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        rows=file_manifest(st);manifest={'stage':'Stage5B','version':ver['version'],'files':rows,'aggregate_sha256':aggregate_manifest(rows)};write_json(st/'STAGE5B_RELEASE_MANIFEST.json',manifest)
        deterministic_zip(st,rel)
    print('Stage5B production result bundle:',rel)
    print('SHA-256:',sha256_file(rel))
    print('Release manifest:',len(rows),'files; aggregate',manifest['aggregate_sha256'])
if __name__=='__main__':main()
