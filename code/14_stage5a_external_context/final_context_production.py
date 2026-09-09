from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from src.production import run
from src.verify import verify
p=argparse.ArgumentParser();p.add_argument('--base-root',required=True);p.add_argument('--output-dir',default='');a=p.parse_args()
out=run(ROOT,Path(a.base_root),Path(a.output_dir) if a.output_dir else None)
status,checks=verify(ROOT,Path(a.base_root),out)
print('\nFINAL CONTEXT PRODUCTION PASSED')
print(json.dumps(status,indent=2))
print('Final bundle:',out/'GeoDose_Stage5A_EXTERNAL_CONTEXT_FINAL_OUTPUTS.zip')
