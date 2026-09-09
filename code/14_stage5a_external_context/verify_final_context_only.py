from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from src.verify import verify
p=argparse.ArgumentParser();p.add_argument('--base-root',required=True);p.add_argument('--output-dir',default='');a=p.parse_args()
out=Path(a.output_dir).resolve() if a.output_dir else ROOT/'FINAL_CONTEXT_PRODUCTION_OUTPUTS'
status,checks=verify(ROOT,Path(a.base_root),out);print(json.dumps(status,indent=2))
