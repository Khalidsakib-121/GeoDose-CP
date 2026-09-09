from __future__ import annotations
import hashlib
from pathlib import Path
from .io import sha256_file
def source_inventory(root:Path)->dict:
    root=Path(root);rows=[];skip={'.venv','outputs_stage4','checkpoints','__pycache__','.git'}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or any(part in skip for part in p.parts) or p.suffix.lower()=='.pyc':continue
        rel=p.relative_to(root).as_posix()
        if rel=='configs/resolved_paths_windows.json':continue
        if rel.startswith('GeoDose_Stage4_PRODUCTION') and rel.endswith('_OUTPUTS.zip'):continue
        rows.append({'path':rel,'sha256':sha256_file(p),'size':p.stat().st_size})
    agg=hashlib.sha256('\n'.join(f"{r['path']}|{r['sha256']}|{r['size']}" for r in rows).encode()).hexdigest()
    return {'file_count':len(rows),'aggregate_sha256':agg,'files':rows}
def output_manifest(out:Path,exclude:set[str]|None=None)->dict:
    exclude=exclude or set();rows=[]
    for p in sorted(Path(out).iterdir()):
        if p.is_file() and p.name not in exclude:rows.append({'path':p.name,'sha256':sha256_file(p),'size':p.stat().st_size})
    return {'file_count':len(rows),'files':rows}
