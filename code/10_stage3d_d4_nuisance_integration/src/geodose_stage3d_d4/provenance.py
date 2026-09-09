from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from .io import sha256_file


def _ok_file(fp: Path) -> bool:
    return fp.suffix.lower() not in {'.pyc','.pyo'} and '__pycache__' not in fp.parts and '.venv' not in fp.parts and 'outputs_stage3d_d4' not in fp.parts


def source_inventory(package_root: Path) -> dict[str,Any]:
    roots=['src','vendor','configs','docs','tests']
    top=['README.md','CHANGELOG.md','START_HERE.txt','requirements_py310.txt','RUN_STAGE3D_D4.bat','RUN_STAGE3D_D4.ps1','preflight_windows.py','stage3d_d4_run.py','verify_runtime.py','verify_stage3d_d4.py','package_stage3d_d4_outputs.py']
    rows=[]
    for r in roots:
        base=package_root/r
        if base.exists():
            for fp in sorted(x for x in base.rglob('*') if x.is_file() and _ok_file(x)):
                rows.append({'path':fp.relative_to(package_root).as_posix(),'sha256':sha256_file(fp),'size':fp.stat().st_size})
    for n in top:
        fp=package_root/n
        if fp.is_file(): rows.append({'path':n,'sha256':sha256_file(fp),'size':fp.stat().st_size})
    rows=sorted(rows,key=lambda z:z['path'])
    h=hashlib.sha256()
    for r in rows: h.update(f"{r['path']}\0{r['sha256']}\0{r['size']}\n".encode())
    return {'stage':'Stage3D_D4_M6_NewData_Nuisance_Integration','version':'1.1.0','file_count':len(rows),'aggregate_sha256':h.hexdigest(),'files':rows}


def output_manifest(output_dir: Path, exclude: set[str]|None=None) -> dict[str,Any]:
    exclude=exclude or set(); rows=[]
    for fp in sorted(x for x in output_dir.iterdir() if x.is_file() and x.name not in exclude):
        rows.append({'name':fp.name,'sha256':sha256_file(fp),'size':fp.stat().st_size})
    h=hashlib.sha256()
    for r in rows: h.update(f"{r['name']}\0{r['sha256']}\0{r['size']}\n".encode())
    return {'file_count':len(rows),'aggregate_sha256':h.hexdigest(),'files':rows}
