from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from .io import sha256_file, write_json


def source_inventory(package_root: Path) -> dict[str,Any]:
    include_roots=["src","vendor","configs","docs","tests"]
    top_files=["README.md","CHANGELOG.md","requirements_py310.txt","RUN_STAGE3D_D3_INTEGRATION.bat","RUN_STAGE3D_D3_INTEGRATION.ps1","stage3d_d3_integration_run.py","verify_runtime.py","verify_stage3d_d3_integration.py","package_stage3d_d3_integration_outputs.py"]
    rows=[]
    for root in include_roots:
        rp=package_root/root
        if rp.exists():
            for fp in sorted(x for x in rp.rglob('*') if x.is_file() and '__pycache__' not in x.parts and fp_suffix_ok(x)):
                rows.append({"path":fp.relative_to(package_root).as_posix(),"sha256":sha256_file(fp),"size":fp.stat().st_size})
    for name in top_files:
        fp=package_root/name
        if fp.is_file(): rows.append({"path":name,"sha256":sha256_file(fp),"size":fp.stat().st_size})
    rows=sorted(rows,key=lambda x:x['path'])
    h=hashlib.sha256()
    for r in rows: h.update(f"{r['path']}\0{r['sha256']}\0{r['size']}\n".encode())
    return {"stage":"Stage3D_D3_M1_M6_Exact_Integration","version":"1.0.1","file_count":len(rows),"aggregate_sha256":h.hexdigest(),"files":rows}


def fp_suffix_ok(fp: Path) -> bool:
    return fp.suffix.lower() not in {'.pyc','.pyo'}


def output_manifest(output_dir: Path, exclude: set[str]|None=None) -> dict[str,Any]:
    exclude=exclude or set(); rows=[]
    for fp in sorted(x for x in output_dir.iterdir() if x.is_file() and x.name not in exclude):
        rows.append({"name":fp.name,"sha256":sha256_file(fp),"size":fp.stat().st_size})
    h=hashlib.sha256()
    for r in rows: h.update(f"{r['name']}\0{r['sha256']}\0{r['size']}\n".encode())
    return {"file_count":len(rows),"aggregate_sha256":h.hexdigest(),"files":rows}
