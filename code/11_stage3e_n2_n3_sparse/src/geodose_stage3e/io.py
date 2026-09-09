from __future__ import annotations
import gzip, hashlib, io, json, shutil, zipfile
from pathlib import Path
from typing import Any
import pandas as pd

class Stage3EError(RuntimeError):
    pass

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise Stage3EError(message)

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(chunk_size),b''): h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path,index=False,lineterminator='\n',float_format='%.17g')

def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    text=df.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode('utf-8')
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz: gz.write(text)

def read_json_from_zip(path: Path, logical_name: str) -> Any:
    with zipfile.ZipFile(path) as zf:
        matches=[n for n in zf.namelist() if n.replace('\\','/').endswith(logical_name)]
        require(len(matches)==1,f'Expected one {logical_name} in {path}, found {len(matches)}')
        return json.loads(zf.read(matches[0]).decode('utf-8'))

def read_csv_from_zip(path: Path, logical_name: str, **kwargs: Any) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        matches=[n for n in zf.namelist() if n.replace('\\','/').endswith(logical_name)]
        require(len(matches)==1,f'Expected one {logical_name} in {path}, found {len(matches)}')
        raw=zf.read(matches[0])
        if logical_name.endswith('.gz'): raw=gzip.decompress(raw)
        return pd.read_csv(io.BytesIO(raw),**kwargs)

def verify_zip(path: Path, expected_sha256: str) -> dict[str,Any]:
    p=Path(path); require(p.is_file(),f'Missing input archive: {p}')
    obs=sha256_file(p); require(obs.lower()==expected_sha256.lower(),f'SHA-256 mismatch for {p.name}: expected {expected_sha256}, observed {obs}')
    try:
        with zipfile.ZipFile(p) as zf:
            bad=zf.testzip(); require(bad is None,f'ZIP CRC failure in {p}: {bad}')
            count=len(zf.namelist())
    except zipfile.BadZipFile as exc: raise Stage3EError(f'Invalid ZIP: {p}') from exc
    return {'path':str(p),'sha256':obs,'zip_integrity':True,'member_count':count}

def verify_source_tree(root: Path, expected_files: dict[str,str]) -> dict[str,Any]:
    root=Path(root); require(root.is_dir(),f'Missing source root: {root}')
    missing=[]; mismatched=[]
    for rel,exp in sorted(expected_files.items()):
        fp=root/Path(rel)
        if not fp.is_file(): missing.append(rel); continue
        obs=sha256_file(fp)
        if obs.lower()!=str(exp).lower(): mismatched.append({'path':rel,'expected':str(exp),'observed':obs})
    return {'root':str(root),'expected_file_count':len(expected_files),'matched_file_count':len(expected_files)-len(missing)-len(mismatched),'missing':missing,'mismatched':mismatched,'pass':not missing and not mismatched}

def ensure_empty_output_dir(path: Path, overwrite: bool) -> None:
    path=Path(path)
    if path.exists() and any(path.iterdir()):
        require(overwrite,f'Output directory is not empty: {path}')
        for fp in path.iterdir():
            if fp.is_dir(): shutil.rmtree(fp)
            else: fp.unlink()
    path.mkdir(parents=True,exist_ok=True)
