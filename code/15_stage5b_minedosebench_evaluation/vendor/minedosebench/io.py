from __future__ import annotations
import hashlib, json, os, shutil, tempfile, zipfile
from pathlib import Path
from typing import Any, Dict

class MineDoseBenchError(RuntimeError):
    pass

def require(cond: bool, msg: str) -> None:
    if not bool(cond):
        raise MineDoseBenchError(msg)

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical_json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(',',':'), ensure_ascii=False)+'\n').encode('utf-8')

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(obj))

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def safe_extract_zip(src: Path, dst: Path) -> None:
    dst=dst.resolve(); dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        bad=z.testzip(); require(bad is None, f'ZIP CRC failure: {src}: {bad}')
        for info in z.infolist():
            target=(dst / info.filename).resolve()
            require(str(target).startswith(str(dst)), f'Unsafe ZIP member: {info.filename}')
        z.extractall(dst)

def deterministic_zip(root: Path, out_zip: Path) -> None:
    root=root.resolve(); out_zip.parent.mkdir(parents=True, exist_ok=True)
    fixed=(2026,8,10,0,0,0)
    with zipfile.ZipFile(out_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted([p for p in root.rglob('*') if p.is_file()], key=lambda x:x.relative_to(root).as_posix()):
            rel=p.relative_to(root).as_posix()
            info=zipfile.ZipInfo(rel,date_time=fixed)
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644 << 16
            z.writestr(info,p.read_bytes())

def staged_dir(prefix: str='mdb_'):
    return tempfile.TemporaryDirectory(prefix=prefix)
