from __future__ import annotations
import hashlib, json, os, zipfile
from pathlib import Path
from datetime import datetime, timezone

class GateError(RuntimeError):
    pass

def require(ok, msg):
    if not bool(ok):
        raise GateError(str(msg))

def sha256_file(path: Path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def aggregate_rows(rows):
    """Production-package/output aggregate used by this package (file + sha256, sorted)."""
    return hashlib.sha256(''.join(r['file']+r['sha256'] for r in sorted(rows,key=lambda x:x['file'])).encode()).hexdigest()

def acquisition_manifest_aggregate(rows):
    """Exact aggregate contract used by External Context Acquisition v1.0.1.

    Its verify_package.py hashes UTF-8 lines:
      file<TAB>size<TAB>sha256<LF>
    including a final newline.  Do not substitute the production aggregate rule.
    """
    payload=''.join(f"{r['file']}\t{int(r['size'])}\t{r['sha256']}\n" for r in rows)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path: Path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')

def safe_extract_zip(zpath: Path, dest: Path):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);root=dest.resolve()
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            p=(dest/info.filename).resolve()
            require(str(p).startswith(str(root)),f'Unsafe ZIP member: {info.filename}')
        z.extractall(dest)

def deterministic_zip(src_dir: Path, out_zip: Path, include_paths=None):
    src_dir=Path(src_dir);out_zip=Path(out_zip);out_zip.parent.mkdir(parents=True,exist_ok=True)
    files=[p for p in src_dir.rglob('*') if p.is_file()] if include_paths is None else [src_dir/Path(x) for x in include_paths]
    files=sorted(files,key=lambda p:p.relative_to(src_dir).as_posix())
    tmp=out_zip.with_suffix(out_zip.suffix+'.tmp');fixed=(2026,8,10,0,0,0)
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            rel=p.relative_to(src_dir).as_posix();info=zipfile.ZipInfo(rel,date_time=fixed);info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    os.replace(tmp,out_zip)
