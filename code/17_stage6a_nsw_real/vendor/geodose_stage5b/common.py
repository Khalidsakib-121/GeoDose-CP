from __future__ import annotations
import gzip, hashlib, io, json, math, os, shutil, tempfile, zipfile
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd

class Stage5BError(RuntimeError): pass

def require(cond: bool, msg: str):
    if not bool(cond): raise Stage5BError(str(msg))

def sha256_file(path: Path, chunk: int=1<<20)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def canonical_json(obj:Any)->bytes:
    return (json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')

def write_json(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(canonical_json(obj))

def read_json(path:Path): return json.loads(path.read_text(encoding='utf-8'))

def _encode_nonfinite(obj):
    if isinstance(obj,float):
        if np.isnan(obj): return {"__float__":"nan"}
        if np.isposinf(obj): return {"__float__":"+inf"}
        if np.isneginf(obj): return {"__float__":"-inf"}
        return obj
    if isinstance(obj,(np.floating,)): return _encode_nonfinite(float(obj))
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if isinstance(obj,dict): return {str(k):_encode_nonfinite(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [_encode_nonfinite(v) for v in obj]
    return obj

def _decode_nonfinite(obj):
    if isinstance(obj,dict) and set(obj)=={"__float__"}:
        return {"nan":np.nan,"+inf":np.inf,"-inf":-np.inf}[obj["__float__"]]
    if isinstance(obj,dict): return {k:_decode_nonfinite(v) for k,v in obj.items()}
    if isinstance(obj,list): return [_decode_nonfinite(v) for v in obj]
    return obj

def atomic_json_gz(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=canonical_json(_encode_nonfinite(obj))
    buf=io.BytesIO()
    with gzip.GzipFile(fileobj=buf,mode='wb',compresslevel=6,mtime=0) as g: g.write(raw)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_bytes(buf.getvalue()); os.replace(tmp,path)

def read_json_gz(path:Path):
    with gzip.open(path,'rt',encoding='utf-8') as f: return _decode_nonfinite(json.load(f))

def deterministic_csv_gz(df:pd.DataFrame,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=df.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode('utf-8')
    buf=io.BytesIO()
    with gzip.GzipFile(fileobj=buf,mode='wb',compresslevel=9,mtime=0) as g:g.write(raw)
    path.write_bytes(buf.getvalue())

def safe_extract_zip(src:Path,dst:Path):
    dst=dst.resolve(); dst.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(src) as z:
        bad=z.testzip(); require(bad is None,f'ZIP CRC failure: {src}: {bad}')
        for info in z.infolist():
            target=(dst/info.filename).resolve(); require(str(target).startswith(str(dst)),f'Unsafe ZIP member: {info.filename}')
        z.extractall(dst)

def deterministic_zip(root:Path,out_zip:Path):
    root=root.resolve(); out_zip.parent.mkdir(parents=True,exist_ok=True); fixed=(2026,8,11,0,0,0)
    with zipfile.ZipFile(out_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted((p for p in root.rglob('*') if p.is_file()),key=lambda p:p.relative_to(root).as_posix()):
            rel=p.relative_to(root).as_posix(); info=zipfile.ZipInfo(rel,date_time=fixed); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16; z.writestr(info,p.read_bytes())

def aggregate_manifest(rows:list[dict])->str:
    h=hashlib.sha256()
    for r in sorted(rows,key=lambda x:x['file']):
        h.update(f"{r['file']}\t{int(r.get('bytes',r.get('size')))}\t{r['sha256']}\n".encode())
    return h.hexdigest()

def file_manifest(root:Path,exclude:set[str]|None=None):
    exclude=exclude or set(); rows=[]
    for p in sorted((p for p in root.rglob('*') if p.is_file()),key=lambda p:p.relative_to(root).as_posix()):
        rel=p.relative_to(root).as_posix()
        if rel in exclude: continue
        rows.append({'file':rel,'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    return rows

def finite_or_inf(x): return bool(np.isfinite(x) or np.isinf(x))

def interval_score(lo:float,hi:float,y:float,alpha:float)->float:
    if not np.isfinite(y): return np.nan
    if np.isneginf(lo) or np.isposinf(hi): return np.inf
    if not(np.isfinite(lo) and np.isfinite(hi)): return np.nan
    s=hi-lo
    if y<lo:s+=(2/alpha)*(lo-y)
    elif y>hi:s+=(2/alpha)*(y-hi)
    return float(s)

def stable_seed(*parts,mod=2_147_483_647):
    raw='|'.join(map(str,parts)).encode(); v=int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%mod; return int(v or 1)
