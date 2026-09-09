from __future__ import annotations
import hashlib,json,zipfile,shutil,os,io,gzip
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd

class Stage6AError(RuntimeError): pass
def require(cond,msg):
    if not bool(cond): raise Stage6AError(str(msg))
def sha256_file(path:Path,chunk=1<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''):h.update(b)
    return h.hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def canonical_json(o): return (json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def _clean(o):
    if isinstance(o,np.integer):return int(o)
    if isinstance(o,np.floating):
        x=float(o)
        if np.isnan(x):return None
        if np.isposinf(x):return "Infinity"
        if np.isneginf(x):return "-Infinity"
        return x
    if isinstance(o,np.bool_):return bool(o)
    if isinstance(o,dict):return {str(k):_clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)):return [_clean(v) for v in o]
    return o
def write_json(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canonical_json(_clean(o)))
def stable_seed(*parts,mod=2_147_483_647):
    v=int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')%mod
    return int(v or 1)
def safe_extract_zip(src:Path,dst:Path):
    dst=Path(dst);dst.mkdir(parents=True,exist_ok=True);root=dst.resolve()
    with zipfile.ZipFile(src) as z:
        require(z.testzip() is None,f'ZIP CRC failure: {src}')
        for info in z.infolist():
            name=info.filename.replace('\\','/').lstrip('/')
            parts=[x for x in name.split('/') if x not in ('','.')]
            require('..' not in parts,f'Unsafe ZIP member: {info.filename}')
            target=root.joinpath(*parts)
            if info.is_dir() or name.endswith('/'): target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as a,open(target,'wb') as b:shutil.copyfileobj(a,b)
def conservative_split_quantile(scores,alpha):
    x=np.sort(np.asarray(scores,float));require(x.ndim==1 and len(x)>0 and np.isfinite(x).all(),'Invalid conformal scores')
    aug=np.r_[x,np.inf];k=int(np.ceil((len(x)+1)*(1-float(alpha))));k=max(1,min(k,len(aug)));return float(aug[k-1])
def interval_bounds(center,q,domain):
    rawlo=float(center-q);rawhi=float(center+q);lo=max(float(domain[0]),rawlo);hi=min(float(domain[1]),rawhi)
    if lo>hi:return rawlo,rawhi,np.nan,np.nan,np.nan
    return rawlo,rawhi,lo,hi,float(hi-lo)
def coverage(y,lo,hi):
    return bool(np.isfinite(y) and np.isfinite(lo) and np.isfinite(hi) and float(lo)<=float(y)<=float(hi))
