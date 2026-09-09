from __future__ import annotations
import gzip,hashlib,json,os,shutil,zipfile
from pathlib import Path
from typing import Any
class Stage4Error(RuntimeError): pass
def require(cond:bool,msg:str)->None:
    if not cond: raise Stage4Error(msg)
def sha256_file(path:str|Path)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def verify_zip(path:Path,expected_sha:str)->dict[str,Any]:
    path=Path(path);require(path.is_file(),f'MISSING_ZIP:{path}');got=sha256_file(path);require(got==expected_sha,f'ZIP_SHA256_MISMATCH:{path}:expected={expected_sha}:got={got}')
    with zipfile.ZipFile(path) as z:
        bad=z.testzip();require(bad is None,f'ZIP_CRC_FAILURE:{path}:{bad}');n=len(z.namelist())
    return {'path':str(path),'sha256':got,'zip_members':n,'zip_crc':'PASS'}
def verify_source_root(root:Path,lock:dict[str,Any])->dict[str,Any]:
    root=Path(root);require(root.is_dir(),f'MISSING_SOURCE_ROOT:{root}');mism=[]
    for rel,expected in lock['files'].items():
        p=root/rel
        if not p.is_file(): mism.append({'path':rel,'status':'missing'});continue
        got=sha256_file(p)
        if got!=expected:mism.append({'path':rel,'status':'hash_mismatch','expected':expected,'got':got})
    require(not mism,f'SOURCE_ROOT_MISMATCH:{root}:{mism[:5]}')
    return {'root':str(root),'verified_files':len(lock['files']),'mismatches':0,'expected_aggregate':lock.get('aggregate')}
def json_dump(path:Path,obj:Any)->None:
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False,default=_json_default)+'\n',encoding='utf-8')
def _json_default(x:Any):
    try:
        import numpy as np
        if isinstance(x,np.integer):return int(x)
        if isinstance(x,np.floating):
            v=float(x);return None if not np.isfinite(v) else v
        if isinstance(x,np.bool_):return bool(x)
    except Exception:pass
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def deterministic_gzip_csv(df,path:Path)->None:
    raw=df.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode('utf-8')
    with open(path,'wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as gz:gz.write(raw)
def atomic_gzip_json(path:Path,obj:Any)->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    raw=json.dumps(obj,sort_keys=True,allow_nan=True,separators=(',',':'),default=_json_default).encode('utf-8')
    with open(tmp,'wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as gz:gz.write(raw)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def read_gzip_json(path:Path)->Any:
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
def prepare_output_dir(path:Path,overwrite:bool)->None:
    path=Path(path)
    if overwrite and path.exists():shutil.rmtree(path)
    path.mkdir(parents=True,exist_ok=True)
