from __future__ import annotations
from collections import deque
import hashlib
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .io import require

ROLE_ORDER=['nuisance_training','buffer_excluded','support_audit','calibration','test_target']

def _edge_index(blocks: pd.DataFrame, edges: pd.DataFrame):
    ids=blocks.block_id.astype(str).tolist(); idx={b:i for i,b in enumerate(ids)}
    e=edges[edges.source_block_id.astype(str).isin(idx)&edges.target_block_id.astype(str).isin(idx)].copy()
    s=e.source_block_id.astype(str).map(idx).to_numpy(int); t=e.target_block_id.astype(str).map(idx).to_numpy(int)
    return idx,s,t,e

def largest_component(blocks: pd.DataFrame,edges: pd.DataFrame):
    idx,s,t,_=_edge_index(blocks,edges); n=len(blocks)
    A=sparse.csr_matrix((np.ones(len(s)*2),(np.r_[s,t],np.r_[t,s])),shape=(n,n))
    nc,lab=connected_components(A,directed=False,return_labels=True)
    counts=np.bincount(lab); best=int(np.argmax(counts)); return lab==best,lab,counts

def spatial_axis_roles(blocks: pd.DataFrame,edges: pd.DataFrame):
    b=blocks.copy().reset_index(drop=True); x=b[['centroid_x','centroid_y']].to_numpy(float); x=x-x.mean(axis=0)
    _,_,vt=np.linalg.svd(x,full_matrices=False); v=vt[0];
    if v[0]<0 or (v[0]==0 and v[1]<0): v=-v
    score=x@v; order=np.argsort(score,kind='mergesort'); rank=np.empty(len(b),float); rank[order]=(np.arange(len(b))+0.5)/len(b)
    role=np.full(len(b),'buffer_excluded',object)
    role[rank<=.30]='nuisance_training'; role[(rank>.36)&(rank<=.50)]='support_audit'; role[(rank>.56)&(rank<=.80)]='calibration'; role[rank>.80]='test_target'
    # Fail-safe graph erosion for forbidden information edges. Calibration-test edges remain allowed.
    idx,s,t,_=_edge_index(b,edges)
    forbidden={frozenset(('nuisance_training','support_audit')),frozenset(('nuisance_training','calibration')),frozenset(('nuisance_training','test_target')),frozenset(('support_audit','calibration')),frozenset(('support_audit','test_target'))}
    for _ in range(20):
        bad=[]
        for a,c in zip(s,t):
            if frozenset((role[a],role[c])) in forbidden: bad.extend([a,c])
        if not bad: break
        role[np.unique(bad)]='buffer_excluded'
    else: raise RuntimeError('Spatial split erosion did not converge')
    for a,c in zip(s,t): require(frozenset((role[a],role[c])) not in forbidden,'Forbidden role adjacency remains')
    b['spatial_axis_score']=score; b['spatial_axis_rank']=rank; b['benchmark_role']=role
    counts=b.benchmark_role.value_counts().to_dict()
    for r,min_n in [('nuisance_training',50),('support_audit',30),('calibration',30),('test_target',15)]: require(counts.get(r,0)>=min_n,f'Insufficient {r}: {counts}')
    return b,v

def exact_block_map(blocks: pd.DataFrame,edges: pd.DataFrame):
    ids=blocks.block_id.astype(str).tolist(); idset=set(ids); role=dict(zip(ids,blocks.benchmark_role))
    adj={b:set() for b in ids}
    for r in edges.itertuples(index=False):
        a=str(r.source_block_id); c=str(r.target_block_id)
        if a in idset and c in idset: adj[a].add(c); adj[c].add(a)
    rows=[]
    for target in sorted([b for b in ids if role[b]=='test_target']):
        q=deque(); dist={};
        for n in sorted(adj[target]):
            if role.get(n)=='calibration': q.append(n); dist[n]=1
        while q and len(dist)<5:
            u=q.popleft()
            for v in sorted(adj[u]):
                if role.get(v)=='calibration' and v not in dist: dist[v]=dist[u]+1; q.append(v)
        if len(dist)>=5:
            cal=sorted(dist,key=lambda b:(dist[b],b))[:5]
            rows.append(dict(target_block_id=target,calibration_block_1=cal[0],calibration_block_2=cal[1],calibration_block_3=cal[2],calibration_block_4=cal[3],calibration_block_5=cal[4],exact_block_size=6))
    return pd.DataFrame(rows)

def deterministic_subsample(block_ids, fraction:float, label:str):
    if fraction>=1:return np.ones(len(block_ids),bool)
    vals=[]
    for b in map(str,block_ids): vals.append(int.from_bytes(hashlib.sha256((label+'|'+b).encode()).digest()[:8],'big')/2**64)
    return np.array(vals)<fraction
