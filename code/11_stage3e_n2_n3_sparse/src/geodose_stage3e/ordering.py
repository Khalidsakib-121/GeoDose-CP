from __future__ import annotations
import numpy as np
from .io import require

def deterministic_maximin_order(coords: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    coords=np.asarray(coords,dtype=float); node_ids=np.asarray(node_ids,dtype=int)
    require(coords.ndim==2 and coords.shape[1]>=2,'ORDERING_INVALID: coords')
    require(len(coords)==len(node_ids) and len(np.unique(node_ids))==len(node_ids),'ORDERING_INVALID: node ids')
    require(np.all(np.isfinite(coords)),'ORDERING_INVALID: nonfinite coords')
    n=len(node_ids); require(n>0,'ORDERING_INVALID: empty')
    first=int(np.argmin(node_ids))
    order=[first]; remaining=np.ones(n,dtype=bool); remaining[first]=False
    min_d2=np.sum((coords-coords[first])**2,axis=1); min_d2[first]=-1.0
    for _ in range(1,n):
        max_d=float(np.max(min_d2[remaining]))
        # Deterministic float-tie handling. Coordinates are frozen, but use a scale-aware tiny tolerance.
        tol=1e-14*max(1.0,abs(max_d))
        cand=np.flatnonzero(remaining & (np.abs(min_d2-max_d)<=tol))
        require(len(cand)>0,'ORDERING_INVALID: no maximin candidate')
        pick=int(cand[np.argmin(node_ids[cand])])
        order.append(pick); remaining[pick]=False
        d2=np.sum((coords-coords[pick])**2,axis=1)
        min_d2=np.minimum(min_d2,d2); min_d2[~remaining]=-1.0
    out=np.asarray(order,dtype=int)
    require(np.array_equal(np.sort(out),np.arange(n)),'ORDERING_INVALID: not a permutation')
    return out

def predecessor_neighborhoods(order: np.ndarray, coords: np.ndarray, node_ids: np.ndarray, max_m: int) -> list[np.ndarray]:
    order=np.asarray(order,dtype=int); coords=np.asarray(coords,dtype=float); node_ids=np.asarray(node_ids,dtype=int)
    n=len(order); require(max_m>=0,'NEIGHBORHOOD_INVALID: negative m')
    result=[]
    for pos in range(n):
        if pos==0 or max_m==0:
            result.append(np.empty(0,dtype=int)); continue
        pred=np.arange(pos,dtype=int)
        d2=np.sum((coords[order[pred]]-coords[order[pos]])**2,axis=1)
        pred_node=node_ids[order[pred]]
        ranked=np.lexsort((pred_node,d2))
        result.append(pred[ranked[:min(max_m,pos)]].astype(int))
    return result
