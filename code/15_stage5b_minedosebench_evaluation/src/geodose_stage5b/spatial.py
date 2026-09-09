from __future__ import annotations
from dataclasses import dataclass
import hashlib,math
import numpy as np,pandas as pd
from scipy import sparse
from .common import require,Stage5BError,stable_seed

@dataclass
class GraphInfo:
    ids:list[str]; index:dict[str,int]; adjacency:list[set[int]]; degrees:np.ndarray; edges_idx:list[tuple[int,int]]; edge_hash:str

def graph_info(units:pd.DataFrame,edges:pd.DataFrame)->GraphInfo:
    ids=units.block_id.astype(str).tolist(); idx={b:i for i,b in enumerate(ids)}; adj=[set() for _ in ids]; pairs=[]
    for r in edges.itertuples(index=False):
        a=idx.get(str(r.source_block_id));b=idx.get(str(r.target_block_id))
        if a is None or b is None or a==b: continue
        x,y=(a,b) if a<b else (b,a)
        pairs.append((x,y))
    pairs=sorted(set(pairs))
    for a,b in pairs:adj[a].add(b);adj[b].add(a)
    deg=np.array([len(x) for x in adj],float); text='|'.join(f'{ids[a]}::{ids[b]}' for a,b in pairs); h=hashlib.sha256(text.encode()).hexdigest()
    return GraphInfo(ids,idx,adj,deg,pairs,h)

def deterministic_graph_safe(target_idx:int,cal_idx:np.ndarray,g:GraphInfo)->np.ndarray:
    forbidden=set(g.adjacency[int(target_idx)])|{int(target_idx)}; selected=[]; ss=set()
    for node in sorted(map(int,cal_idx)):
        if node in forbidden:continue
        if g.adjacency[node]&ss:continue
        selected.append(node);ss.add(node)
    return np.asarray(selected,int)

def fit_spatial_pseudolikelihood(residual:np.ndarray,nuisance_idx:np.ndarray,g:GraphInfo,rho_grid=(0,.2,.4,.6,.8),sigma_bounds=(.005,.25)):
    e=np.asarray(residual,float); nidx=np.asarray(nuisance_idx,int); mask=set(map(int,nidx)); pos={node:k for k,node in enumerate(nidx)}; x=e[nidx]; best=None
    for rho in rho_grid:
        neigh=np.zeros(len(nidx),float)
        for node in nidx:
            k=pos[int(node)]; di=g.degrees[node]
            if di<=0:continue
            s=0.
            for nb in g.adjacency[node]:
                if nb in mask and g.degrees[nb]>0:s+=e[nb]/math.sqrt(di*g.degrees[nb])
            neigh[k]=s
        u=x-float(rho)*neigh; sigma=float(np.clip(np.sqrt(np.mean(u*u)),sigma_bounds[0],sigma_bounds[1])); obj=float(len(u)*np.log(sigma)+.5*np.sum((u/sigma)**2))
        rec=(obj,float(rho),sigma)
        if best is None or rec[0]<best[0]-1e-12 or (abs(rec[0]-best[0])<=1e-12 and rec[1]<best[1]):best=rec
    return {'rho':best[1],'sigma':best[2],'objective':best[0],'grid':list(map(float,rho_grid)),'theorem_certified':False}

def _same_mine_calibration(target_idx:int,cal_idx:np.ndarray,units:pd.DataFrame):
    target=units.iloc[int(target_idx)]; arr=np.asarray(cal_idx,int); sub=units.iloc[arr]
    keep=sub.MineID.astype(str).to_numpy()==str(target.MineID)
    arr=arr[keep]; require(len(arr)>0,'R11_INSUFFICIENT_LOCAL_CALIBRATION same-mine calibration empty')
    return target,arr

def nearest_calibration_block(target_idx:int,cal_idx:np.ndarray,units:pd.DataFrame,n_slots=5):
    # Exact/local graph orbits may never be fabricated across disconnected mines.
    target,arr=_same_mine_calibration(target_idx,cal_idx,units); cand=units.iloc[arr][['block_id','centroid_x','centroid_y']].copy(); dx=cand.centroid_x.to_numpy(float)-float(target.centroid_x);dy=cand.centroid_y.to_numpy(float)-float(target.centroid_y);cand['_d2']=dx*dx+dy*dy;cand['_idx']=arr;cand=cand.sort_values(['_d2','block_id'],kind='mergesort');require(len(cand)>=n_slots,'R11_INSUFFICIENT_LOCAL_CALIBRATION');return cand.head(n_slots)._idx.to_numpy(int)

def local_context(target_idx:int,cal_idx:np.ndarray,units:pd.DataFrame,m:int):
    target,arr=_same_mine_calibration(target_idx,cal_idx,units); cand=units.iloc[arr][['block_id','centroid_x','centroid_y']].copy(); dx=cand.centroid_x.to_numpy(float)-float(target.centroid_x);dy=cand.centroid_y.to_numpy(float)-float(target.centroid_y);cand['_d2']=dx*dx+dy*dy;cand['_idx']=arr;cand=cand.sort_values(['_d2','block_id'],kind='mergesort');return np.r_[int(target_idx),cand.head(min(int(m),len(cand)))._idx.to_numpy(int)]

def six_precision(target_idx:int,block_cal_idx:np.ndarray,cal_idx:np.ndarray,units:pd.DataFrame,g:GraphInfo,rho:float,m:int):
    # Sparse/localized GMRF reference. We form only a <=(m+1) local matrix, never a dense global inverse.
    ctx=local_context(target_idx,cal_idx,units,m); cset=set(map(int,ctx)); loc={node:k for k,node in enumerate(ctx)}; n=len(ctx); Q=np.eye(n,dtype=float)
    for a in ctx:
        da=g.degrees[a]
        if da<=0:continue
        for b in g.adjacency[a]:
            if b not in cset or a>=b or g.degrees[b]<=0:continue
            v=-float(rho)/math.sqrt(da*g.degrees[b]);ia,ib=loc[a],loc[b];Q[ia,ib]=v;Q[ib,ia]=v
    ev=np.linalg.eigvalsh(Q);require(float(ev.min())>1e-10,f'Local sparse precision not positive definite: {ev.min()}')
    block=np.r_[np.asarray(block_cal_idx,int),int(target_idx)]; require(len(set(map(int,block)))==6,'Local block not six distinct slots'); bpos=np.array([loc[int(x)] for x in block],int); E=np.zeros((n,6));E[bpos,np.arange(6)]=1.; X=np.linalg.solve(Q,E);cov=X[bpos,:];cov=.5*(cov+cov.T);cev=np.linalg.eigvalsh(cov);require(cev.min()>1e-12,'Six-slot covariance not PD');P=np.linalg.solve(cov,np.eye(6));P=.5*(P+P.T)
    return P,cov,len(ctx)-1,{'local_min_precision_eigenvalue':float(ev.min()),'six_cov_min_eigenvalue':float(cev.min()),'m':int(m),'context_calibration_count':len(ctx)-1}

def gaussian_kl_cov(cov_p,cov_q):
    P=np.asarray(cov_p,float);Q=np.asarray(cov_q,float);k=P.shape[0];invQ=np.linalg.solve(Q,np.eye(k));signp,ldp=np.linalg.slogdet(P);signq,ldq=np.linalg.slogdet(Q);require(signp>0 and signq>0,'KL covariance determinant nonpositive');v=.5*(np.trace(invQ@P)-k+ldq-ldp);return float(max(0.,v))

def localization_diagnostic(target_idx,block_cal_idx,cal_idx,units,g,rho,m=64,m_small=32):
    P64,C64,count,meta=six_precision(target_idx,block_cal_idx,cal_idx,units,g,rho,m);P32,C32,count32,meta32=six_precision(target_idx,block_cal_idx,cal_idx,units,g,rho,m_small);kl=gaussian_kl_cov(C64,C32);tv=min(1.,math.sqrt(kl/2.));return P64,{'delta_sparse_localization_kl':kl,'delta_sparse_pinsker_tv':tv,'n3_diagnostic_lower_bound':max(0.,.9-tv),'m64_count':count,'m32_count':count32,**meta}

def spatial_diagnostics(case_id,rep,track,residual,units,g:GraphInfo,support_idx,seed):
    idx=np.asarray(support_idx,int); vals=np.asarray(residual,float)[idx]; rows=[]
    # Moran I on edges fully inside support-audit frame.
    pos={node:k for k,node in enumerate(idx)};z=vals-float(np.mean(vals));den=float(np.sum(z*z));num=0.;ned=0
    for a,b in g.edges_idx:
        if a in pos and b in pos:num+=2*z[pos[a]]*z[pos[b]];ned+=1
    mor=float((len(idx)/(2*ned))*num/den) if ned and den>0 else np.nan
    rows.append({'case_id':case_id,'replication':rep,'track':track,'diagnostic':'morans_I','bin':'all','value':mor,'pair_count':ned,'diagnostic_only':True})
    # Deterministic Euclidean semivariogram in 5 distance-quantile bins, capped sample for runtime.
    rng=np.random.default_rng(int(seed));n=len(idx);maxpairs=min(12000,max(0,n*(n-1)//2));pairs=set();attempt=0
    while len(pairs)<maxpairs and attempt<maxpairs*4+100:
        i,j=sorted(rng.integers(0,n,size=2).tolist());attempt+=1
        if i!=j:pairs.add((i,j))
    if pairs:
        arr=np.array(sorted(pairs),int);u=units.iloc[idx];xy=u[['centroid_x','centroid_y']].to_numpy(float);d=np.sqrt(np.sum((xy[arr[:,0]]-xy[arr[:,1]])**2,axis=1));semi=.5*(vals[arr[:,0]]-vals[arr[:,1]])**2;cuts=np.quantile(d,[0,.2,.4,.6,.8,1.]);
        for b in range(5):
            mask=(d>=cuts[b]) & ((d<=cuts[b+1]) if b==4 else (d<cuts[b+1]));rows.append({'case_id':case_id,'replication':rep,'track':track,'diagnostic':'semivariogram','bin':f'Q{b+1}','value':float(np.mean(semi[mask])) if mask.any() else np.nan,'pair_count':int(mask.sum()),'diagnostic_only':True})
    else:
        for b in range(5):rows.append({'case_id':case_id,'replication':rep,'track':track,'diagnostic':'semivariogram','bin':f'Q{b+1}','value':np.nan,'pair_count':0,'diagnostic_only':True})
    return rows
