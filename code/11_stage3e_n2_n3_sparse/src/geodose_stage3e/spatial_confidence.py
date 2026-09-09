from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from .io import require, Stage3EError
from .gaussian_n2 import minimum_eigenvalue

@dataclass
class RhoConfidenceResult:
    rho_hat: float
    h_at_hat: float
    outer_intervals: list[tuple[float,float]]
    radius: float
    delta: float
    t_value: float
    lipschitz_h: float
    cell_tolerance: float
    exact_set_nonempty: bool


def normalized_adjacency(n_nodes: int, edges: pd.DataFrame) -> sparse.csr_matrix:
    rows=[]; cols=[]; vals=[]
    for r in edges.itertuples(index=False):
        i=int(r.source_node); j=int(r.target_node)
        require(0<=i<n_nodes and 0<=j<n_nodes and i!=j,'S3_EDGE_INVALID')
        rows.extend([i,j]); cols.extend([j,i]); vals.extend([1.0,1.0])
    W=sparse.csr_matrix((vals,(rows,cols)),shape=(n_nodes,n_nodes)); W.sum_duplicates(); W.data[:]=1.0
    degree=np.asarray(W.sum(axis=1)).reshape(-1); inv=np.zeros(n_nodes,dtype=float); pos=degree>0; inv[pos]=1/np.sqrt(degree[pos])
    D=sparse.diags(inv,format='csr'); S=(D@W@D).tocsr(); S=0.5*(S+S.T); return S


def affine_precision_from_S(S: sparse.spmatrix, rho: float, scale: float) -> sparse.csr_matrix:
    require(scale>0 and 0<=rho<0.999,'S3_PRECISION_PARAMETER_INVALID')
    n=S.shape[0]; Q=(sparse.eye(n,format='csr')-float(rho)*sparse.csr_matrix(S))/(scale*scale); Q=0.5*(Q+Q.T); return Q.tocsr()


def _score_components(eigenvalues: np.ndarray, eSe: float, scale: float, rho: float, t: float) -> tuple[float,float,float,float]:
    den=1.0-rho*eigenvalues
    require(np.all(den>0),'S3_PARAMETER_NOT_PD')
    b=-eigenvalues/den
    score=0.5*float(np.sum(b))+0.5*float(eSe)/(scale*scale)
    fro=float(np.linalg.norm(b)); op=float(np.max(np.abs(b))) if len(b) else 0.0
    bound=fro*math.sqrt(t)+op*t
    return score,bound,fro,op


def rho_score_confidence_set(
    S_train: sparse.spmatrix,
    residual_train: np.ndarray,
    scale: float,
    rho_domain: tuple[float,float],
    delta: float,
    cell_tolerance: float,
) -> RhoConfidenceResult:
    """Conservative outer score-inversion set for the one-parameter centered affine GMRF.

    The recursive exclusion/acceptance tests use a global Lipschitz bound for
    h(rho)=|score|-certificate_radius. Cells unresolved at the frozen width are
    included, so the returned set is an outer approximation and cannot shrink
    the mathematical confidence set.
    """
    S=sparse.csr_matrix(S_train); e=np.asarray(residual_train,dtype=float); n=S.shape[0]
    require(S.shape==(n,n) and len(e)==n and np.all(np.isfinite(e)),'S3_INPUT_INVALID')
    lo,hi=map(float,rho_domain); require(0<=lo<hi<0.999,'S3_RHO_DOMAIN_INVALID')
    require(0<delta<1 and cell_tolerance>0,'S3_CONFIDENCE_CONFIG_INVALID')
    # Exact symmetric eigenvalues for the independent nuisance component; no inverse is formed.
    lam=np.linalg.eigvalsh(S.toarray())
    eSe=float(e@(S@e)); t=math.log(2.0/delta)  # p_E=1
    def h(rho: float) -> float:
        score,bound,_,_=_score_components(lam,eSe,scale,float(rho),t)
        return abs(score)-bound
    # MLE within frozen compact parameter space.
    ee=float(e@e)
    def nll(rho: float) -> float:
        den=1.0-float(rho)*lam
        if np.any(den<=0): return float('inf')
        return -0.5*float(np.sum(np.log(den))) + 0.5*(ee-float(rho)*eSe)/(scale*scale)
    opt=minimize_scalar(nll,bounds=(lo,hi),method='bounded',options={'xatol':1e-13,'maxiter':1000})
    require(bool(opt.success) and math.isfinite(float(opt.x)),'S3_MLE_FAILED')
    rho_hat=float(opt.x); h_hat=float(h(rho_hat))
    # If MLE is outside the exact confidence set, find the most favorable h point.
    if h_hat>0:
        opt_h=minimize_scalar(h,bounds=(lo,hi),method='bounded',options={'xatol':1e-13,'maxiter':1000})
        if not bool(opt_h.success) or float(opt_h.fun)>1e-10:
            return RhoConfidenceResult(rho_hat,float(opt_h.fun if opt_h.success else h_hat),[],math.inf,delta,t,math.inf,cell_tolerance,False)
        rho_hat=float(opt_h.x); h_hat=float(opt_h.fun)
    # Global Lipschitz upper bound on h over [lo,hi].
    # For b(lambda,rho)=-lambda/(1-rho lambda), |b'|=lambda^2/(1-rho lambda)^2.
    den_lo=np.minimum(np.abs(1.0-lo*lam),np.abs(1.0-hi*lam))
    # Linear denominators cannot cross zero because every endpoint is PD and domain is convex for affine Q.
    require(np.all(den_lo>0),'S3_LIPSCHITZ_DENOMINATOR_INVALID')
    bprime_abs=(lam*lam)/(den_lo*den_lo)
    score_prime_bound=0.5*float(np.sum(bprime_abs))
    bound_prime_bound=math.sqrt(t)*float(np.linalg.norm(bprime_abs))+t*float(np.max(bprime_abs) if len(bprime_abs) else 0.0)
    L=score_prime_bound+bound_prime_bound
    cells=[(lo,hi)]; included=[]
    while cells:
        a,b=cells.pop(); mid=0.5*(a+b); half=0.5*(b-a); hm=float(h(mid))
        if hm-L*half>0:
            continue  # certifiably outside
        if hm+L*half<=0 or (b-a)<=cell_tolerance:
            included.append((a,b)); continue
        cells.append((a,mid)); cells.append((mid,b))
    included.sort(); merged=[]
    for a,b in included:
        if not merged or a>merged[-1][1]+1e-15: merged.append([a,b])
        else: merged[-1][1]=max(merged[-1][1],b)
    intervals=[(float(a),float(b)) for a,b in merged]
    require(intervals,'S3_OUTER_CONFIDENCE_EMPTY_DESPITE_ACCEPTED_POINT')
    radius=max(max(abs(a-rho_hat),abs(b-rho_hat)) for a,b in intervals)
    return RhoConfidenceResult(rho_hat,h_hat,intervals,float(radius),delta,t,float(L),cell_tolerance,True)


def graph_boundary_from_precision(Q: sparse.spmatrix, block_nodes: np.ndarray) -> np.ndarray:
    Q=sparse.csr_matrix(Q); B=np.asarray(block_nodes,dtype=int); bset=set(map(int,B)); out=set()
    for i in B:
        row=Q.getrow(int(i))
        for j,v in zip(row.indices,row.data):
            if int(j) not in bset and float(v)!=0.0: out.add(int(j))
    return np.asarray(sorted(out),dtype=int)


def gmrf_boundary_certificate(
    S_full: sparse.spmatrix,
    scale: float,
    confidence: RhoConfidenceResult,
    block_nodes: np.ndarray,
    boundary_residual: np.ndarray,
) -> dict[str,float|bool|str]:
    """Unified Math Section 12 realized-boundary GMRF-C bound, p_E=1.

    Bounds q_lo/q_hi/M_BD are taken uniformly over the conservative outer
    confidence set, so on the S3 event they cover the unknown true rho.
    """
    if not confidence.exact_set_nonempty or not confidence.outer_intervals:
        return {'eligible':False,'refusal_code':'S3_CONFIDENCE_SET_EMPTY'}
    S=sparse.csr_matrix(S_full); B=np.asarray(block_nodes,dtype=int)
    Qhat=affine_precision_from_S(S,confidence.rho_hat,scale)
    # Use the frozen structural graph boundary, not a parameter-specific nonzero
    # pattern.  This remains valid even when the confidence set includes rho=0,
    # where Q(rho) would temporarily zero all off-diagonal graph coefficients.
    D=graph_boundary_from_precision(S,B)
    eD=np.asarray(boundary_residual,dtype=float)
    require(len(D)==len(eD),'GMRF_C_BOUNDARY_RESIDUAL_MISMATCH')
    # Q(rho)=Qbase+rho L with L=-S/scale^2.
    L=(-S/(scale*scale)).tocsr(); LBB=L[B][:,B].toarray(); LBD=L[B][:,D].toarray() if len(D) else np.zeros((len(B),0))
    L_BB=float(np.linalg.norm(LBB,2)); L_BD=float(np.linalg.norm(LBD,2)) if LBD.size else 0.0
    rE=float(confidence.radius); dB=L_BB*rE; dBD=L_BD*rE
    endpoints=sorted(set(x for ab in confidence.outer_intervals for x in ab))
    qlo=math.inf; qhi=0.0; MBD=0.0
    for rho in endpoints:
        Q=affine_precision_from_S(S,float(rho),scale); QBB=Q[B][:,B].toarray(); QBD=Q[B][:,D].toarray() if len(D) else np.zeros((len(B),0))
        ev=np.linalg.eigvalsh(QBB); qlo=min(qlo,float(ev.min())); qhi=max(qhi,float(ev.max()));
        if QBD.size: MBD=max(MBD,float(np.linalg.norm(QBD,2)))
    if not math.isfinite(qlo) or qlo<=0: return {'eligible':False,'refusal_code':'GMRF_C_NO_POSITIVE_QLO'}
    if not dB<qlo:
        return {'eligible':False,'refusal_code':'GMRF_C_PERTURBATION_NOT_PD','d_B':dB,'q_lo':qlo,'radius':rE}
    rhoB=dB/qlo
    Kcov=0.5*len(B)*(-rhoB-math.log1p(-rhoB))
    KA=(dBD+dB*MBD/qlo)/(qlo-dB)
    Kmean=0.5*(qhi+dB)*(KA**2)*float(eD@eD)
    kl=Kcov+Kmean
    return {'eligible':True,'refusal_code':'','block_size':len(B),'boundary_size':len(D),'radius':rE,'L_BB':L_BB,'L_BD':L_BD,'d_B':dB,'d_BD':dBD,'q_lo':qlo,'q_hi':qhi,'M_BD':MBD,'rho_B':rhoB,'K_cov':Kcov,'K_mean':Kmean,'kl_upper_bound':kl,'tv_pinsker_upper_bound':min(1.0,math.sqrt(max(0.0,kl)/2.0))}
