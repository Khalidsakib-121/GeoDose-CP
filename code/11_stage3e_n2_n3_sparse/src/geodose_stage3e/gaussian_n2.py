from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, eigsh
from .io import require, Stage3EError
from .ordering import predecessor_neighborhoods

@dataclass
class VecchiaResult:
    m: int
    ordering: np.ndarray
    neighborhoods: list[np.ndarray]
    conditional_variance: np.ndarray
    full_predecessor_variance: np.ndarray
    local_kl_terms: np.ndarray
    total_kl: float
    precision: sparse.csr_matrix
    min_precision_eigenvalue: float
    precision_nnz: int
    max_neighbors_used: int


def minimum_eigenvalue(Q: sparse.spmatrix) -> float:
    Q=sparse.csr_matrix(Q); n=Q.shape[0]
    if n<=3: return float(np.min(np.linalg.eigvalsh(Q.toarray())))
    v0=np.linspace(1.0,2.0,n,dtype=float); v0/=np.linalg.norm(v0)
    return float(eigsh(Q,k=1,which='SA',return_eigenvectors=False,tol=1e-11,v0=v0)[0])


def sparse_logdet_spd(Q: sparse.spmatrix) -> float:
    Q=sparse.csc_matrix(Q)
    require(Q.shape[0]==Q.shape[1],'LOGDET_INVALID: nonsquare')
    # For positive-definite Q, |det(Q)| = det(Q). SuperLU permutations do not
    # affect the absolute determinant, so sum log|diag(U)| is sufficient.
    lu=splu(Q)
    d=np.asarray(lu.U.diagonal(),dtype=float)
    require(np.all(np.isfinite(d)) and np.all(np.abs(d)>0.0),'LOGDET_INVALID: singular/nonfinite LU')
    return float(np.sum(np.log(np.abs(d))))


def full_predecessor_variances(Q_order: sparse.spmatrix) -> tuple[np.ndarray,dict[str,object]]:
    """Exact Var(E_i | E_1,...,E_{i-1}) without forming Q^{-1}.

    Reverse-order symmetric elimination produces the scalar Schur-complement
    precision for each sequential conditional.  For the frozen SPD GMRFs,
    SuperLU is required to preserve the natural row/column order; otherwise we
    fail closed rather than reinterpret pivoted factors.
    """
    Q=sparse.csr_matrix(Q_order); n=Q.shape[0]
    rev=np.arange(n-1,-1,-1,dtype=int)
    lu=splu(Q[rev][:,rev].tocsc(),permc_spec='NATURAL',diag_pivot_thresh=0.0,options={'SymmetricMode':True,'Equil':False})
    identity=np.arange(n,dtype=int)
    require(np.array_equal(lu.perm_r,identity) and np.array_equal(lu.perm_c,identity),
            'N2_FULL_CONDITIONAL_FACTOR_PIVOTED: cannot certify reverse Schur diagonals')
    diag=np.asarray(lu.U.diagonal(),dtype=float)
    require(np.all(np.isfinite(diag)) and np.all(diag>0.0),'N2_FULL_CONDITIONAL_INVALID: nonpositive Schur precision')
    out=np.empty(n,dtype=float); out[rev]=1.0/diag
    return out,{'reverse_lu_no_pivot':True,'min_schur_precision':float(diag.min()),'max_schur_precision':float(diag.max())}


def _needed_pairs(neighborhoods_max: list[np.ndarray]) -> dict[int,set[int]]:
    by_col: dict[int,set[int]]={}
    for i,N in enumerate(neighborhoods_max):
        S=[int(i),*map(int,N.tolist())]
        for a in S:
            for b in S:
                # Store requested Sigma[row,col]; symmetric requests are cheap to dedupe by pair.
                row,col=(a,b) if a<=b else (b,a)
                by_col.setdefault(col,set()).add(row)
    return by_col


def selected_covariance_entries(Q_order: sparse.spmatrix, neighborhoods_max: list[np.ndarray]) -> tuple[dict[tuple[int,int],float],dict[str,object]]:
    """Compute only covariance entries required by local conditioning sets.

    A sparse LU of Q is reused. Each solve vector is transient; no dense n-by-n
    inverse is created or retained.
    """
    Q=sparse.csc_matrix(Q_order); n=Q.shape[0]
    requests=_needed_pairs(neighborhoods_max)
    lu=splu(Q)
    entries: dict[tuple[int,int],float]={}
    e=np.zeros(n,dtype=float)
    for col in sorted(requests):
        e.fill(0.0); e[col]=1.0
        x=np.asarray(lu.solve(e),dtype=float)
        require(np.all(np.isfinite(x)),'N2_SELECTED_INVERSE_SOLVE_NONFINITE')
        for row in requests[col]:
            key=(row,col) if row<=col else (col,row)
            entries[key]=float(x[row])
    return entries,{'selected_pair_count':len(entries),'solve_count':len(requests),'dense_inverse_formed':False}


def _cov(entries: dict[tuple[int,int],float], i: int, j: int) -> float:
    key=(i,j) if i<=j else (j,i)
    if key not in entries: raise Stage3EError(f'N2_SELECTED_COVARIANCE_MISSING: {key}')
    return entries[key]


def build_vecchia_family(
    Q: sparse.spmatrix,
    ordering: np.ndarray,
    coords: np.ndarray,
    node_ids: np.ndarray,
    m_values: list[int],
    *,
    minimum_precision_eigenvalue: float,
    kl_nonnegative_tolerance: float,
) -> tuple[dict[int,VecchiaResult],dict[str,object]]:
    Q=sparse.csr_matrix(Q); n=Q.shape[0]
    require(Q.shape==(n,n) and n==len(ordering),'N2_INPUT_DIMENSION_MISMATCH')
    m_values=sorted(set(int(m) for m in m_values)); require(m_values and m_values[0]>=0,'N2_M_GRID_INVALID')
    max_m=max(m_values)
    neighborhoods_max=predecessor_neighborhoods(ordering,coords,node_ids,max_m)
    Qord=Q[ordering][:,ordering].tocsr()
    full_var,full_meta=full_predecessor_variances(Qord)
    cov_entries,cov_meta=selected_covariance_entries(Qord,neighborhoods_max)
    results={}
    for m in m_values:
        cond=np.empty(n,dtype=float); terms=np.empty(n,dtype=float)
        rows=[]; cols=[]; vals=[]
        actual_neighbors=[]
        for i in range(n):
            N=neighborhoods_max[i][:min(m,len(neighborhoods_max[i]))]
            actual_neighbors.append(int(len(N)))
            sii=_cov(cov_entries,i,i)
            require(sii>0.0 and math.isfinite(sii),'N2_COVARIANCE_DIAGONAL_INVALID')
            if len(N)==0:
                v=sii; coeff=np.empty(0,dtype=float)
            else:
                SNN=np.array([[_cov(cov_entries,int(a),int(b)) for b in N] for a in N],dtype=float)
                sNi=np.array([_cov(cov_entries,int(a),i) for a in N],dtype=float)
                try: coeff=np.linalg.solve(SNN,sNi)
                except np.linalg.LinAlgError as exc: raise Stage3EError(f'N2_LOCAL_COVARIANCE_SINGULAR: i={i}, m={m}') from exc
                v=float(sii-float(sNi@coeff))
            require(math.isfinite(v) and v>0.0,f'N2_CONDITIONAL_VARIANCE_INVALID: i={i}, m={m}, v={v}')
            cond[i]=v
            ratio=v/full_var[i]
            require(ratio>0.0 and math.isfinite(ratio),'N2_KL_RATIO_INVALID')
            term=0.5*math.log(ratio)
            if term < -kl_nonnegative_tolerance:
                raise Stage3EError(f'N2_NEGATIVE_INFORMATION_TERM: i={i}, m={m}, term={term}')
            terms[i]=max(0.0,term)
            rows.append(i); cols.append(i); vals.append(1.0)
            for j,c in zip(N,coeff):
                rows.append(i); cols.append(int(j)); vals.append(-float(c))
        T=sparse.csr_matrix((vals,(rows,cols)),shape=(n,n))
        Dinv=sparse.diags(1.0/cond,format='csr')
        Qq=(T.T@Dinv@T).tocsr(); Qq=0.5*(Qq+Qq.T); Qq.eliminate_zeros()
        # Map ordered precision back to original node indexing.
        coo=Qq.tocoo()
        Qorig=sparse.csr_matrix((coo.data,(ordering[coo.row],ordering[coo.col])),shape=(n,n)); Qorig=0.5*(Qorig+Qorig.T); Qorig.eliminate_zeros()
        mineig=minimum_eigenvalue(Qorig)
        require(mineig>minimum_precision_eigenvalue,f'N2_VECCHIA_PRECISION_NOT_PD: m={m}, eig={mineig}')
        results[m]=VecchiaResult(m=m,ordering=ordering.copy(),neighborhoods=[x[:min(m,len(x))].copy() for x in neighborhoods_max],conditional_variance=cond,full_predecessor_variance=full_var.copy(),local_kl_terms=terms,total_kl=float(np.sum(terms)),precision=Qorig,min_precision_eigenvalue=mineig,precision_nnz=int(Qorig.nnz),max_neighbors_used=max(actual_neighbors))
    meta={**full_meta,**cov_meta,'n_nodes':n,'max_m':max_m,'ordering_is_permutation':bool(np.array_equal(np.sort(ordering),np.arange(n)))}
    return results,meta


def trace_precision_covariance(Q_left: sparse.spmatrix, Q_cov_precision: sparse.spmatrix) -> tuple[float,dict[str,object]]:
    """Compute tr(Q_left Q_cov_precision^{-1}) using selected sparse solves only."""
    A=sparse.csc_matrix(Q_left); P=sparse.csc_matrix(Q_cov_precision); n=A.shape[0]
    require(A.shape==P.shape,'GAUSSIAN_KL_DIMENSION_MISMATCH')
    coo=A.tocoo(); by_col: dict[int,list[tuple[int,float]]]={}
    for i,j,v in zip(coo.row,coo.col,coo.data): by_col.setdefault(int(j),[]).append((int(i),float(v)))
    lu=splu(P); e=np.zeros(n,dtype=float); total=0.0
    for j,items in by_col.items():
        e.fill(0.0); e[j]=1.0; x=np.asarray(lu.solve(e),dtype=float)
        for i,v in items: total += v*float(x[i])
    return float(total),{'solve_count':len(by_col),'q_left_nnz':int(A.nnz),'dense_inverse_formed':False}


def gaussian_kl_from_precisions(P_precision: sparse.spmatrix, Q_precision: sparse.spmatrix, *, nonnegative_tolerance: float=1e-9) -> tuple[float,dict[str,object]]:
    """KL[N(0,P^{-1}) || N(0,Q^{-1})] without dense inversion."""
    P=sparse.csr_matrix(P_precision); Q=sparse.csr_matrix(Q_precision); n=P.shape[0]
    require(P.shape==Q.shape,'GAUSSIAN_KL_DIMENSION_MISMATCH')
    tr,meta=trace_precision_covariance(Q,P)
    logdetP=sparse_logdet_spd(P); logdetQ=sparse_logdet_spd(Q)
    kl=0.5*(tr-n+logdetP-logdetQ)
    if kl < -nonnegative_tolerance: raise Stage3EError(f'GAUSSIAN_KL_NEGATIVE: {kl}')
    kl=max(0.0,float(kl))
    return kl,{**meta,'trace':tr,'logdet_P':logdetP,'logdet_Q':logdetQ}


def n2_tv_bound(delta_n2: float) -> float:
    require(delta_n2>=0.0 and math.isfinite(delta_n2),'N2_DELTA_INVALID')
    return min(1.0,math.sqrt(delta_n2/2.0))


def target_shift_identity(r: np.ndarray, L: np.ndarray, weights: np.ndarray|None=None) -> dict[str,float]:
    r=np.asarray(r,dtype=float); L=np.asarray(L,dtype=float)
    require(r.shape==L.shape and r.size>0,'TARGET_SHIFT_ARRAY_INVALID')
    if weights is None: w=np.full(r.size,1.0/r.size)
    else:
        w=np.asarray(weights,dtype=float); require(w.shape==r.shape and np.all(w>=0),'TARGET_SHIFT_WEIGHTS_INVALID'); w=w/w.sum()
    require(np.all(r>=0) and np.all(np.isfinite(r)) and np.all(np.isfinite(L)),'TARGET_SHIFT_NONFINITE')
    mean_r=float(np.sum(w*r)); require(abs(mean_r-1.0)<=1e-8,f'TARGET_RATIO_NOT_NORMALIZED: {mean_r}')
    mean_L=float(np.sum(w*L)); delta=float(np.sum(w*r*L))
    cov=float(np.sum(w*(r-mean_r)*(L-mean_L)))
    vr=float(np.sum(w*(r-mean_r)**2)); vl=float(np.sum(w*(L-mean_L)**2))
    cs=mean_L+math.sqrt(max(0.0,vr*vl))
    return {'mean_r':mean_r,'mean_L':mean_L,'target_weighted_delta':delta,'cov_r_L':cov,'identity_rhs':mean_L+cov,'cs_upper_bound':cs,'var_r':vr,'var_L':vl}


def target_weighted_n2(loss: np.ndarray, target_design_ratio: np.ndarray, weights: np.ndarray | None = None, *, normalization_tolerance: float = 1e-8) -> dict[str,float]:
    """Generic N2 target-frame aggregation Delta_N2=E_obs[r(U)L(U)].

    The ratio must already represent the normalized supported target-design tilt.
    This helper is independent of the frozen Stage3B special case where L(U) is
    constant and therefore the target weighting cancels numerically.
    """
    L=np.asarray(loss,dtype=float); r=np.asarray(target_design_ratio,dtype=float)
    require(L.shape==r.shape and L.size>0,'N2_TARGET_WEIGHT_ARRAY_INVALID')
    require(np.all(np.isfinite(L)) and np.all(L>=-1e-12),'N2_TARGET_WEIGHT_LOSS_INVALID')
    require(np.all(np.isfinite(r)) and np.all(r>=0.0),'N2_TARGET_WEIGHT_RATIO_INVALID')
    L=np.maximum(L,0.0)
    if weights is None:
        w=np.full(L.size,1.0/L.size,dtype=float)
    else:
        w=np.asarray(weights,dtype=float)
        require(w.shape==L.shape and np.all(np.isfinite(w)) and np.all(w>=0.0) and float(w.sum())>0.0,'N2_TARGET_WEIGHT_BASE_WEIGHTS_INVALID')
        w=w/float(w.sum())
    mean_r=float(np.sum(w*r))
    require(abs(mean_r-1.0)<=float(normalization_tolerance),f'N2_TARGET_DESIGN_RATIO_NOT_NORMALIZED:{mean_r}')
    obs=float(np.sum(w*L)); target=float(np.sum(w*r*L))
    cov=float(np.sum(w*(r-mean_r)*(L-obs)))
    vr=float(np.sum(w*(r-mean_r)**2)); vl=float(np.sum(w*(L-obs)**2))
    cs=float(obs+math.sqrt(max(0.0,vr*vl)))
    return {'mean_target_design_ratio':mean_r,'observational_mean_loss':obs,'target_weighted_delta_N2':target,'covariance_ratio_loss':cov,'identity_rhs':obs+cov,'var_ratio':vr,'var_loss':vl,'cauchy_schwarz_upper_bound':cs,'pinsker_tv_upper_bound':n2_tv_bound(target)}
