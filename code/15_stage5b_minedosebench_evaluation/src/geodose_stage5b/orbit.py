from __future__ import annotations
import itertools,math
import numpy as np
from .common import require,Stage5BError

PERMS=np.asarray(list(itertools.permutations(range(6))),dtype=np.int16)
SLOTS=np.arange(6,dtype=int)[None,:]
SCORE_ABS_TOL=1.0e-12
SCORE_REL_TOL=1.0e-12
NON_GAUSSIAN_MIN_ABS_RESIDUAL=1.0e-14

def upper_tail_mask(scores,obs,abs_tol=SCORE_ABS_TOL,rel_tol=SCORE_REL_TOL):
    scores=np.asarray(scores,float);obs=np.asarray(obs,float);tol=float(abs_tol)+float(rel_tol)*np.maximum(np.abs(scores),np.abs(obs));return scores+tol>=obs

def normalize_logweights(logw):
    x=np.asarray(logw,float); require(not np.isnan(x).any() and not np.isposinf(x).any(),'R07_ALL_ORBIT_WEIGHTS_ZERO invalid log weight');fin=np.isfinite(x);require(fin.any(),'R07_ALL_ORBIT_WEIGHTS_ZERO');m=float(x[fin].max());w=np.zeros(len(x));w[fin]=np.exp(x[fin]-m);s=float(w.sum());require(np.isfinite(s) and s>0,'R07_ALL_ORBIT_WEIGHTS_ZERO');return w/s

def _inverse_power(e,power):
    e=np.asarray(e,float)
    if abs(float(power)-1.)<1e-15:return e
    return np.sign(e)*np.abs(e)**(1./float(power))

def spatial_log(assign_residuals,precision,power=1.0,include_transform_jacobian=False):
    e=np.asarray(assign_residuals,float);P=np.asarray(precision,float);require(e.ndim==2 and e.shape[1]==6 and P.shape==(6,6),'Orbit spatial shape mismatch');z=_inverse_power(e,power);log=-.5*np.einsum('bi,ij,bj->b',z,P,z,optimize=True)
    singular=np.zeros(len(e),bool)
    if include_transform_jacobian and abs(float(power)-1.)>1e-15:
        ae=np.abs(e);singular=(ae<NON_GAUSSIAN_MIN_ABS_RESIDUAL).any(axis=1);ae=np.maximum(ae,NON_GAUSSIAN_MIN_ABS_RESIDUAL);log+=np.sum((1./float(power)-1.)*np.log(ae)-math.log(float(power)),axis=1)
    return log,singular

def eval_m3(source_residuals,precision,power=1.0,tie_tolerance=1e-12,keep_probs=True):
    e=np.asarray(source_residuals,float);require(e.shape==(6,) and np.isfinite(e).all(),'M3 residual source invalid');assign=e[PERMS];lw,_=spatial_log(assign,precision,power,False);p=normalize_logweights(lw);scores=np.abs(assign[:,5]);obs=abs(float(e[5]));tail=float(p[upper_tail_mask(scores,obs,tie_tolerance,SCORE_REL_TOL)].sum());src=np.bincount(PERMS[:,5],weights=p,minlength=6).astype(float);require(abs(src.sum()-1)<1e-10,'M3 source marginal normalization failure');ess=float(1/np.sum(src*src));return {'pvalue':min(max(tail,0.),1.),'source_marginal':src,'orbit_probs':p if keep_probs else None,'orbit_logweights':lw if keep_probs else None,'identity_score':obs,'ess':ess,'max_weight':float(src.max()),'state_count':720}

def eval_m4(m3_result,source_residuals,log_dose_ratio,tie_tolerance=1e-12):
    s=np.asarray(m3_result['source_marginal'],float);e=np.asarray(source_residuals,float);d=np.asarray(log_dose_ratio,float);require(s.shape==(6,) and e.shape==(6,) and d.shape==(6,),'M4 source arrays invalid');logs=np.full(6,-np.inf);pos=s>0;logs[pos]=np.log(s[pos])+d[pos];q=normalize_logweights(logs);obs=abs(float(e[5]));score=np.abs(e);pv=float(q[upper_tail_mask(score,obs,tie_tolerance,SCORE_REL_TOL)].sum());ess=float(1/np.sum(q*q));return {'pvalue':min(max(pv,0.),1.),'source_probs':q,'source_logweights':logs,'ess':ess,'max_weight':float(q.max()),'normalization_count':1,'identity_score':obs}

def eval_m6(residual_matrix,treatment_log_matrix,log_inverse_scale_matrix,precision,power=1.0,tie_tolerance=1e-12,keep_probs=True):
    R=np.asarray(residual_matrix,float);T=np.asarray(treatment_log_matrix,float);J=np.asarray(log_inverse_scale_matrix,float);require(R.shape==(6,6) and T.shape==(6,6) and J.shape==(6,6),'M6 matrix shape invalid');require(np.isfinite(R).all(),'M6 residual matrix nonfinite');assign=R[np.arange(6)[None,:],PERMS];tl=T[np.arange(6)[None,:],PERMS];jl=J[np.arange(6)[None,:],PERMS];sl,sing=spatial_log(assign,precision,power,include_transform_jacobian=True);lw=np.sum(tl+jl,axis=1)+sl;p=normalize_logweights(lw);scores=np.abs(assign[:,5]);obs=abs(float(R[5,5]));# conservative: structural/singular states are included in upper tail
    mask=upper_tail_mask(scores,obs,tie_tolerance,SCORE_REL_TOL)|sing;pv=float(p[mask].sum());ess=float(1/np.sum(p*p));return {'pvalue':min(max(pv,0.),1.),'orbit_probs':p if keep_probs else None,'orbit_logweights':lw if keep_probs else None,'ess':ess,'max_weight':float(p.max()),'identity_score':obs,'singular_state_count':int(sing.sum()),'state_count':720}

def orbit_kl(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float);require(p.shape==q.shape,'Orbit KL shape mismatch');pos=p>0
    if np.any(q[pos]<=0):return np.inf
    v=float(np.sum(p[pos]*(np.log(p[pos])-np.log(q[pos])))); require(v>=-1e-12,f'Orbit KL negative beyond tolerance: {v}'); return float(max(0.0,v))

def invert_candidate(evaluator,lo=-1.,hi=1.,grid_points=4096,alpha=.1,bisect_iter=40,grid_pvalues=None,boundary_abs_tol=1e-6,boundary_rel_tol=1e-5):
    grid=np.linspace(float(lo),float(hi),int(grid_points));pv=np.asarray(grid_pvalues,float) if grid_pvalues is not None else np.array([float(evaluator(float(y))) for y in grid]);require(pv.shape==(len(grid),) and np.isfinite(pv).all(),'Candidate inversion p-value nonfinite');ok=pv>float(alpha);components=[]
    if ok.any():
        starts=np.flatnonzero(ok & np.r_[True,~ok[:-1]]);ends=np.flatnonzero(ok & np.r_[~ok[1:],True])
        for s,e in zip(starts,ends):
            left=float(grid[s]);right=float(grid[e])
            if s>0:
                a=float(grid[s-1]);b=float(grid[s])
                for _ in range(int(bisect_iter)):
                    if (b-a)<=float(boundary_abs_tol)+float(boundary_rel_tol)*max(abs(a),abs(b),1.0):break
                    m=.5*(a+b)
                    if evaluator(m)>alpha:b=m
                    else:a=m
                left=b
            if e<len(grid)-1:
                a=float(grid[e]);b=float(grid[e+1])
                for _ in range(int(bisect_iter)):
                    if (b-a)<=float(boundary_abs_tol)+float(boundary_rel_tol)*max(abs(a),abs(b),1.0):break
                    m=.5*(a+b)
                    if evaluator(m)>alpha:a=m
                    else:b=m
                right=a
            components.append((left,right))
    if components:
        hull_lo=components[0][0];hull_hi=components[-1][1];raw_width=float(sum(max(0,b-a) for a,b in components));hull_width=float(hull_hi-hull_lo);infl=float(hull_width-raw_width)
    else:hull_lo=np.nan;hull_hi=np.nan;raw_width=0.;hull_width=0.;infl=0.
    return {'components':components,'component_count':len(components),'hull_lower':hull_lo,'hull_upper':hull_hi,'raw_set_width':raw_width,'hull_width':hull_width,'hull_inflation':infl,'left_domain_truncated':bool(components and components[0][0]<=lo+1e-10),'right_domain_truncated':bool(components and components[-1][1]>=hi-1e-10),'grid_min_p':float(pv.min()),'grid_max_p':float(pv.max()),'grid_points':int(grid_points),'bisection_iterations':int(bisect_iter)}

def _normalize_logweight_matrix(logw):
    x=np.asarray(logw,float);require(x.ndim==2 and not np.isnan(x).any() and not np.isposinf(x).any(),'Grid orbit logweights invalid')
    fin=np.isfinite(x);require(fin.any(axis=1).all(),'R07_ALL_ORBIT_WEIGHTS_ZERO grid row')
    safe=np.where(fin,x,-np.inf);m=np.max(safe,axis=1,keepdims=True);w=np.where(fin,np.exp(safe-m),0.0);den=w.sum(axis=1,keepdims=True);require((den>0).all() and np.isfinite(den).all(),'Grid orbit normalization failed');return w/den

def _eval_m3_m4_grid_chunk(base5_residuals,candidate_y,center,precision,power,log_dose_ratio,tie_tolerance=SCORE_ABS_TOL):
    b=np.asarray(base5_residuals,float);yy=np.asarray(candidate_y,float);P=np.asarray(precision,float);d=np.asarray(log_dose_ratio,float)
    require(b.shape==(5,) and yy.ndim==1 and P.shape==(6,6) and d.shape==(6,),'Grid M3/M4 shape mismatch')
    src=np.empty((len(yy),6),float);src[:,:5]=b[None,:];src[:,5]=yy-float(center)
    assign=src[:,PERMS]
    z=_inverse_power(assign,power);lw=-.5*np.einsum('gbi,ij,gbj->gb',z,P,z,optimize=True);prob=_normalize_logweight_matrix(lw)
    score=np.abs(assign[:,:,5]);obs=np.abs(src[:,5])[:,None];p3=np.sum(prob*upper_tail_mask(score,obs,tie_tolerance,SCORE_REL_TOL),axis=1)
    H=np.zeros((720,6),float);H[np.arange(720),PERMS[:,5]]=1.0;sm=prob@H
    logs=np.where(sm>0,np.log(np.maximum(sm,1e-300))+d[None,:],-np.inf);q=_normalize_logweight_matrix(logs)
    p4=np.sum(q*upper_tail_mask(np.abs(src),np.abs(src[:,5])[:,None],tie_tolerance,SCORE_REL_TOL),axis=1)
    return p3,p4

def eval_m3_m4_grid(base5_residuals,candidate_y,center,precision,power,log_dose_ratio,tie_tolerance=SCORE_ABS_TOL,chunk_size=384):
    """Chunked vectorized M3/M4 grid evaluator; algebraically identical to scalar definitions.
    Chunking keeps the D2-lineage 4096-point grid memory-safe on Windows.
    """
    yy=np.asarray(candidate_y,float);o3=[];o4=[]
    for a in range(0,len(yy),int(chunk_size)):
        p3,p4=_eval_m3_m4_grid_chunk(base5_residuals,yy[a:a+int(chunk_size)],center,precision,power,log_dose_ratio,tie_tolerance);o3.append(p3);o4.append(p4)
    return np.concatenate(o3) if o3 else np.empty(0),np.concatenate(o4) if o4 else np.empty(0)

def _eval_m6_grid_chunk(R0,T,J,candidate_y,precision,power,tie_tolerance=SCORE_ABS_TOL):
    R0=np.asarray(R0,float);T=np.asarray(T,float);J=np.asarray(J,float);yy=np.asarray(candidate_y,float);P=np.asarray(precision,float)
    require(R0.shape==(6,6) and T.shape==(6,6) and J.shape==(6,6) and yy.ndim==1 and P.shape==(6,6),'Grid M6 shape mismatch')
    base=R0[np.arange(6)[None,:],PERMS];mask=(PERMS==5).astype(float);assign=base[None,:,:]+yy[:,None,None]*mask[None,:,:]
    tl=T[np.arange(6)[None,:],PERMS];jl=J[np.arange(6)[None,:],PERMS]
    z=_inverse_power(assign,power);sl=-.5*np.einsum('gbi,ij,gbj->gb',z,P,z,optimize=True);sing=np.zeros(sl.shape,bool)
    if abs(float(power)-1.)>1e-15:
        ae=np.abs(assign);sing=(ae<NON_GAUSSIAN_MIN_ABS_RESIDUAL).any(axis=2);ae=np.maximum(ae,NON_GAUSSIAN_MIN_ABS_RESIDUAL);sl+=np.sum((1./float(power)-1.)*np.log(ae)-math.log(float(power)),axis=2)
    lw=sl+np.sum(tl+jl,axis=1)[None,:];prob=_normalize_logweight_matrix(lw)
    score=np.abs(assign[:,:,5]);obs=np.abs(R0[5,5]+yy)[:,None];return np.sum(prob*(upper_tail_mask(score,obs,tie_tolerance,SCORE_REL_TOL)|sing),axis=1)

def eval_m6_grid(R0,T,J,candidate_y,precision,power,tie_tolerance=SCORE_ABS_TOL,chunk_size=384):
    """Chunked vectorized M6 conservative p-values for the frozen candidate grid."""
    yy=np.asarray(candidate_y,float);out=[]
    for a in range(0,len(yy),int(chunk_size)):out.append(_eval_m6_grid_chunk(R0,T,J,yy[a:a+int(chunk_size)],precision,power,tie_tolerance))
    return np.concatenate(out) if out else np.empty(0)
