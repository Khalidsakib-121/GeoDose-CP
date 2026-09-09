from __future__ import annotations
import hashlib, math
from typing import Dict, Tuple
import numpy as np
from scipy.special import expit, betaln, ndtr
from .io import require

DOSES=np.array([0.0,0.10,0.25,0.50,0.75,0.90,1.0],dtype=float)

def robust_z(x: np.ndarray) -> Tuple[np.ndarray,float,float]:
    x=np.asarray(x,dtype=float)
    med=float(np.nanmedian(x)); mad=float(np.nanmedian(np.abs(x-med)))
    scale=1.4826*mad
    if not np.isfinite(scale) or scale < 1e-10:
        scale=float(np.nanstd(x,ddof=0))
    if not np.isfinite(scale) or scale < 1e-10: scale=1.0
    return (x-med)/scale, med, scale

def softmax3(a,b,c):
    x=np.column_stack([a,b,c]).astype(float)
    m=np.max(x,axis=1,keepdims=True); e=np.exp(x-m); return e/e.sum(axis=1,keepdims=True)

def beta_log_pdf(x,a,b):
    x=np.asarray(x,float); a=np.asarray(a,float); b=np.asarray(b,float)
    return (a-1)*np.log(x)+(b-1)*np.log1p(-x)-betaln(a,b)

def treatment_parameters(case: Dict, xpv, xrain, xsoc, xslope, xcoord, hidden_u):
    n=len(xpv)
    conf=case['confounding']; overlap=case['overlap']; endpoint=case['endpoint_structure']
    if conf=='observed_none':
        eta=np.zeros(n); l0=np.full(n,-1.55); l1=np.full(n,-1.70)
    else:
        strength=1.0 if conf in {'observed_strong','hidden_C2_negative_control'} else 0.65
        eta=-0.10+strength*(0.80*xpv-0.50*xrain+0.35*xsoc+0.20*xcoord)
        l0=-1.60+strength*(0.45*xpv-0.30*xsoc+0.10*xcoord)
        l1=-1.80+strength*(-0.35*xpv+0.50*xrain+0.20*xslope)
        if conf=='hidden_C2_negative_control':
            eta=eta+1.00*hidden_u; l0=l0-0.55*hidden_u; l1=l1+0.55*hidden_u
    if overlap=='good': kappa=6.0
    elif overlap=='moderate': kappa=10.0
    elif overlap=='poor':
        eta=eta-1.40; l1=l1-2.0; l0=l0+0.6; kappa=24.0
    elif overlap=='severe':
        eta=eta-2.20; l1=l1-3.5; l0=l0+1.0; kappa=45.0
    else: raise ValueError(overlap)
    if endpoint=='interior_only': probs=np.column_stack([np.zeros(n),np.zeros(n),np.ones(n)])
    else:
        if endpoint=='mixed_atoms_strong': l0=l0+0.75; l1=l1+0.55
        probs=softmax3(l0,l1,np.zeros(n))
    mean=expit(eta); alpha=np.clip(mean*kappa,1,100); beta=np.clip((1-mean)*kappa,1,100)
    return probs,alpha,beta,alpha/(alpha+beta)

def draw_mixed(rng,probs,alpha,beta):
    interior=rng.beta(alpha,beta); u=rng.random(len(alpha)); cat=np.where(u<probs[:,0],0,np.where(u<probs[:,0]+probs[:,1],1,2))
    a=interior.copy(); a[cat==0]=0.; a[cat==1]=1.; return a,cat,interior,u

def mixed_density_at(a,probs,alpha,beta):
    a=np.asarray(a,float); out=np.zeros(len(a)); z=a==0.; o=a==1.; interior=(a>0)&(a<1)
    out[z]=probs[z,0]; out[o]=probs[o,1]
    if interior.any(): out[interior]=probs[interior,2]*np.exp(beta_log_pdf(a[interior],alpha[interior],beta[interior]))
    return out

def truncated_normal_density(a, center: float, h: float):
    a=np.asarray(a,float); lo=ndtr((0-center)/h); hi=ndtr((1-center)/h); Z=hi-lo
    return np.where((a>0)&(a<1), np.exp(-0.5*((a-center)/h)**2)/(h*math.sqrt(2*math.pi)*Z),0.0)

def q_at(a, dose: float, h: float|None, endpoint_audited: bool=True):
    a=np.asarray(a,float)
    if dose==0.: return (a==0.).astype(float) if endpoint_audited else np.zeros(len(a))
    if dose==1.: return (a==1.).astype(float) if endpoint_audited else np.zeros(len(a))
    require(h is not None and h>0,'Interior localized intervention requires positive bandwidth')
    return truncated_normal_density(a,float(dose),float(h))

def sample_localized(rng,dose: float,h: float|None,endpoint_audited: bool=True)->float:
    if dose in (0.,1.):
        require(endpoint_audited,'Requested endpoint is not audited'); return float(dose)
    require(h is not None and h>0,'Interior target requires bandwidth')
    # inverse-CDF draw from N(dose,h^2) truncated to (0,1)
    lo=ndtr((0-dose)/h); hi=ndtr((1-dose)/h); u=float(rng.uniform(lo,hi))
    # scipy-free inverse normal via scipy.special.ndtri imported lazily
    from scipy.special import ndtri
    return float(dose+h*ndtri(u))

def outcome_baseline(case: Dict, pv_trend,xpv,xrain,xsoc,xslope,xawc,xcoord,hidden_u):
    if case['outcome_form']=='linear':
        base=0.60*pv_trend + 0.030*xpv -0.018*xrain +0.010*xsoc
    else:
        base=(0.65*pv_trend +0.032*xpv-0.020*xrain+0.012*(xsoc*xsoc-1.0)
              +0.015*np.sin(np.pi*xcoord)-0.010*xslope+0.008*xawc)
    if case['confounding']=='hidden_C2_negative_control': base=base+0.050*hidden_u
    return base

def conditional_mean(case: Dict, baseline,xpv,a):
    a=np.asarray(a,float)
    if case['outcome_form']=='linear': return baseline+0.050*a
    return baseline + 0.055*a -0.035*a*a +0.025*np.sin(2*np.pi*a)+0.020*a*xpv

def measurement_error(case: Dict,a,ue,substrate,z):
    a=np.asarray(a,float)
    if case['measurement_error']=='none': return np.zeros_like(a)
    sigma=0.012+0.0015*np.clip(np.asarray(ue,float),0,50)
    rand=sigma*np.asarray(z,float)
    tr=float(case.get('measurement_lambda',0.0))*(a-0.5)
    ss=float(case.get('measurement_substrate_lambda',0.0))*np.asarray(substrate,float)
    mode=case['measurement_error']
    if mode=='random': return rand
    if mode=='treatment_correlated': return rand+tr
    if mode=='substrate_bias': return rand+ss
    if mode=='combined': return rand+tr+ss
    raise ValueError(mode)

def empirical_target_design_ratio(d,delta: float):
    d=np.asarray(d,float); logw=delta*d; m=np.max(logw); w=np.exp(logw-m); return w/np.mean(w)

def empirical_target_design_ratio_stratified(d, strata, delta: float):
    """Finite-frame ratio for an observational/target design that preserves equal mine mass.
    Within each mine, observational slots are uniform and the target frame is exponentially tilted.
    Therefore E_obs[r(U) | mine] = 1 exactly (up to floating point).
    """
    d=np.asarray(d,float); strata=np.asarray(strata).astype(str); out=np.empty(len(d),float)
    for g in np.unique(strata):
        idx=np.flatnonzero(strata==g); out[idx]=empirical_target_design_ratio(d[idx],delta)
    return out

def stable_seed(namespace: str, *parts: object, mod: int=2_147_483_647) -> int:
    s='|'.join([namespace,*map(str,parts)]).encode(); v=int.from_bytes(hashlib.sha256(s).digest()[:8],'big')%mod
    return int(v if v>0 else 1)
