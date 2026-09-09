from __future__ import annotations
import math
from .io import require

def pinsker(kl: float) -> float:
    require(kl>=0 and math.isfinite(kl),'N3_KL_INVALID')
    return min(1.0,math.sqrt(kl/2.0))

def log_oscillation_tv(omega: float) -> float:
    require(omega>=0 and math.isfinite(omega),'N3_OSCILLATION_INVALID')
    return math.tanh(omega/4.0)

def tight_same_discrepancy_bound(*, kl: float|None=None, omega: float|None=None) -> dict[str,float|str]:
    vals=[]
    if kl is not None: vals.append(('pinsker_kl',pinsker(float(kl))))
    if omega is not None: vals.append(('log_oscillation',log_oscillation_tv(float(omega))))
    require(vals,'N3_NO_DISCREPANCY_BOUND')
    method,value=min(vals,key=lambda x:x[1])
    return {'method':method,'tv_bound':float(value),'candidate_count':len(vals)}

def coverage_lower_bound(alpha: float, delta_sparse: float, delta_mis: float, delta_est: float, delta_comp: float, eta: float) -> float:
    for name,v in [('alpha',alpha),('delta_sparse',delta_sparse),('delta_mis',delta_mis),('delta_est',delta_est),('delta_comp',delta_comp),('eta',eta)]:
        require(math.isfinite(float(v)) and float(v)>=0,f'N3_COMPONENT_INVALID:{name}')
    require(0<alpha<1,'N3_ALPHA_INVALID')
    return max(0.0,1.0-alpha-delta_sparse-delta_mis-delta_est-delta_comp-eta)

def n3_record(*, alpha: float, delta_sparse: float, delta_mis: float=0.0, delta_est: float=0.0, delta_comp: float=0.0, eta: float=0.0, theorem_eligible: bool, eligibility_reason: str) -> dict[str,object]:
    lb=coverage_lower_bound(alpha,delta_sparse,delta_mis,delta_est,delta_comp,eta)
    return {'alpha':float(alpha),'delta_sparse':float(delta_sparse),'delta_mis':float(delta_mis),'delta_est':float(delta_est),'delta_comp':float(delta_comp),'certificate_failure_probability_eta':float(eta),'coverage_lower_bound':float(lb),'theorem_eligible':bool(theorem_eligible),'eligibility_reason':str(eligibility_reason),'nonvacuous':bool(lb>0.0)}
