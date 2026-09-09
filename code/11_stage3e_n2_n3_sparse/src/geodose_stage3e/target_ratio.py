from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from .io import require, Stage3EError

@dataclass
class RatioFit:
    beta0: float
    beta1: float
    converged: bool
    n_observational: int
    n_target: int
    prior_odds: float
    raw_normalizer_analytic: float


def fit_normal_shift_logistic(training: pd.DataFrame) -> RatioFit:
    """Diagnostic classifier for the frozen Stage3B normal-location-shift fixture.

    This fit is intentionally *not* assigned the bounded-feature Hoeffding
    certificate from Unified Math 10.2 because D is Gaussian and unbounded.
    """
    d=training.copy()
    require(set(d['target_design_class'].astype(str).unique())=={'observational_design','target_design'},'RATIO_TRAINING_CLASSES_INVALID')
    x=d['X_nonlinear_1'].to_numpy(dtype=float)
    y=(d['target_design_class'].astype(str)=='target_design').to_numpy(dtype=float)
    require(np.all(np.isfinite(x)),'RATIO_FEATURE_NONFINITE')
    n0=int(np.sum(y==0)); n1=int(np.sum(y==1)); require(n0>0 and n1>0,'RATIO_TRAINING_CLASS_EMPTY')
    def obj(b):
        eta=b[0]+b[1]*x
        # log(1+exp eta) - y eta, stable
        return float(np.mean(np.logaddexp(0.0,eta)-y*eta))
    def grad(b):
        eta=b[0]+b[1]*x; p=expit(eta); g=p-y
        return np.array([np.mean(g),np.mean(g*x)],dtype=float)
    res=minimize(obj,np.zeros(2),jac=grad,method='BFGS',options={'gtol':1e-12,'maxiter':1000})
    if not bool(res.success) and float(np.linalg.norm(grad(res.x)))>1e-8:
        raise Stage3EError(f'RATIO_LOGISTIC_FIT_FAILED: {res.message}; grad={np.linalg.norm(grad(res.x))}')
    b0,b1=map(float,res.x); prior=float(n0/n1)
    # Under observational D~N(0,1), E prior*exp(b0+b1 D)=prior*exp(b0+b1^2/2).
    z=prior*math.exp(b0+0.5*b1*b1)
    require(math.isfinite(z) and z>0,'RATIO_ANALYTIC_NORMALIZER_INVALID')
    return RatioFit(b0,b1,True,n0,n1,prior,z)


def normalized_ratio_from_fit(fit: RatioFit, x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=float)
    # prior odds and intercept cancel after exact observational normalization.
    return np.exp(fit.beta1*x-0.5*fit.beta1*fit.beta1)


def raw_ratio_from_fit(fit: RatioFit, x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=float)
    return fit.prior_odds*np.exp(fit.beta0+fit.beta1*x)


def normal_shift_true_to_fitted_kl(delta_true: float, fitted_slope: float) -> float:
    # N(delta_true,1) || N(fitted_slope,1)
    return 0.5*(float(delta_true)-float(fitted_slope))**2


def theorem_eligibility_for_stage3b(mode: str, feature_distribution: str='normal') -> tuple[bool,str]:
    if str(mode)=='identity': return True,'EXACT_IDENTITY_RATIO_NO_ESTIMATION'
    if str(mode)=='normal_location_shift' and feature_distribution=='normal':
        return False,'THEOREM_INELIGIBLE_UNBOUNDED_GAUSSIAN_FEATURE_FOR_BOUNDED_FEATURE_HOEFFDING_CERTIFICATE'
    return False,'THEOREM_INELIGIBLE_UNPROVED_RATIO_TRAINING_REGIME'

def empirical_normalized_ratio_from_fit(fit: RatioFit, x: np.ndarray, observational_normalization_x: np.ndarray) -> tuple[np.ndarray,float]:
    """Apply the required class-prior odds factor, then normalize on a frozen observational frame.

    The returned ratios have empirical mean exactly one on the supplied observational
    normalization frame (up to floating-point error). This is an implementation audit
    object; theorem eligibility is still governed separately by the actual training design.
    """
    x=np.asarray(x,dtype=float); x0=np.asarray(observational_normalization_x,dtype=float)
    require(x0.size>0 and np.all(np.isfinite(x0)) and np.all(np.isfinite(x)),'RATIO_EMPIRICAL_NORMALIZATION_INPUT_INVALID')
    raw0=raw_ratio_from_fit(fit,x0); z=float(np.mean(raw0))
    require(math.isfinite(z) and z>0.0,'RATIO_EMPIRICAL_NORMALIZER_INVALID')
    return raw_ratio_from_fit(fit,x)/z,z


def bounded_logistic_ratio_certificate(*, B_phi: float, B_infinity: float, R: float, gamma: float, p_r: int, K_r: int, delta_r: float, n_r: int) -> dict[str,float]:
    """Unified Math 10.2 bounded-feature target-design ratio certificate.

    This implements the frozen finite-sample formula only. Callers must separately
    verify the bounded-feature and dependency-graph assumptions; the Stage3B
    Gaussian design-shift fixture is deliberately marked ineligible.
    """
    vals=[B_phi,B_infinity,R,gamma,delta_r]
    require(all(math.isfinite(float(v)) for v in vals),'RATIO_CERTIFICATE_NONFINITE')
    require(B_phi>0 and B_infinity>0 and R>=0 and gamma>0 and int(p_r)>0 and int(K_r)>0 and 0<delta_r<1 and int(n_r)>0,'RATIO_CERTIFICATE_CONFIG_INVALID')
    M=float(R)*float(B_phi)
    # Stable logistic curvature w_M=e^M/(1+e^M)^2 = expit(M)*(1-expit(M)).
    pM=float(expit(M)); wM=pM*(1.0-pM)
    require(wM>0.0 and math.isfinite(wM),'RATIO_CERTIFICATE_CURVATURE_ZERO')
    lam=wM*float(gamma)
    grad_inf=float(B_infinity)*math.sqrt(2.0*int(K_r)*math.log(2.0*int(K_r)*int(p_r)/float(delta_r))/int(n_r))
    grad_l2=float(B_infinity)*math.sqrt(2.0*int(p_r)*int(K_r)*math.log(2.0*int(K_r)*int(p_r)/float(delta_r))/int(n_r))
    beta_l2=(2.0*float(B_infinity)/lam)*math.sqrt(2.0*int(p_r)*int(K_r)*math.log(2.0*int(K_r)*int(p_r)/float(delta_r))/int(n_r))
    eta=float(B_phi)*beta_l2
    eps=2.0*eta
    # Equation 10.20, algebraically equivalent to 4 B_phi B_inf/lambda * sqrt(...).
    eps_direct=(4.0*float(B_phi)*float(B_infinity)/lam)*math.sqrt(2.0*int(p_r)*int(K_r)*math.log(2.0*int(K_r)*int(p_r)/float(delta_r))/int(n_r))
    require(abs(eps-eps_direct)<=1e-12*max(1.0,abs(eps_direct)),'RATIO_CERTIFICATE_FORMULA_INTERNAL_MISMATCH')
    return {'M':M,'w_M':wM,'lambda_r':lam,'gradient_infinity_bound':grad_inf,'gradient_l2_bound':grad_l2,'beta_l2_error_bound':beta_l2,'eta_r_n':eta,'epsilon_r_n':eps,'target_design_KL_upper_bound':eps,'failure_probability_delta_r':float(delta_r)}
