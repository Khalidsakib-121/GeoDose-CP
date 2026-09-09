from __future__ import annotations
import ast, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np, pandas as pd, yaml
from scipy import sparse
from geodose_stage3e.ordering import deterministic_maximin_order
from geodose_stage3e.gaussian_n2 import build_vecchia_family, n2_tv_bound, target_shift_identity, gaussian_kl_from_precisions, full_predecessor_variances, target_weighted_n2
from geodose_stage3e.target_ratio import fit_normal_shift_logistic, normalized_ratio_from_fit, empirical_normalized_ratio_from_fit, theorem_eligibility_for_stage3b, bounded_logistic_ratio_certificate
from geodose_stage3e.spatial_confidence import normalized_adjacency, rho_score_confidence_set, RhoConfidenceResult, gmrf_boundary_certificate
from geodose_stage3e.n3 import coverage_lower_bound, tight_same_discrepancy_bound
from geodose_stage3e.version import VERSION

def check(name,fn):
    fn(); print('PASS',name)


def t_full_predecessor_dense_reference():
    rng=np.random.default_rng(20260809); A=rng.normal(size=(9,9)); Q=sparse.csr_matrix(A.T@A+2.0*np.eye(9))
    got,meta=full_predecessor_variances(Q)
    Sigma=np.linalg.solve(Q.toarray(),np.eye(9))
    truth=[]
    for i in range(9):
        if i==0: truth.append(float(Sigma[i,i]))
        else:
            S=Sigma[:i,:i]; s=Sigma[:i,i]; truth.append(float(Sigma[i,i]-s@np.linalg.solve(S,s)))
    assert np.max(np.abs(got-np.asarray(truth)))<2e-12 and meta['reverse_lu_no_pivot']

def t_n2_sum_matches_global_gaussian_kl():
    n=14; edges=pd.DataFrame({'source_node':np.arange(n-1),'target_node':np.arange(1,n)})
    from geodose_stage3e.spatial_confidence import affine_precision_from_S
    S=normalized_adjacency(n,edges); Q=affine_precision_from_S(S,.55,.35); c=np.column_stack([np.arange(n),np.zeros(n)]); ids=np.arange(n)
    fam,_=build_vecchia_family(Q,np.arange(n),c,ids,[0,2,4],minimum_precision_eigenvalue=1e-10,kl_nonnegative_tolerance=1e-9)
    for m,v in fam.items():
        kl,_=gaussian_kl_from_precisions(Q,v.precision); assert abs(kl-v.total_kl)<2e-9,(m,kl,v.total_kl)

def t_generic_target_weighted_n2():
    L=np.array([.1,.2,.3,.5]); r=np.array([.5,.8,1.2,1.5]); o=target_weighted_n2(L,r)
    assert abs(o['mean_target_design_ratio']-1)<1e-15
    assert abs(o['target_weighted_delta_N2']-np.mean(L*r))<1e-15
    assert abs(o['target_weighted_delta_N2']-o['identity_rhs'])<1e-15 and o['target_weighted_delta_N2']<=o['cauchy_schwarz_upper_bound']+1e-15

def t_ratio_empirical_normalization_and_prior_odds():
    rng=np.random.default_rng(2026); n0=500;n1=250; x0=rng.normal(size=n0);x1=rng.normal(.75,1,size=n1)
    d=pd.DataFrame({'X_nonlinear_1':np.r_[x0,x1],'target_design_class':['observational_design']*n0+['target_design']*n1})
    fit=fit_normal_shift_logistic(d); rr,z=empirical_normalized_ratio_from_fit(fit,x0,x0)
    assert abs(rr.mean()-1)<2e-15 and abs(fit.prior_odds-2.0)<1e-15 and z>0

def t_bounded_ratio_certificate_formula():
    o=bounded_logistic_ratio_certificate(B_phi=2,B_infinity=1,R=1,gamma=.5,p_r=3,K_r=1,delta_r=.05,n_r=500)
    assert o['epsilon_r_n']>0 and abs(o['target_design_KL_upper_bound']-o['epsilon_r_n'])<1e-15 and o['lambda_r']>0

def t_structural_boundary_at_zero_rho():
    n=8; edges=pd.DataFrame({'source_node':np.arange(n-1),'target_node':np.arange(1,n)}); S=normalized_adjacency(n,edges)
    conf=RhoConfidenceResult(0.0,-1,[(0.0,.5)],.5,.05,1,1,1e-4,True)
    out=gmrf_boundary_certificate(S,.35,conf,np.array([3,4]),np.zeros(2))
    # Frozen graph boundary contains nodes 2 and 5 even though Q(rho_hat=0) has zero off-diagonals.
    assert out['boundary_size']==2

def t_ordering():
    c=np.array([[0,0],[1,0],[0,1],[1,1]],float); ids=np.array([3,2,1,0]); a=deterministic_maximin_order(c,ids); b=deterministic_maximin_order(c,ids); assert np.array_equal(a,b) and len(set(a))==4

def t_iid_zero():
    n=12; Q=sparse.eye(n,format='csr')/0.35**2; c=np.column_stack([np.arange(n),np.zeros(n)]); ids=np.arange(n); order=np.arange(n)
    fam,meta=build_vecchia_family(Q,order,c,ids,[0,1,4],minimum_precision_eigenvalue=1e-10,kl_nonnegative_tolerance=1e-9)
    assert max(x.total_kl for x in fam.values())<1e-12 and meta['dense_inverse_formed'] is False

def t_ar1_exact():
    n=30; r=.6; den=1-r*r; d=np.full(n,(1+r*r)/den); d[0]=d[-1]=1/den; off=np.full(n-1,-r/den); Q=sparse.diags([off,d,off],[-1,0,1],format='csr'); c=np.column_stack([np.arange(n),np.zeros(n)]); ids=np.arange(n)
    fam,_=build_vecchia_family(Q,np.arange(n),c,ids,[1],minimum_precision_eigenvalue=1e-10,kl_nonnegative_tolerance=1e-9); assert fam[1].total_kl<1e-10

def t_monotone():
    n=20; edges=pd.DataFrame({'source_node':np.arange(n-1),'target_node':np.arange(1,n)}); from geodose_stage3e.spatial_confidence import normalized_adjacency, affine_precision_from_S
    S=normalized_adjacency(n,edges); Q=affine_precision_from_S(S,.5,.35); c=np.column_stack([np.arange(n),np.zeros(n)]); ids=np.arange(n)
    fam,_=build_vecchia_family(Q,np.arange(n),c,ids,[0,1,2,4],minimum_precision_eigenvalue=1e-10,kl_nonnegative_tolerance=1e-9); vals=[fam[m].total_kl for m in [0,1,2,4]]; assert all(vals[i+1]<=vals[i]+1e-10 for i in range(len(vals)-1))

def t_kl_identity():
    Q=sparse.diags([2.,3.,4.],format='csr'); kl,_=gaussian_kl_from_precisions(Q,Q); assert kl<1e-12

def t_pinsker(): assert abs(n2_tv_bound(.02)-.1)<1e-15

def t_shift_identity():
    o=target_shift_identity(np.array([.5,.8,1.2,1.5]),np.array([.1,.2,.3,.5])); assert abs(o['target_weighted_delta']-o['identity_rhs'])<1e-15 and o['target_weighted_delta']<=o['cs_upper_bound']+1e-15

def t_ratio_fit():
    rng=np.random.default_rng(123); n0=1200;n1=600; delta=.75; x0=rng.normal(size=n0);x1=rng.normal(delta,1,size=n1); d=pd.DataFrame({'X_nonlinear_1':np.r_[x0,x1],'target_design_class':['observational_design']*n0+['target_design']*n1})
    fit=fit_normal_shift_logistic(d); assert abs(fit.beta1-delta)<.12; rr=normalized_ratio_from_fit(fit,np.array([0.])); assert rr[0]>0

def t_ratio_scope():
    assert theorem_eligibility_for_stage3b('identity')[0] is True and theorem_eligibility_for_stage3b('normal_location_shift')[0] is False

def t_s3_nonempty():
    n=30; edges=pd.DataFrame({'source_node':np.arange(n-1),'target_node':np.arange(1,n)}); S=normalized_adjacency(n,edges); rng=np.random.default_rng(5); e=.35*rng.normal(size=n); c=rho_score_confidence_set(S,e,.35,(0,.95),.05,1e-3); assert c.exact_set_nonempty and c.radius>=0

def t_gmrf_zero_radius():
    n=10; edges=pd.DataFrame({'source_node':np.arange(n-1),'target_node':np.arange(1,n)}); S=normalized_adjacency(n,edges); conf=RhoConfidenceResult(.4,-1,[(.4,.4)],0,.05,1,0,1e-4,True); B=np.array([4,5]); from geodose_stage3e.spatial_confidence import affine_precision_from_S,graph_boundary_from_precision; Q=affine_precision_from_S(S,.4,.35); D=graph_boundary_from_precision(Q,B); out=gmrf_boundary_certificate(S,.35,conf,B,np.zeros(len(D))); assert out['eligible'] and abs(out['kl_upper_bound'])<1e-15

def t_n3_arithmetic(): assert abs(coverage_lower_bound(.1,.02,.01,.03,0,.04)-.8)<1e-15

def t_no_double_count():
    x=tight_same_discrepancy_bound(kl=.02,omega=.4); assert abs(x['tv_bound']-min(.1,math.tanh(.1)))<1e-15

def t_contract():
    c=yaml.safe_load((ROOT/'configs'/'stage3e_contract.yaml').read_text()); assert c['primary_pilot_neighborhood_size'] in c['candidate_neighborhood_sizes']; assert c['operational_ess_threshold'] is None and c['operational_max_weight_threshold'] is None

def t_hash_registry():
    h=json.loads((ROOT/'configs'/'expected_upstream_hashes.json').read_text()); assert len(h)==4 and all(len(v)==64 for v in h.values())

def t_api_lock():
    o=json.loads((ROOT/'configs'/'d2_api_lock.json').read_text()); assert 'non_gaussian_min_abs_residual' in o['signatures']['CandidateEvaluator']; assert 'minimum_precision_eigenvalue_required' in o['signatures']['prepare_orbit']

def t_version(): assert VERSION=='1.0.0-stage3e-freeze'

def t_python310_grammar():
    for p in ROOT.rglob('*.py'): ast.parse(p.read_text(encoding='utf-8'),filename=str(p),feature_version=(3,10))

def t_no_dense_inverse_source():
    bad=[]
    for p in (ROOT/'src').rglob('*.py'):
        txt=p.read_text(encoding='utf-8')
        if 'np.linalg.inv(' in txt or 'numpy.linalg.inv(' in txt: bad.append(str(p))
    assert not bad,bad

def t_no_mc_penalty():
    c=yaml.safe_load((ROOT/'configs'/'stage3e_contract.yaml').read_text()); assert c['computation_delta']==0.0 and 'no_monte_carlo' in c['computation_delta_basis']

def t_ratio_separation():
    txt=(ROOT/'docs'/'Stage3E_Math_Algorithm_Contract.md').read_text(encoding='utf-8') if (ROOT/'docs'/'Stage3E_Math_Algorithm_Contract.md').exists() else ''
    assert 'q_h/g' in txt and 'target-design ratio' in txt

def main():
    tests=[('full_predecessor_variance_dense_reference',t_full_predecessor_dense_reference),('N2_sum_matches_global_Gaussian_KL',t_n2_sum_matches_global_gaussian_kl),('generic_target_weighted_N2',t_generic_target_weighted_n2),('target_ratio_empirical_normalization_prior_odds',t_ratio_empirical_normalization_and_prior_odds),('bounded_ratio_certificate_formula',t_bounded_ratio_certificate_formula),('structural_boundary_zero_rho',t_structural_boundary_at_zero_rho),('ordering_deterministic',t_ordering),('iid_N2_zero',t_iid_zero),('AR1_exact_sparse',t_ar1_exact),('neighborhood_monotonicity',t_monotone),('Gaussian_KL_identity',t_kl_identity),('Pinsker_formula',t_pinsker),('target_shift_identity_CS',t_shift_identity),('target_ratio_fit_diagnostic',t_ratio_fit),('target_ratio_theorem_scope',t_ratio_scope),('S3_score_set_nonempty',t_s3_nonempty),('GMRF_C_zero_radius',t_gmrf_zero_radius),('N3_good_event_arithmetic',t_n3_arithmetic),('no_double_counting',t_no_double_count),('contract_thresholds_deferred',t_contract),('frozen_hash_registry',t_hash_registry),('D2_API_lock_complete',t_api_lock),('version_consistency',t_version),('Python310_grammar',t_python310_grammar),('no_dense_inverse_source',t_no_dense_inverse_source),('no_MC_penalty',t_no_mc_penalty),('treatment_vs_design_ratio_separation',t_ratio_separation)]
    for n,f in tests: check(n,f)
    print(f'STAGE3E UNIT TESTS PASSED ({len(tests)}/{len(tests)})')
if __name__=='__main__': main()
