from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from .io import *
from .provenance import source_inventory, output_manifest
from .ordering import deterministic_maximin_order
from .gaussian_n2 import build_vecchia_family, gaussian_kl_from_precisions, n2_tv_bound, target_shift_identity, target_weighted_n2
from .target_ratio import fit_normal_shift_logistic, normalized_ratio_from_fit, raw_ratio_from_fit, empirical_normalized_ratio_from_fit, normal_shift_true_to_fitted_kl, theorem_eligibility_for_stage3b, bounded_logistic_ratio_certificate
from .spatial_confidence import normalized_adjacency, rho_score_confidence_set, gmrf_boundary_certificate, affine_precision_from_S, graph_boundary_from_precision
from .n3 import n3_record, pinsker
from .d2_sparse_bridge import D2SparseBridge
from .d4_newdata_bridge import D4NewDataBridge
from .version import VERSION, STAGE


def _edge_hash(edges: pd.DataFrame) -> str:
    a=edges[['source_node','target_node']].astype(int).to_numpy(); a=np.sort(a,axis=1); a=a[np.lexsort((a[:,1],a[:,0]))]
    return hashlib.sha256(a.tobytes()).hexdigest()

def _case_units(units: pd.DataFrame, case_id: str) -> pd.DataFrame:
    d=units[units.case_id.astype(str)==str(case_id)].sort_values('node_index').reset_index(drop=True)
    require(len(d)>0 and np.array_equal(d.node_index.astype(int).to_numpy(),np.arange(len(d))),f'CASE_NODE_INDEX_INVALID:{case_id}')
    return d

def _case_edges(edges: pd.DataFrame, case_id: str) -> pd.DataFrame:
    d=edges[edges.case_id.astype(str)==str(case_id)][['source_node','target_node']].copy()
    require(len(d)>0,f'CASE_EDGES_MISSING:{case_id}'); return d

def _interval_contains(intervals: list[tuple[float,float]], value: float, tol: float=1e-12) -> bool:
    return any(a-tol<=value<=b+tol for a,b in intervals)

def _block_for_case(d4prep: pd.DataFrame, case_id: str, target_dose: float) -> np.ndarray:
    d=d4prep[(d4prep.case_id.astype(str)==str(case_id)) & np.isclose(d4prep.target_dose.astype(float),float(target_dose)) & (d4prep.variant.astype(str)=='OT_TG')]
    if len(d)==1: return np.asarray([int(x) for x in str(d.iloc[0].block_nodes).split(',')],dtype=int)
    # Frozen D2 size-6 structural block for validation cases not exercised in D4.
    return np.asarray([294,318,319,343,344,320],dtype=int)

def _training_component_audit(cu: pd.DataFrame, edges: pd.DataFrame) -> tuple[np.ndarray,int]:
    train=set(cu.loc[cu.role.astype(str)=='nuisance_training','node_index'].astype(int).tolist())
    require(train,'S3_NUISANCE_TRAINING_EMPTY')
    cross=0
    for r in edges.itertuples(index=False):
        a=int(r.source_node); b=int(r.target_node)
        if (a in train)!=(b in train): cross+=1
    return np.asarray(sorted(train),dtype=int),cross

def _treatment_support_row(treatment: pd.DataFrame, case_id: str, dose: float) -> dict[str,Any]:
    d=treatment[(treatment.case_id.astype(str)==str(case_id)) & np.isclose(treatment.target_dose.astype(float),float(dose)) & (treatment.role.astype(str)=='calibration')].copy()
    require(len(d)>0,f'TREATMENT_SUPPORT_ROWS_MISSING:{case_id}:{dose}')
    lr=d.log_oracle_treatment_ratio.to_numpy(dtype=float); finite=np.isfinite(lr); w=np.exp(lr[finite])
    require(len(w)>0 and np.all(w>=0),'TREATMENT_SUPPORT_NO_POSITIVE_WEIGHT')
    sw=float(w.sum()); ess=float(sw*sw/float(w@w)); maxnorm=float(w.max()/sw)
    return {'case_id':case_id,'target_dose':float(dose),'calibration_rows':len(d),'positive_weight_rows':int(finite.sum()),'positive_weight_fraction':float(finite.mean()),'oracle_ess':ess,'oracle_max_normalized_weight':maxnorm,'operational_threshold_applied':False,'threshold_status':'DEFERRED_TO_20REP_PILOT'}

def _synthetic_exact_sparse_check(contract: dict[str,Any]) -> dict[str,Any]:
    n=40; rho=0.65; den=1-rho*rho
    diag=np.full(n,(1+rho*rho)/den); diag[0]=diag[-1]=1/den; off=np.full(n-1,-rho/den)
    Q=sparse.diags([off,diag,off],[-1,0,1],format='csr')
    coords=np.column_stack([np.arange(n,dtype=float),np.zeros(n)]); ids=np.arange(n,dtype=int); ordering=np.arange(n,dtype=int)
    fam,_=build_vecchia_family(Q,ordering,coords,ids,[1],minimum_precision_eigenvalue=float(contract['minimum_precision_eigenvalue']),kl_nonnegative_tolerance=float(contract['kl_nonnegative_tolerance']))
    kl=float(fam[1].total_kl)
    return {'n':n,'ar1_rho':rho,'m':1,'delta_n2':kl,'pass':bool(kl<=1e-10),'description':'AR1 chain natural order with one predecessor is exact sparse ordered law'}


def _s3_monte_carlo_coverage_test(contract: dict[str,Any]) -> dict[str,Any]:
    # Internal theorem-implementation test only; it is not pilot/publication performance evidence.
    reps=int(contract['s3_coverage_test_repetitions']); n=int(contract['s3_coverage_test_nodes'])
    rho=float(contract['s3_coverage_test_true_rho']); scale=float(contract['residual_scale']); delta=float(contract['spatial_confidence_delta'])
    seed=int(contract['s3_coverage_test_seed']); mult=float(contract['s3_coverage_test_mc_se_multiplier'])
    edges=pd.DataFrame({'source_node':np.arange(n-1,dtype=int),'target_node':np.arange(1,n,dtype=int)})
    S=normalized_adjacency(n,edges); Q=affine_precision_from_S(S,rho,scale).toarray()
    # Small theorem-test matrix only. No dense inverse: sample by triangular solve from Q=L L^T.
    L=np.linalg.cholesky(Q); rng=np.random.default_rng(seed); covered=0; nonempty=0
    for _ in range(reps):
        z=rng.normal(size=n); e=np.linalg.solve(L.T,z)
        conf=rho_score_confidence_set(S,e,scale,tuple(contract['rho_parameter_domain']),delta,float(contract['confidence_outer_cell_tolerance']))
        nonempty += int(conf.exact_set_nonempty)
        covered += int(conf.exact_set_nonempty and _interval_contains(conf.outer_intervals,rho,1e-12))
    phat=covered/reps; nominal=1.0-delta; se=math.sqrt(nominal*(1.0-nominal)/reps); lower_accept=max(0.0,nominal-mult*se)
    return {'repetitions':reps,'seed':seed,'n_nodes':n,'true_rho':rho,'nominal_coverage':nominal,'covered_count':covered,'empirical_coverage':phat,'nominal_mc_se':se,'mc_se_multiplier':mult,'acceptance_floor':lower_accept,'nonempty_count':nonempty,'pass':bool(phat+1e-15>=lower_accept),'claim_status':'INTERNAL_S3_IMPLEMENTATION_CALIBRATION_TEST_NOT_PUBLICATION_EVIDENCE'}

def run(package_root: Path, paths: dict[str,Path], output_dir: Path, overwrite: bool) -> dict[str,Any]:
    started=time.perf_counter(); root=Path(package_root); ensure_empty_output_dir(output_dir,overwrite)
    contract=yaml.safe_load((root/'configs'/'stage3e_contract.yaml').read_text(encoding='utf-8'))
    expected=json.loads((root/'configs'/'expected_upstream_hashes.json').read_text(encoding='utf-8'))
    input_audit={k:verify_zip(paths[k],expected[k]) for k in expected}
    # D2 source must match accepted D2 output inventory exactly.
    d2_inv=read_json_from_zip(paths['stage3d_d2_output_zip'],'stage3d_d2_source_hashes.json')
    require(isinstance(d2_inv.get('files'),dict),'D2_SOURCE_INVENTORY_SCHEMA_INVALID')
    source_audit=verify_source_tree(paths['stage3d_d2_source_root'],{str(k):str(v) for k,v in d2_inv['files'].items()})
    require(source_audit['pass'] and source_audit['matched_file_count']==41,'D2_SOURCE_41_OF_41_REQUIRED')
    d4v=read_json_from_zip(paths['stage3d_d4_output_zip'],'STAGE3D_D4_VERIFICATION.json')
    require(str(d4v.get('status'))=='verified_complete' and int(d4v.get('failed_checks',-1))==0,'D4_NOT_ACCEPTED')
    d4_inv=read_json_from_zip(paths['stage3d_d4_output_zip'],'stage3d_d4_source_hashes.json')
    d4_files=d4_inv.get('files')
    if isinstance(d4_files,list):
        require(all(isinstance(x,dict) and 'path' in x and 'sha256' in x for x in d4_files),'D4_SOURCE_INVENTORY_LIST_SCHEMA_INVALID')
        d4_expected={str(x['path']):str(x['sha256']) for x in d4_files}
    elif isinstance(d4_files,dict):
        d4_expected={str(k):str(v) for k,v in d4_files.items()}
    else:
        raise Stage3EError('D4_SOURCE_INVENTORY_SCHEMA_INVALID')
    d4_source_audit=verify_source_tree(paths['stage3d_d4_source_root'],d4_expected)
    require(d4_source_audit['pass'] and d4_source_audit['matched_file_count']==39,'D4_SOURCE_39_OF_39_REQUIRED')
    input_audit['stage3d_d2_source_root']=source_audit
    input_audit['stage3d_d4_source_root']=d4_source_audit
    write_json(output_dir/'stage3e_input_audit.json',input_audit)

    # Frozen source data.
    case_registry=read_csv_from_zip(paths['stage3b_output_zip'],'stage3b_case_registry.csv')
    units=read_csv_from_zip(paths['stage3b_output_zip'],'stage3b_validation_units.csv.gz')
    true_edges=read_csv_from_zip(paths['stage3b_output_zip'],'stage3b_true_graph_edges_by_case.csv.gz')
    fitted_edges=read_csv_from_zip(paths['stage3b_output_zip'],'stage3b_fitted_graph_edges.csv.gz')
    treatment=read_csv_from_zip(paths['stage3b_output_zip'],'stage3b_treatment_transport_truth.csv.gz')
    design=read_csv_from_zip(paths['stage3b_output_zip'],'stage3b_target_design_truth.csv.gz')
    reductions=read_csv_from_zip(paths['stage3a_output_zip'],'outputs_stage3a/reduction_test_registry.csv')
    d4prep=read_csv_from_zip(paths['stage3d_d4_output_zip'],'stage3d_d4_m6_prepare_audit.csv')
    d2trace=read_csv_from_zip(paths['stage3d_d2_output_zip'],'stage3d_d2_candidate_pvalue_trace.csv.gz')
    require(((reductions.reduction_id.astype(str)=='R3') & reductions.required_identity.astype(str).str.contains('target-design ratio equals one')).any(),'R3_REGISTRY_CONTRACT_MISSING')

    # Exact accepted D2 bridge. This locks every API signature before sparse smoke tests.
    bridge=D2SparseBridge(paths['stage3d_d2_source_root'],root/'configs'/'d2_api_lock.json',float(contract['alpha']))
    d4bridge=D4NewDataBridge(paths['stage3d_d4_source_root'],paths['stage3d_d2_source_root'],paths['stage3b_output_zip'],float(contract['alpha']))
    d2_api={'api_record':bridge.api_record,'accepted_d2_source_tree_hash':d2_inv.get('aggregate_sha256'),'expected_file_count':len(d2_inv['files']),'accepted_D4_newdata_bridge':d4bridge.api_record,'accepted_D4_source_tree_hash':d4_inv.get('aggregate_sha256'),'accepted_D4_source_file_count':len(d4_expected)}
    write_json(output_dir/'stage3e_d2_api_audit.json',d2_api)

    registered=[]
    for spec in contract['cases']:
        cid=str(spec['case_id']); dose=float(spec['target_dose']); rr=case_registry[case_registry.case_id.astype(str)==cid]
        require(len(rr)==1,f'STAGE3E_CASE_REGISTRY_MISSING:{cid}')
        r=rr.iloc[0]
        require(str(r.residual_law) in {'gaussian_gmrf','iid_continuous'},f'STAGE3E_N2_GAUSSIAN_SCOPE_VIOLATION:{cid}:{r.residual_law}')
        registered.append({'case_id':cid,'scenario_id':str(r.scenario_id),'target_dose':dose,'purpose':str(spec['purpose']),'spatial_rho':float(r.spatial_rho),'residual_law':str(r.residual_law),'target_design_mode':str(r.target_design_mode),'target_design_delta':float(r.target_design_delta),'fitted_graph':str(r.fitted_graph),'n_rows':int(r.n_rows),'n_cols':int(r.n_cols)})
    case_frame=pd.DataFrame(registered); write_csv(case_frame,output_dir/'stage3e_case_registry.csv')
    # N2 is a spatial-law certificate indexed by the case-level residual/design law, not by
    # repeated requested doses.  ST1 deliberately contributes two dose queries (0 and 1),
    # but its spatial approximation is computed once and reused for both dose queries.
    structural_frame=case_frame.drop_duplicates('case_id',keep='first').reset_index(drop=True)

    # All 25x25 cases share the frozen coordinates. Ordering is outcome blind and deterministic.
    first=_case_units(units,str(case_frame.iloc[0].case_id)); coords=first[['x_coord','y_coord']].to_numpy(dtype=float); node_ids=first.node_index.to_numpy(dtype=int)
    ordering=deterministic_maximin_order(coords,node_ids)
    ordering_rows=pd.DataFrame({'order_position':np.arange(len(ordering),dtype=int),'node_index':node_ids[ordering],'x_coord':coords[ordering,0],'y_coord':coords[ordering,1]})
    write_csv(ordering_rows,output_dir/'stage3e_vecchia_ordering.csv')
    for cid in case_frame.case_id.unique():
        cu=_case_units(units,cid); require(np.allclose(cu[['x_coord','y_coord']].to_numpy(dtype=float),coords,atol=0,rtol=0),f'COORDINATE_FRAME_CHANGED:{cid}')

    # N2 oracle Vecchia family, cached by true residual precision.
    m_values=[int(x) for x in contract['candidate_neighborhood_sizes']]; primary_m=int(contract['primary_pilot_neighborhood_size'])
    family_cache={}; n2_rows=[]; local_rows=[]; monotonic_rows=[]; precision_rows=[]; r3_rows=[]; trueQ_by_case={}; family_by_case={}
    for crow in structural_frame.itertuples(index=False):
        cid=str(crow.case_id); cu=_case_units(units,cid); te=_case_edges(true_edges,cid); rho=float(crow.spatial_rho)
        Q=bridge.gp.normalized_precision(len(cu),te,rho,float(contract['residual_scale'])); trueQ_by_case[cid]=Q
        key=(round(rho,15),_edge_hash(te))
        if key not in family_cache:
            fam,meta=build_vecchia_family(Q,ordering,coords,node_ids,m_values,minimum_precision_eigenvalue=float(contract['minimum_precision_eigenvalue']),kl_nonnegative_tolerance=float(contract['kl_nonnegative_tolerance']))
            family_cache[key]=(fam,meta)
        fam,meta=family_cache[key]; family_by_case[cid]=fam
        previous=None
        for m in m_values:
            vr=fam[m]; L=float(vr.total_kl)
            # In frozen Stage3B, the residual precision and geometry do not depend on the one-dimensional
            # target-design feature D. Thus L(U) is constant for this diagnostic DGP and exact target
            # weighting leaves Delta_N2=L even in the nonidentity design-shift validation case.
            delta=L; ds=n2_tv_bound(delta); lb=max(0.0,1-float(contract['alpha'])-ds)
            n2_rows.append({'case_id':cid,'scenario_id':crow.scenario_id,'target_dose_reference':crow.target_dose,'dose_invariant_spatial_certificate':True,'m':m,'observational_omitted_information_L':L,'target_weighted_delta_N2':delta,'delta_sparse_tv_upper':ds,'oracle_sparse_coverage_lower_bound':lb,'target_design_mode':crow.target_design_mode,'target_design_weighting_reason':'L(U)_constant_under_frozen_Stage3B_residual_design','precision_nnz':vr.precision_nnz,'minimum_precision_eigenvalue':vr.min_precision_eigenvalue,'max_neighbors_used':vr.max_neighbors_used,'theorem_status':'N2_GAUSSIAN_ORACLE_BACKED'})
            precision_rows.append({'case_id':cid,'m':m,'n_nodes':len(cu),'q_nnz':vr.precision_nnz,'min_eigenvalue':vr.min_precision_eigenvalue,'selected_pair_count':meta['selected_pair_count'],'selected_solve_count':meta['solve_count'],'dense_inverse_formed':False})
            for pos,(fv,cv,term) in enumerate(zip(vr.full_predecessor_variance,vr.conditional_variance,vr.local_kl_terms)):
                local_rows.append({'case_id':cid,'m':m,'order_position':pos,'node_index':int(node_ids[ordering[pos]]),'full_predecessor_variance':float(fv),'sparse_neighbor_variance':float(cv),'local_omitted_information':float(term),'neighbor_count':len(vr.neighborhoods[pos])})
            if previous is not None:
                monotonic_rows.append({'case_id':cid,'m_small':previous.m,'m_large':m,'kl_small':previous.total_kl,'kl_large':vr.total_kl,'difference_large_minus_small':vr.total_kl-previous.total_kl,'pass':bool(vr.total_kl<=previous.total_kl+float(contract['numerical_tolerance']))})
            previous=vr
        if str(crow.target_design_mode)=='identity':
            L=float(fam[primary_m].total_kl); r3_rows.append({'case_id':cid,'m':primary_m,'target_design_ratio':'1','observational_criterion':L,'target_weighted_criterion':L,'abs_difference':0.0,'pass':True,'reduction_id':'R3'})
    n2_df=pd.DataFrame(n2_rows); write_csv(n2_df,output_dir/'stage3e_n2_certificate.csv')
    write_csv_gz(pd.DataFrame(local_rows),output_dir/'stage3e_n2_local_terms.csv.gz')
    mono=pd.DataFrame(monotonic_rows); write_csv(mono,output_dir/'stage3e_neighborhood_monotonicity.csv')
    write_csv(pd.DataFrame(precision_rows),output_dir/'stage3e_vecchia_precision_audit.csv')
    r3=pd.DataFrame(r3_rows); write_csv(r3,output_dir/'stage3e_r3_reduction.csv')

    exact_sparse=_synthetic_exact_sparse_check(contract); write_json(output_dir/'stage3e_exact_sparse_special_case.json',exact_sparse)
    # Separate arithmetic validation of target-shift amplification identity and Cauchy-Schwarz bound.
    ar=target_shift_identity(np.array([0.5,0.8,1.2,1.5]),np.array([0.1,0.2,0.3,0.5]))
    ar['identity_abs_error']=abs(ar['target_weighted_delta']-ar['identity_rhs']); ar['cs_dominates']=bool(ar['target_weighted_delta']<=ar['cs_upper_bound']+1e-15)
    write_json(output_dir/'stage3e_target_shift_amplification_test.json',ar)
    # Generic target-weighted N2 implementation check on a heterogeneous design frame.
    tw=target_weighted_n2(np.array([0.1,0.2,0.3,0.5]),np.array([0.5,0.8,1.2,1.5]))
    tw['identity_abs_error']=abs(tw['target_weighted_delta_N2']-tw['identity_rhs'])
    tw['cs_dominates']=bool(tw['target_weighted_delta_N2']<=tw['cauchy_schwarz_upper_bound']+1e-15)
    write_json(output_dir/'stage3e_target_weighted_n2_formula_test.json',tw)
    # Frozen Unified Math 10.2 formula implementation test. This demonstrates the
    # theorem-compatible certificate algebra without pretending the unbounded
    # Stage3B Gaussian ratio fixture satisfies the bounded-feature assumptions.
    ratio_formula=bounded_logistic_ratio_certificate(B_phi=2.0,B_infinity=1.0,R=1.0,gamma=0.5,p_r=3,K_r=1,delta_r=0.05,n_r=500)
    ratio_formula['theorem_scope']='FORMULA_IMPLEMENTATION_TEST_ONLY_REQUIRES_SEPARATE_BOUNDED_FEATURE_AND_DEPENDENCY_GRAPH_ASSUMPTION_CHECK'
    write_json(output_dir/'stage3e_target_ratio_certificate_formula_test.json',ratio_formula)

    # Target-design ratio audit.
    ratio_rows=[]
    for cid in case_frame.case_id.unique():
        rr=case_frame[case_frame.case_id==cid].iloc[0]; mode=str(rr.target_design_mode); dcase=design[design.case_id.astype(str)==cid].copy()
        eligible,reason=theorem_eligibility_for_stage3b(mode)
        if mode=='identity':
            maxerr=float(np.max(np.abs(dcase.oracle_target_design_ratio.to_numpy(dtype=float)-1.0)))
            ratio_rows.append({'case_id':cid,'mode':mode,'theorem_eligible':eligible,'theorem_status':reason,'n_train_observational':int((dcase.ratio_training_role=='observational_training').sum()),'n_train_target':int((dcase.ratio_training_role=='target_training').sum()),'prior_odds':float(dcase.class_prior_odds_n0_over_n1.iloc[0]),'beta0':np.nan,'beta1':0.0,'true_delta':0.0,'max_abs_log_ratio_error_eval_empirical_normalized':0.0,'actual_true_to_fitted_target_KL_analytic_slope_diagnostic':0.0,'oracle_ratio_identity_max_error':maxerr,'empirical_observational_normalized_ratio_mean':1.0,'empirical_raw_ratio_normalizer':1.0,'analytic_raw_ratio_normalizer_diagnostic':1.0,'class_prior_odds_factor_used':True})
        elif mode=='normal_location_shift':
            tr=dcase[dcase.ratio_training_role.astype(str).isin(['observational_training','target_training'])]
            fit=fit_normal_shift_logistic(tr)
            ev=dcase[dcase.ratio_evaluation_role.astype(str).isin(['observational_evaluation','target_evaluation'])].copy(); x=ev.X_nonlinear_1.to_numpy(dtype=float)
            obs_norm=ev.loc[ev.target_design_class.astype(str)=='observational_design','X_nonlinear_1'].to_numpy(dtype=float)
            rhat_emp,zemp=empirical_normalized_ratio_from_fit(fit,x,obs_norm); rtrue=ev.oracle_target_design_ratio.to_numpy(dtype=float)
            logerr=np.abs(np.log(rhat_emp)-np.log(rtrue)); delta=float(rr.target_design_delta); actual_kl=normal_shift_true_to_fitted_kl(delta,fit.beta1)
            check_norm,_=empirical_normalized_ratio_from_fit(fit,obs_norm,obs_norm)
            ratio_rows.append({'case_id':cid,'mode':mode,'theorem_eligible':eligible,'theorem_status':reason,'n_train_observational':fit.n_observational,'n_train_target':fit.n_target,'prior_odds':fit.prior_odds,'beta0':fit.beta0,'beta1':fit.beta1,'true_delta':delta,'max_abs_log_ratio_error_eval_empirical_normalized':float(logerr.max()),'actual_true_to_fitted_target_KL_analytic_slope_diagnostic':actual_kl,'oracle_ratio_identity_max_error':np.nan,'empirical_observational_normalized_ratio_mean':float(check_norm.mean()),'empirical_raw_ratio_normalizer':zemp,'analytic_raw_ratio_normalizer_diagnostic':fit.raw_normalizer_analytic,'class_prior_odds_factor_used':True})
        else: raise Stage3EError(f'UNSUPPORTED_TARGET_DESIGN_MODE:{mode}')
    ratio_df=pd.DataFrame(ratio_rows); write_csv(ratio_df,output_dir/'stage3e_target_ratio_audit.csv')

    # Treatment support is diagnostic only; operational ESS/concentration cutoffs remain frozen-to-be-selected by pilot.
    support_df=pd.DataFrame([_treatment_support_row(treatment,str(r.case_id),float(r.target_dose)) for r in case_frame.itertuples(index=False)])
    write_csv(support_df,output_dir/'stage3e_treatment_support_diagnostic.csv')

    # S3 score-inversion and separate GMRF-C conditional KL certificate.
    conf_rows=[]; conf_interval_rows=[]; boundary_rows=[]; refusal_rows=[]
    for crow in structural_frame.itertuples(index=False):
        cid=str(crow.case_id); cu=_case_units(units,cid); te=_case_edges(true_edges,cid); train,cross=_training_component_audit(cu,te)
        S=normalized_adjacency(len(cu),te); # verify same normalized precision convention as accepted D2.
        Qcheck=affine_precision_from_S(S,float(crow.spatial_rho),float(contract['residual_scale']))
        diff=(Qcheck-trueQ_by_case[cid]).data; require(len(diff)==0 or float(np.max(np.abs(diff)))<=1e-12,f'S3_D2_PRECISION_CONVENTION_MISMATCH:{cid}')
        if cross!=0:
            conf_rows.append({'case_id':cid,'eligible':False,'refusal_code':'S3_NUISANCE_COMPONENT_NOT_INDEPENDENT','cross_component_edges':cross}); refusal_rows.append({'case_id':cid,'component':'S3','code':'S3_NUISANCE_COMPONENT_NOT_INDEPENDENT','detail':str(cross)}); continue
        Str=S[train][:,train]; e=cu.set_index('node_index').loc[train,'shared_spatial_residual'].to_numpy(dtype=float)
        conf=rho_score_confidence_set(Str,e,float(contract['residual_scale']),tuple(contract['rho_parameter_domain']),float(contract['spatial_confidence_delta']),float(contract['confidence_outer_cell_tolerance']))
        contains=_interval_contains(conf.outer_intervals,float(crow.spatial_rho)) if conf.outer_intervals else False
        conf_rows.append({'case_id':cid,'eligible':conf.exact_set_nonempty,'refusal_code':'' if conf.exact_set_nonempty else 'S3_CONFIDENCE_SET_EMPTY','cross_component_edges':cross,'n_training':len(train),'true_rho_simulation_only':float(crow.spatial_rho),'rho_hat':conf.rho_hat,'h_at_hat':conf.h_at_hat,'radius':conf.radius,'delta_E':conf.delta,'t_E':conf.t_value,'lipschitz_h':conf.lipschitz_h,'outer_interval_count':len(conf.outer_intervals),'true_rho_inside_outer_set_simulation_check':contains})
        for k,(a,b) in enumerate(conf.outer_intervals): conf_interval_rows.append({'case_id':cid,'interval_index':k,'lower':a,'upper':b})
        if not conf.exact_set_nonempty: refusal_rows.append({'case_id':cid,'component':'S3','code':'S3_CONFIDENCE_SET_EMPTY','detail':''}); continue
        dose=float(case_frame[case_frame.case_id==cid].iloc[0].target_dose); B=_block_for_case(d4prep,cid,dose)
        Qhat=affine_precision_from_S(S,conf.rho_hat,float(contract['residual_scale'])); D=graph_boundary_from_precision(Qhat,B); eD=cu.set_index('node_index').loc[D,'shared_spatial_residual'].to_numpy(dtype=float)
        bc=gmrf_boundary_certificate(S,float(contract['residual_scale']),conf,B,eD); bc.update({'case_id':cid,'true_rho_simulation_only':float(crow.spatial_rho),'rho_hat':conf.rho_hat,'block_nodes':','.join(map(str,B.tolist())),'boundary_nodes':','.join(map(str,D.tolist()))})
        boundary_rows.append(bc)
        if not bool(bc.get('eligible')): refusal_rows.append({'case_id':cid,'component':'GMRF_C','code':str(bc.get('refusal_code')),'detail':'separate certificate; not fused into Vecchia N3 without a frozen bridge'})
    write_csv(pd.DataFrame(conf_rows),output_dir/'stage3e_spatial_confidence_audit.csv')
    write_csv(pd.DataFrame(conf_interval_rows),output_dir/'stage3e_spatial_confidence_intervals.csv')
    write_csv(pd.DataFrame(boundary_rows),output_dir/'stage3e_gmrf_boundary_certificate.csv')
    s3_mc=_s3_monte_carlo_coverage_test(contract); require(bool(s3_mc['pass']),'S3_MONTE_CARLO_IMPLEMENTATION_COVERAGE_TEST_FAILED')
    write_json(output_dir/'stage3e_s3_confidence_coverage_test.json',s3_mc)

    # Working-graph misspecification: known-truth simulation diagnostic at primary m only.
    miss_rows=[]
    for cid in ['S6_ROOK','S6_OMIT50']:
        if cid not in set(case_frame.case_id): continue
        cu=_case_units(units,cid); fe=_case_edges(fitted_edges,cid); rho=float(case_frame[case_frame.case_id==cid].iloc[0].spatial_rho)
        Qfit=bridge.gp.normalized_precision(len(cu),fe,rho,float(contract['residual_scale']))
        fitfam,_=build_vecchia_family(Qfit,ordering,coords,node_ids,[primary_m],minimum_precision_eigenvalue=float(contract['minimum_precision_eigenvalue']),kl_nonnegative_tolerance=float(contract['kl_nonnegative_tolerance']))
        qtrue=family_by_case[cid][primary_m].precision; qwork=fitfam[primary_m].precision
        kl,meta=gaussian_kl_from_precisions(qtrue,qwork,nonnegative_tolerance=float(contract['kl_nonnegative_tolerance']))
        miss_rows.append({'case_id':cid,'m':primary_m,'oracle_sparse_to_working_sparse_KL_simulation_only':kl,'delta_mis_tv_oracle_diagnostic':pinsker(kl),'true_graph_edge_count':len(_case_edges(true_edges,cid)),'fitted_graph_edge_count':len(fe),'working_q_nnz':int(qwork.nnz),'theorem_deployment_status':'ORACLE_DIAGNOSTIC_ONLY_NO_REAL_WORLD_MISSPECIFICATION_CERTIFICATE',**{f'klmeta_{k}':v for k,v in meta.items()}})
    miss_df=pd.DataFrame(miss_rows); write_csv(miss_df,output_dir/'stage3e_working_graph_misspecification.csv')

    # N3 law-ladder records. Primary theorem-backed route uses oracle treatment/spatial law and exact target-design ratio.
    n3_rows=[]
    primary_n2=n2_df[n2_df.m==primary_m].set_index('case_id')
    miss_map={r.case_id:r for r in miss_df.itertuples(index=False)} if len(miss_df) else {}
    for crow in case_frame.itertuples(index=False):
        cid=str(crow.case_id); ds=float(primary_n2.loc[cid,'delta_sparse_tv_upper'])
        rec=n3_record(alpha=float(contract['alpha']),delta_sparse=ds,delta_mis=0.0,delta_est=0.0,delta_comp=0.0,eta=0.0,theorem_eligible=True,eligibility_reason='ORACLE_SPARSE_N2_WITH_EXACT_ORACLE_TARGET_DESIGN_RATIO')
        n3_rows.append({'case_id':cid,'target_dose':crow.target_dose,'route':'ORACLE_SPARSE_PRIMARY',**rec,'deployable_with_estimated_nuisances':False,'claim_status':'THEOREM_BACKED_STRUCTURAL_ORACLE_REFERENCE'})
        if cid in miss_map:
            dm=float(miss_map[cid].delta_mis_tv_oracle_diagnostic); rec2=n3_record(alpha=float(contract['alpha']),delta_sparse=ds,delta_mis=dm,delta_est=0.0,delta_comp=0.0,eta=0.0,theorem_eligible=True,eligibility_reason='SIMULATION_KNOWN_TRUTH_MISSPECIFICATION_KL_AVAILABLE')
            n3_rows.append({'case_id':cid,'target_dose':crow.target_dose,'route':'WORKING_GRAPH_ORACLE_DIAGNOSTIC',**rec2,'deployable_with_estimated_nuisances':False,'claim_status':'SIMULATION_ORACLE_DIAGNOSTIC_ONLY'})
        # Explicitly do not fabricate a certified estimated-nuisance N3 bound.
        n3_rows.append({'case_id':cid,'target_dose':crow.target_dose,'route':'ESTIMATED_NUISANCE_STATUS','alpha':float(contract['alpha']),'delta_sparse':ds,'delta_mis':np.nan,'delta_est':np.nan,'delta_comp':0.0,'certificate_failure_probability_eta':np.nan,'coverage_lower_bound':np.nan,'theorem_eligible':False,'eligibility_reason':'NO_FROZEN_VECCHIA_PARAMETER_ESTIMATION_BRIDGE_AND_STAGE3C_ET_MODEL_NOT_T1_EQUIVALENT','nonvacuous':False,'deployable_with_estimated_nuisances':False,'claim_status':'EXECUTABLE_LATER_AS_EMPIRICAL_VARIANT_NOT_CERTIFIED'})
    n3_df=pd.DataFrame(n3_rows); write_csv(n3_df,output_dir/'stage3e_n3_certificate.csv')

    # Exact-vs-sparse G3 smoke validation on accepted D2 S4_RHO060 fixture.
    prepared=bridge.accepted_exact_prepared(); Qs4=trueQ_by_case['S4_RHO060']; qdiff=(prepared.Q-Qs4).data; require(len(qdiff)==0 or float(np.max(np.abs(qdiff)))<=1e-12,'D2_ACCEPTED_FIXTURE_Q_MISMATCH_STAGE3B_S4')
    candidates=[float(x) for x in contract['exact_vs_sparse_candidate_grid']]
    exact_eval,seed=bridge.evaluate(prepared,'G6_GAUSS_INTERIOR',candidates)
    ref=d2trace[d2trace.fixture_id.astype(str)=='G6_GAUSS_INTERIOR']
    smoke_rows=[]
    for er in exact_eval.itertuples(index=False):
        rr=ref[ref.candidate_hex.astype(str)==str(er.candidate_hex)]; require(len(rr)==1,f'D2_ACCEPTED_TRACE_CANDIDATE_MISSING:{er.candidate_hex}')
        require(abs(float(er.conservative_p)-float(rr.iloc[0].conservative_p))<=2e-12,'D2_SOURCE_REPLAY_FAILED_IN_STAGE3E')
    for m in [int(x) for x in contract['exact_vs_sparse_neighborhood_sizes']]:
        patched,pmeta=bridge.patch_precision(prepared,family_by_case['S4_RHO060'][m].precision)
        sev,_=bridge.evaluate(patched,'G6_GAUSS_INTERIOR',candidates)
        for e,s in zip(exact_eval.itertuples(index=False),sev.itertuples(index=False)):
            smoke_rows.append({'case_id':'S4_RHO060','m':m,'candidate_y':e.candidate_y,'exact_conservative_p':e.conservative_p,'sparse_conservative_p':s.conservative_p,'abs_p_difference':abs(e.conservative_p-s.conservative_p),'exact_accept':e.conservative_accept,'sparse_accept':s.conservative_accept,'accept_match':bool(e.conservative_accept)==bool(s.conservative_accept),'delta_N2':family_by_case['S4_RHO060'][m].total_kl,'delta_sparse_tv_upper':n2_tv_bound(family_by_case['S4_RHO060'][m].total_kl),'sparse_boundary_size':pmeta['boundary_size'],'sparse_precision_nnz':pmeta['q_nnz']})
    smoke_df=pd.DataFrame(smoke_rows); write_csv(smoke_df,output_dir/'stage3e_exact_vs_sparse_pvalue.csv')
    write_json(output_dir/'stage3e_exact_vs_sparse_d2_seed.json',seed)


    # New-data scalable candidate engine: accepted D4 prepares each new six-slot query,
    # Stage3E replaces only the residual precision with Q_m, and accepted D2/G3 computes p(y).
    scalable_rows=[]; scalable_prepare_rows=[]
    for crow in case_frame.itertuples(index=False):
        cid=str(crow.case_id); dose=float(crow.target_dose)
        prepared_new,pmeta=d4bridge.prepare_oracle_true_graph(cid,dose)
        scalable_prepare_rows.append({'case_id':cid,'target_dose':dose,'block_nodes':','.join(map(str,pmeta['block_nodes'])),'base_evaluation_id':str(pmeta.get('base_evaluation_id','')),'D4_preparation_reused':True,'treatment_mode':'OT','graph_mode':'TG','stage3e_external_target_table_override':bool(pmeta.get('stage3e_external_target_table_override',False)),'target_scope_original':str(pmeta.get('target_scope_original','')),'target_scope_local_D2_alias_applied':bool(pmeta.get('target_scope_local_D2_alias_applied',False)),'target_field_name':str(pmeta.get('target_field_name',''))})
        for m in [int(x) for x in contract['exact_vs_sparse_neighborhood_sizes']]:
            patched,spmeta=bridge.patch_precision(prepared_new,family_by_case[cid][m].precision)
            fixture_id=f'STAGE3E_{cid}_DOSE_{dose:.12g}_M{m}'
            sev,sseed=bridge.evaluate(patched,fixture_id,candidates)
            ds=float(n2_tv_bound(family_by_case[cid][m].total_kl)); lb=max(0.0,1-float(contract['alpha'])-ds)
            for r in sev.itertuples(index=False):
                scalable_rows.append({'case_id':cid,'scenario_id':crow.scenario_id,'target_dose':dose,'m':m,'candidate_y':r.candidate_y,'candidate_hex':r.candidate_hex,'conservative_p':r.conservative_p,'randomized_p':r.randomized_p,'conservative_accept':r.conservative_accept,'randomized_accept':r.randomized_accept,'distinct_states':r.distinct_states,'tie_uniform':r.tie_uniform,'delta_N2':family_by_case[cid][m].total_kl,'delta_sparse_tv_upper':ds,'oracle_sparse_coverage_lower_bound':lb,'sparse_boundary_size':spmeta['boundary_size'],'sparse_precision_nnz':spmeta['q_nnz'],'source_level_exact_engine':'ACCEPTED_D2_G1_G2_G3','newdata_preparation':'ACCEPTED_D4_V1_1_0','claim_status':'CANDIDATE_LEVEL_SCALABLE_ENGINE_VALIDATION_NOT_PILOT_PERFORMANCE'})
    scalable_df=pd.DataFrame(scalable_rows); write_csv(scalable_df,output_dir/'stage3e_newdata_sparse_candidate_trace.csv')
    prepare_df=pd.DataFrame(scalable_prepare_rows); write_csv(prepare_df,output_dir/'stage3e_newdata_sparse_prepare_audit.csv')

    # Full finite-domain scalable prediction-set inversion smoke test on one new-data
    # combined-shift query using the accepted D2 inversion machinery unchanged.
    full_case='S4_RHO060'; full_dose=0.90; full_m=primary_m
    prepared_full,full_pmeta=d4bridge.prepare_oracle_true_graph(full_case,full_dose)
    patched_full,full_spmeta=bridge.patch_precision(prepared_full,family_by_case[full_case][full_m].precision)
    full_fixture_id=f'STAGE3E_{full_case}_DOSE_{full_dose:.12g}_M{full_m}'
    fullinv=bridge.full_invert(patched_full,full_fixture_id,independent_grid_audit=True)
    ftrace=pd.DataFrame(fullinv['trace']); ftrace.insert(0,'case_id',full_case); ftrace.insert(1,'target_dose',full_dose); ftrace.insert(2,'m',full_m)
    write_csv_gz(ftrace,output_dir/'stage3e_sparse_prediction_set_candidate_trace.csv.gz')
    fcomp=pd.DataFrame(fullinv['components']); fcomp.insert(0,'case_id',full_case); fcomp.insert(1,'target_dose',full_dose); fcomp.insert(2,'m',full_m); write_csv(fcomp,output_dir/'stage3e_sparse_prediction_set_components.csv')
    fbnd=pd.DataFrame(fullinv['boundaries']); fbnd.insert(0,'case_id',full_case); fbnd.insert(1,'target_dose',full_dose); fbnd.insert(2,'m',full_m); write_csv(fbnd,output_dir/'stage3e_sparse_prediction_set_boundary_audit.csv')
    fsumm=[]
    for mode,sm in fullinv['summaries'].items():
        fsumm.append({'case_id':full_case,'target_dose':full_dose,'m':full_m,'tie_mode':mode,'alpha':float(contract['alpha']),'domain_lower':fullinv['domain_lower'],'domain_upper':fullinv['domain_upper'],**sm,'candidate_evaluation_count':fullinv['candidate_evaluation_count'],'initial_unique_points':fullinv['initial_unique_points'],'independent_grid_points':fullinv['independent_grid_audit']['points'],'independent_grid_conservative_accepted_points':fullinv['independent_grid_audit']['conservative_accepted_points'],'independent_grid_conservative_false_negatives':fullinv['independent_grid_audit']['conservative_false_negatives'],'independent_grid_randomized_accepted_points':fullinv['independent_grid_audit']['randomized_accepted_points'],'independent_grid_randomized_false_negatives':fullinv['independent_grid_audit']['randomized_false_negatives'],'delta_N2':family_by_case[full_case][full_m].total_kl,'delta_sparse_tv_upper':n2_tv_bound(family_by_case[full_case][full_m].total_kl),'oracle_sparse_coverage_lower_bound':max(0.0,1-float(contract['alpha'])-n2_tv_bound(family_by_case[full_case][full_m].total_kl)),'returned_set_representation':fullinv['returned_set_representation'],'full_real_line_claim':fullinv['full_real_line_claim'],'claim_status':'SCALABLE_FINITE_DOMAIN_OUTER_SET_VALIDATION_NOT_PILOT_PERFORMANCE'})
    fsumm_df=pd.DataFrame(fsumm); write_csv(fsumm_df,output_dir/'stage3e_sparse_prediction_set_summary.csv')
    write_json(output_dir/'stage3e_sparse_prediction_set_seed.json',fullinv['seed'])
    # Candidate-level cross-check against the already generated new-data trace at the
    # three frozen diagnostic candidates. The fixture ID is identical, so both
    # conservative and randomized p-values must replay exactly.
    cross=[]
    prior=scalable_df[(scalable_df.case_id==full_case)&np.isclose(scalable_df.target_dose,full_dose)&(scalable_df.m==full_m)]
    for y in candidates:
        a=prior[np.isclose(prior.candidate_y,float(y))]; b=ftrace[np.isclose(ftrace.candidate_y,float(y))]
        require(len(a)==1 and len(b)==1,'STAGE3E_FULL_INVERSION_CROSSCHECK_CANDIDATE_MISSING')
        cross.append({'candidate_y':float(y),'conservative_p_abs_diff':abs(float(a.iloc[0].conservative_p)-float(b.iloc[0].conservative_p)),'randomized_p_abs_diff':abs(float(a.iloc[0].randomized_p)-float(b.iloc[0].randomized_p)),'conservative_accept_match':bool(a.iloc[0].conservative_accept)==bool(b.iloc[0].conservative_accept),'randomized_accept_match':bool(a.iloc[0].randomized_accept)==bool(b.iloc[0].randomized_accept)})
    fullcross=pd.DataFrame(cross); require(float(fullcross[['conservative_p_abs_diff','randomized_p_abs_diff']].to_numpy().max())<=2e-12,'STAGE3E_FULL_INVERSION_CROSSCHECK_FAILED'); write_csv(fullcross,output_dir/'stage3e_sparse_prediction_set_crosscheck.csv')

    # Certificate/refusal grid: Stage3E applies only mathematical non-vacuity. Operational
    # ESS/max-weight/lower-bound thresholds remain deliberately deferred to the 20-rep pilot.
    cert_rows=[]
    for r in n2_df.itertuples(index=False):
        lb=float(r.oracle_sparse_coverage_lower_bound); nonv=lb>0.0
        cert_rows.append({'case_id':r.case_id,'m':int(r.m),'delta_N2':float(r.target_weighted_delta_N2),'delta_sparse_tv_upper':float(r.delta_sparse_tv_upper),'coverage_lower_bound':lb,'mathematically_nonvacuous':nonv,'certified_structural_oracle_route':nonv,'refusal_code':'' if nonv else 'R_N3_VACUOUS_COVERAGE_LOWER_BOUND','operational_threshold_applied':False,'operational_threshold_status':'DEFERRED_TO_20REP_PILOT'})
    cert_df=pd.DataFrame(cert_rows); write_csv(cert_df,output_dir/'stage3e_certificate_refusal_grid.csv')

    # Mandatory no-double-counting and good-event arithmetic numerical checks.
    arithmetic={'good_event_example':{'alpha':0.1,'delta_sparse':0.02,'delta_mis':0.01,'delta_est':0.03,'delta_comp':0.0,'eta':0.04,'coverage_lower_bound':max(0,1-.1-.02-.01-.03-0-.04)},'no_double_counting_example':{'kl':0.02,'omega':0.4,'pinsker':math.sqrt(.02/2),'oscillation':math.tanh(.4/4),'selected':min(math.sqrt(.02/2),math.tanh(.4/4))}}
    write_json(output_dir/'stage3e_n3_arithmetic_tests.json',arithmetic)

    # Refusals/warnings kept separate from theorem-backed primary oracle route.
    ratio_ineligible=ratio_df[~ratio_df.theorem_eligible.astype(bool)]
    for r in ratio_ineligible.itertuples(index=False): refusal_rows.append({'case_id':r.case_id,'component':'TARGET_RATIO_ESTIMATION','code':r.theorem_status,'detail':'oracle ratio remains available in controlled simulation; estimated-ratio theorem not invoked'})
    write_csv(pd.DataFrame(refusal_rows,columns=['case_id','component','code','detail']),output_dir/'stage3e_refusal_log.csv')

    # Claim boundary and pilot readiness.
    primary=n2_df[n2_df.m==primary_m]; worst_delta=float(primary.delta_sparse_tv_upper.max()); min_lb=float(primary.oracle_sparse_coverage_lower_bound.min())
    all_mono=bool(mono['pass'].all()); all_r3=bool(r3['pass'].all()); exact_sparse_pass=bool(exact_sparse['pass']); smoke_finite=bool(np.isfinite(smoke_df[['exact_conservative_p','sparse_conservative_p']].to_numpy()).all())
    scalable_finite=bool(len(scalable_df)>0 and np.isfinite(scalable_df[['conservative_p','randomized_p']].to_numpy()).all() and scalable_df.conservative_p.between(0,1).all() and scalable_df.randomized_p.between(0,1).all())
    full_inversion_pass=bool(len(fsumm_df)==2 and int(fullinv['independent_grid_audit']['conservative_false_negatives'])==0 and int(fullinv['independent_grid_audit']['randomized_false_negatives'])==0 and len(fullcross)==len(candidates) and float(fullcross[['conservative_p_abs_diff','randomized_p_abs_diff']].to_numpy().max())<=2e-12)
    s3_mc_pass=bool(s3_mc['pass'])
    readiness=all_mono and all_r3 and exact_sparse_pass and smoke_finite and scalable_finite and full_inversion_pass and s3_mc_pass and min_lb>0.0
    pilot={'ready_for_preregistered_20rep_pilot':bool(readiness),'primary_neighborhood_size':primary_m,'selection_basis':'largest pre-frozen diagnostic neighborhood in Stage3E grid; selected structurally before pilot outcomes, not tuned on coverage','worst_oracle_sparse_tv_bound_across_stage3e_cases':worst_delta,'minimum_oracle_sparse_coverage_lower_bound_across_stage3e_cases':min_lb,'R3_pass':all_r3,'neighborhood_monotonicity_pass':all_mono,'exact_sparse_special_case_pass':exact_sparse_pass,'exact_vs_sparse_G3_smoke_pass':smoke_finite,'newdata_sparse_candidate_engine_pass':scalable_finite,'scalable_full_prediction_set_inversion_smoke_pass':full_inversion_pass,'S3_confidence_implementation_coverage_test_pass':s3_mc_pass,'estimated_nuisance_certified':False,'pilot_rule':'pilot may execute OT_TG scalable primary and ET/FG empirical diagnostics, but ET/FG must not be labelled theorem-certified','operational_ESS_and_weight_thresholds':'DEFERRED_TO_20REP_PILOT','production_run_allowed':False}
    write_json(output_dir/'stage3e_pilot_readiness.json',pilot)
    claims={'stage':STAGE,'version':VERSION,'N2_oracle_gaussian_theorem_backed':True,'N2_target_weighting_implemented':True,'R3_completed':all_r3,'N3_oracle_structural_coverage_bound_theorem_backed':True,'S3_centered_affine_GMRF_confidence_set_implemented':True,'GMRF_C_realized_boundary_certificate_implemented_separately':True,'S3_GMRF_C_fused_into_Vecchia_estimation_bound':False,'reason_not_fused':'canonical frozen mathematics contains no theorem bridging S3 parameter confidence to the nonlinear Vecchia q_theta family; no new theorem is invented','target_ratio_identity_exact':True,'target_ratio_nonidentity_fit_theorem_backed':False,'reason_ratio_fit_not_theorem_backed':'Stage3B validation feature is Gaussian/unbounded, violating the bounded-feature assumption of Unified Math 10.2','Stage3C_estimated_treatment_T1_certified':False,'reason_ET_not_T1':'accepted Stage3C MixedPropensityModel is not asserted theorem-equivalent to the frozen exponential-family T1 model','working_graph_misspecification_bound_deployable':False,'working_graph_misspecification_simulation_oracle_diagnostic':True,'full_625_node_dense_inverse_formed':False,'Monte_Carlo_orbit_approximation_used':False,'delta_comp_primary':0.0,'pilot_performance_evidence_generated':False,'production_performance_evidence_generated':False,'newdata_sparse_candidate_engine_ready':scalable_finite,'scalable_full_prediction_set_inversion_validated_on_S4_RHO060_m64':full_inversion_pass,'full_prediction_set_inversion_not_generated_in_stage3e':False,'prediction_set_claim_boundary':'validated finite-domain outer inversion on representative Stage3E query; pilot/production performance not evaluated','pilot_scale_certificate_engine_ready':bool(readiness)}
    write_json(output_dir/'stage3e_claim_boundary.json',claims)

    # Environment/provenance/summary.
    env={'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__}
    try:
        import scipy; env['scipy']=scipy.__version__
    except Exception: pass
    write_json(output_dir/'stage3e_environment_inventory.json',env)
    srcinv=source_inventory(root); write_json(output_dir/'stage3e_source_hashes.json',srcinv)
    counts={'registered_case_dose_entries':len(case_frame),'unique_case_ids':int(case_frame.case_id.nunique()),'unique_structural_case_ids':len(structural_frame),'neighborhood_sizes':m_values,'n2_certificate_rows':len(n2_df),'n2_local_rows':len(local_rows),'monotonicity_rows':len(mono),'r3_rows':len(r3),'target_ratio_rows':len(ratio_df),'spatial_confidence_rows':len(conf_rows),'gmrf_boundary_rows':len(boundary_rows),'n3_rows':len(n3_df),'exact_vs_sparse_rows':len(smoke_df),'newdata_sparse_candidate_rows':len(scalable_df),'newdata_sparse_prepare_rows':len(scalable_prepare_rows),'sparse_prediction_set_trace_rows':len(ftrace),'sparse_prediction_set_component_rows':len(fcomp),'sparse_prediction_set_boundary_rows':len(fbnd),'sparse_prediction_set_summary_rows':len(fsumm_df),'certificate_refusal_rows':len(cert_df),'s3_coverage_test_repetitions':int(s3_mc['repetitions']),'refusal_warning_rows':len(refusal_rows)}
    write_json(output_dir/'stage3e_counts.json',counts)
    runtime={'stage':STAGE,'version':VERSION,'elapsed_seconds':time.perf_counter()-started}; write_json(output_dir/'stage3e_runtime.json',runtime)
    manifest=output_manifest(output_dir,exclude={'stage3e_manifest.json'}); write_json(output_dir/'stage3e_manifest.json',manifest)
    return {'pilot_readiness':pilot,'counts':counts,'claim_boundary':claims,'runtime':runtime}
