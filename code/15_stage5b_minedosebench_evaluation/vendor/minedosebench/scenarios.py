from __future__ import annotations
import pandas as pd

def build_scenario_registry()->pd.DataFrame:
    rows=[]
    def add(cid,sid,name,ctype,crg,rho,shift,overlap,fg,me,ml,conf,outcome,endpoint,resid,target_mode,dose,h,**kw):
        d=dict(case_id='MDB_'+cid,scenario_id=sid,case_name=name,case_type=ctype,common_random_group='MDB_'+crg,
               spatial_rho=rho,target_shift=shift,overlap=overlap,fitted_graph=fg,measurement_error=me,measurement_lambda=ml,
               confounding=conf,outcome_form=outcome,endpoint_structure=endpoint,residual_law=resid,primary_target_mode=target_mode,
               primary_target_dose=dose,primary_bandwidth=h,support_scale='90m_primary',measurement_substrate_lambda=0.0,
               target_design_mode='identity',target_design_delta=0.0,temporal_dependence=False,calibration_subsample_fraction=1.0)
        d.update(kw);rows.append(d)
    add('S1_BASE','S1','weak shift and negligible dependence','scenario','S1_BASE',0.,'weak','good','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','iid_continuous','localized',.50,.10)
    add('S2_SPATIAL','S2','spatial dependence only','scenario','S2_SPATIAL',.6,'none','good','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','observational',None,None)
    add('S3_SHIFT','S3','treatment shift only','scenario','S3_SHIFT',0.,'strong','moderate','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','iid_continuous','localized',.90,.10)
    for rho in (0.,.2,.4,.6,.8): add(f'S4_RHO{int(rho*100):03d}','S4',f'combined shift rho={rho:.1f}','scenario_factor','S4_COMMON',rho,'strong','moderate','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.90,.10)
    add('S4_RHO060_NONGAUSSIAN','S4','non-Gaussian graph-law validation','residual_law_validation','S4_COMMON',.6,'strong','moderate','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','transformed_gmrf_power_1_5','localized',.90,.10)
    add('S4_RHO060_DESIGN_SHIFT','S4','finite-frame nonidentity target-design shift','target_design_transport_validation','S4_COMMON',.6,'strong','moderate','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.90,.10,target_design_mode='finite_exponential_tilt',target_design_delta=.75)
    add('S5_POOR_OVERLAP','S5','poor overlap and endpoint target','scenario','S5_POOR_OVERLAP',.4,'tail','poor','true_queen','none',0.,'observed_strong','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',1.,None)
    add('S6_ROOK','S6','rook working graph misspecification','graph_misspecification','S6_COMMON',.6,'moderate','moderate','rook','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10)
    add('S6_OMIT50','S6','50% deterministic edge omission','graph_misspecification','S6_COMMON',.6,'moderate','moderate','queen_omit50','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10)
    add('S7_EXCHANGEABLE','S7','benign exchangeable observational target','scenario','S7_EXCHANGEABLE',0.,'none','good','true_queen','none',0.,'observed_none','linear','mixed_atoms_0_1','iid_continuous','observational',None,None)
    add('S8_SEVERE_TAIL','S8','severe tail concentration','scenario_failure','S8_SEVERE_TAIL',.6,'outside_near_boundary','severe','true_queen','none',0.,'observed_strong','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.98,.10)
    add('S8_SMALL_CAL','S8','too little independent calibration information','scenario_failure','S8_SMALL_CAL',.6,'moderate','moderate','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10,calibration_subsample_fraction=.08)
    add('S9_RANDOM','S9','heteroscedastic random EO error','measurement_error_factor','S9_COMMON',.4,'moderate','moderate','true_queen','random',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10)
    add('S9_TREATMENT05','S9','random + treatment-correlated EO error','measurement_error_factor','S9_COMMON',.4,'moderate','moderate','true_queen','treatment_correlated',.020,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10)
    add('S9_TREATMENT10','S9','stronger treatment-correlated EO error','measurement_error_factor','S9_COMMON',.4,'moderate','moderate','true_queen','treatment_correlated',.040,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10)
    add('S9_SUBSTRATE','S9','random + substrate-specific EO bias','measurement_error_factor','S9_COMMON',.4,'moderate','moderate','true_queen','substrate_bias',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10,measurement_substrate_lambda=.020)
    add('S9_COMBINED05','S9','combined treatment/substrate EO error','measurement_error_factor','S9_COMMON',.4,'moderate','moderate','true_queen','combined',.020,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10,measurement_substrate_lambda=.020)
    add('S10_90M','S10','primary 90 m support','support_factor','S10_COMMON',.4,'moderate','moderate','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10,support_scale='90m_primary')
    add('S10_180M','S10','anchored 180 m quality-consistent rerun','support_factor','S10_COMMON',.4,'moderate','moderate','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10,support_scale='180m_anchored_rerun')
    add('ST1_MIXED_ATOMS','ST1','strong mixed endpoint atoms','additional_stress','ST1_COMMON',.4,'moderate','moderate','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_strong','gaussian_gmrf','localized',.75,.10)
    add('ST1_INTERIOR_ONLY','ST1','interior-only treatment negative endpoint reference','additional_stress','ST1_COMMON',.4,'moderate','moderate','true_queen','none',0.,'observed_moderate','nonlinear','interior_only','gaussian_gmrf','localized',.75,.10)
    add('ST2_TEMPORAL','ST2','repeated mine-year temporal dependence','additional_stress','ST2_TEMPORAL',.4,'moderate','moderate','true_queen','none',0.,'observed_moderate','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10,temporal_dependence=True)
    add('ST3_HIDDEN_C2','ST3','hidden spatial confounding C2 negative control','additional_stress','ST3_HIDDEN_C2',.4,'moderate','moderate','true_queen','none',0.,'hidden_C2_negative_control','nonlinear','mixed_atoms_0_1','gaussian_gmrf','localized',.75,.10)
    f=pd.DataFrame(rows)
    assert len(f)==27 and f.case_id.is_unique
    return f
