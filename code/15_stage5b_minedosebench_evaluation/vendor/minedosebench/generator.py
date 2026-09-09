from __future__ import annotations
import numpy as np, pandas as pd
from .math import robust_z,treatment_parameters,draw_mixed,mixed_density_at,outcome_baseline,conditional_mean,measurement_error,empirical_target_design_ratio,empirical_target_design_ratio_stratified,sample_localized,stable_seed
from .gmrf import gmrf_draw
from .io import require

# Each stream has a distinct seed for each (common_random_group, replication, stream).
# Cases in the same common-random group intentionally reuse that stream seed.
STREAMS=['covariate','treatment','residual','measurement','target_draw','exact_audit','graph','temporal']

def build_seed_registry(cases:pd.DataFrame, stage3a_seeds:pd.DataFrame, reps:int=20):
    occupied=set()
    for c in stage3a_seeds.columns:
        if c.endswith('_seed'):
            occupied.update(pd.to_numeric(stage3a_seeds[c],errors='coerce').dropna().astype(int).tolist())
    rows=[]; used=set(occupied); assigned={}
    for case in cases.itertuples(index=False):
        for rep in range(1,reps+1):
            rec={'case_id':case.case_id,'scenario_id':case.scenario_id,'common_random_group':case.common_random_group,'replication':rep}
            for stream in STREAMS:
                key=(str(case.common_random_group),int(rep),stream)
                if key not in assigned:
                    salt=0
                    while True:
                        v=stable_seed('GeoDoseCP-MineDoseBench-v1.1',*key,salt)
                        if v not in used: break
                        salt+=1
                    used.add(v); assigned[key]=v
                rec[stream+'_seed']=assigned[key]
            rows.append(rec)
    return pd.DataFrame(rows)

def _coords_axis(df):
    x=df.spatial_axis_score.to_numpy(float); z,_,_=robust_z(x); return z

def _truth_features(df):
    pv23=df.pv_median_2023.to_numpy(float)/100.; pv24=df.pv_median_2024.to_numpy(float)/100.; trend=pv24-pv23
    rain=(df.rainfall_2023_mm.to_numpy(float)+df.rainfall_2024_mm.to_numpy(float))/2
    soc=df.soil_organic_carbon_pct_0_15cm.to_numpy(float); slope=np.log1p(np.maximum(df.terrain_slope_deg.to_numpy(float),0)); awc=df.soil_available_water_capacity_mm_0_100cm.to_numpy(float)
    xpv,_,_=robust_z(pv24);xrain,_,_=robust_z(rain);xsoc,_,_=robust_z(soc);xslope,_,_=robust_z(slope);xawc,_,_=robust_z(awc);xcoord=_coords_axis(df)
    rough,_m,_s=robust_z(df.terrain_roughness.to_numpy(float)); substrate=0.70*rough-0.30*xsoc
    return dict(pv23=pv23,pv24=pv24,pv_trend=trend,xpv=xpv,xrain=xrain,xsoc=xsoc,xslope=xslope,xawc=xawc,xcoord=xcoord,substrate=substrate)

def _edge_indices(df,edges):
    idx={str(b):i for i,b in enumerate(df.block_id.astype(str))}; e=edges[edges.source_block_id.astype(str).isin(idx)&edges.target_block_id.astype(str).isin(idx)]
    return e.source_block_id.astype(str).map(idx).to_numpy(int), e.target_block_id.astype(str).map(idx).to_numpy(int)

def working_edges(case, substrate, edges, seed):
    """Prospectively construct the method working graph without touching outcomes."""
    if case['fitted_graph']=='true_queen': return edges.copy().reset_index(drop=True)
    if case['fitted_graph']=='rook':
        require({'grid_row','grid_col'}.issubset(substrate.columns) or {'grid_row180','grid_col180'}.issubset(substrate.columns),'Rook graph requires frozen grid indices')
        rr='grid_row' if 'grid_row' in substrate.columns else 'grid_row180'; cc='grid_col' if 'grid_col' in substrate.columns else 'grid_col180'
        pos=substrate[['block_id',rr,cc]].copy().rename(columns={rr:'grid_row',cc:'grid_col'}); pos['block_id']=pos.block_id.astype(str)
        q=edges.copy();q['source_block_id']=q.source_block_id.astype(str);q['target_block_id']=q.target_block_id.astype(str)
        q=q.merge(pos.rename(columns={'block_id':'source_block_id','grid_row':'sr','grid_col':'sc'}),on='source_block_id',how='left')
        q=q.merge(pos.rename(columns={'block_id':'target_block_id','grid_row':'tr','grid_col':'tc'}),on='target_block_id',how='left')
        keep=(q.sr-q.tr).abs()+(q.sc-q.tc).abs()==1
        return q.loc[keep,edges.columns].reset_index(drop=True)
    if case['fitted_graph']=='queen_omit50':
        keep=[]
        for r in edges.itertuples(index=False):
            a=min(str(r.source_block_id),str(r.target_block_id));b=max(str(r.source_block_id),str(r.target_block_id))
            keep.append(stable_seed('MDB-OMIT50-v1.1',seed,a,b)%2==0)
        out=edges[np.asarray(keep,bool)].copy().reset_index(drop=True);require(len(out)>0,'Omit50 working graph is empty');return out
    raise ValueError(case['fitted_graph'])

def _hidden_spatial_confounder(case, n, s, t, seed):
    if case['confounding']!='hidden_C2_negative_control':
        return np.zeros(n,float)
    innov=np.random.default_rng(seed).standard_normal(n)
    u,_,err=gmrf_draw(n,s,t,.60,1.0,innov,64)
    require(err<=1e-12,'Hidden spatial confounder GMRF approximation failed')
    u,_,_=robust_z(u)
    return u

def generate_case_rep(case:dict,rep:int,substrate:pd.DataFrame,edges:pd.DataFrame,seeds:dict,residual_scale:float=.04):
    df=substrate[(substrate.benchmark_role!='baseline_ineligible') & (substrate.primary_component==True)].copy().reset_index(drop=True)
    f=_truth_features(df); n=len(df); s,t=_edge_indices(df,edges)
    # Hidden C2 negative-control confounder is itself spatial and is never released as an allowed predictor.
    hidden=_hidden_spatial_confounder(case,n,s,t,int(seeds['covariate_seed']))
    probs,aa,bb,gmean=treatment_parameters(case,f['xpv'],f['xrain'],f['xsoc'],f['xslope'],f['xcoord'],hidden)
    rng_t=np.random.default_rng(seeds['treatment_seed']); A,cat,latent,u=draw_mixed(rng_t,probs,aa,bb); g=mixed_density_at(A,probs,aa,bb)
    # One coherent residual per unit; real graph and geometry are fixed, only innovations are injected.
    innov=np.random.default_rng(seeds['residual_seed']).standard_normal(n)
    e,min_eig,cheb_err=gmrf_draw(n,s,t,float(case['spatial_rho']),residual_scale,innov,64)
    if case['residual_law']=='transformed_gmrf_power_1_5': e=np.sign(e)*np.abs(e)**1.5
    base=outcome_baseline(case,f['pv_trend'],f['xpv'],f['xrain'],f['xsoc'],f['xslope'],f['xawc'],f['xcoord'],hidden)
    mu=conditional_mean(case,base,f['xpv'],A); ytrue=mu+e
    zme=np.random.default_rng(seeds['measurement_seed']).standard_normal(n); me=measurement_error(case,A,df.ue_median_2024.to_numpy(float),f['substrate'],zme); yobs=ytrue+me
    ratio=np.ones(n)
    if case.get('target_design_mode')=='finite_exponential_tilt': ratio=empirical_target_design_ratio_stratified(f['xpv'],df.MineID.astype(str).to_numpy(),float(case.get('target_design_delta',.75)))
    out=pd.DataFrame({
      'case_id':case['case_id'],'scenario_id':case['scenario_id'],'replication':rep,'block_id':df.block_id.astype(str),'MineID':df.MineID.astype(str),'MineN':df.MineN.astype(str),'benchmark_role':df.benchmark_role.astype(str),
      'A':A,'treatment_category':cat,'g_at_A':g,'g_pi0':probs[:,0],'g_pi1':probs[:,1],'g_pii':probs[:,2],'g_alpha':aa,'g_beta':bb,'g_interior_mean':gmean,
      'outcome_baseline_truth':base,'residual_truth':e,'Y_true_at_A':ytrue,'measurement_error_at_A':me,'Y_observed_at_A':yobs,'target_design_ratio_truth':ratio,
      'x_pv_truth':f['xpv'],'x_rain_truth':f['xrain'],'x_soc_truth':f['xsoc'],'x_slope_truth':f['xslope'],'x_awc_truth':f['xawc'],'x_coord_truth':f['xcoord'],'substrate_proxy_truth':f['substrate'],
      'ue_proxy':df.ue_median_2024.to_numpy(float),'measurement_base_innovation_truth':zme,'residual_base_innovation_truth':innov,'hidden_u_truth':hidden,'precision_min_eigenvalue_bound':min_eig,'chebyshev_scalar_error_bound':cheb_err,
    })
    return out,dict(base=base,e=e,xpv=f['xpv'],measurement_z=zme,substrate=f['substrate'],hidden=hidden)

def target_design_weights(case:dict, substrate:pd.DataFrame):
    df=substrate[(substrate.benchmark_role!='baseline_ineligible') & (substrate.primary_component==True)].copy().reset_index(drop=True)
    f=_truth_features(df); w=np.ones(len(df),float)
    if case.get('target_design_mode')=='finite_exponential_tilt':
        w=empirical_target_design_ratio_stratified(f['xpv'],df.MineID.astype(str).to_numpy(),float(case.get('target_design_delta',.75)))
    return pd.DataFrame({'block_id':df.block_id.astype(str),'MineID':df.MineID.astype(str),'target_design_weight':w})

def select_target_registry(case:dict, rep:int, substrate:pd.DataFrame, exact_map:pd.DataFrame, seeds:dict, n_targets:int=5):
    """Freeze target identities using pretreatment information only."""
    # Build the target frame entirely from frozen pretreatment/support information.
    frame=substrate.loc[(substrate.benchmark_role=='test_target') & (substrate.primary_component==True), ['block_id','MineID']].copy()
    frame['block_id']=frame.block_id.astype(str); frame['MineID']=frame.MineID.astype(str)
    if case.get('target_design_mode')=='finite_exponential_tilt':
        weights=target_design_weights(case,substrate)
        frame=frame.merge(weights,on=['block_id','MineID'],how='left',validate='1:1')
    else:
        frame['target_design_weight']=1.0
    rng=np.random.default_rng(int(seeds['target_draw_seed'])); rows=[]
    for mid,g in frame.groupby('MineID',sort=True):
        g=g.sort_values('block_id',kind='mergesort').reset_index(drop=True); require(len(g)>=n_targets,f'Need >= {n_targets} test targets for mine {mid}')
        p=g.target_design_weight.to_numpy(float); require(np.isfinite(p).all() and (p>0).all(),'Invalid target-design weights')
        p=p/p.sum(); chosen=rng.choice(len(g),size=n_targets,replace=False,p=p)
        for rank,j in enumerate(chosen,1):
            r=g.iloc[int(j)]; rows.append({'case_id':case['case_id'],'replication':rep,'MineID':str(mid),'target_rank':rank,'target_block_id':str(r.block_id),'target_selection_mode':'target_design_weighted' if case.get('target_design_mode')=='finite_exponential_tilt' else 'uniform_test_frame','target_design_sampling_weight':float(r.target_design_weight)})
    # One exact audit target per mine, sampled independently from exact-size6-eligible test targets.
    erows=[]; erng=np.random.default_rng(int(seeds['exact_audit_seed']))
    exact_ids=set(exact_map.target_block_id.astype(str)) if len(exact_map) else set()
    for mid,g in frame.groupby('MineID',sort=True):
        ids=sorted(set(g.block_id.astype(str)) & exact_ids)
        if not ids:
            erows.append({'case_id':case['case_id'],'replication':rep,'MineID':str(mid),'exact_audit_available':False,'exact_audit_target_block_id':''})
        else:
            erows.append({'case_id':case['case_id'],'replication':rep,'MineID':str(mid),'exact_audit_available':True,'exact_audit_target_block_id':ids[int(erng.integers(0,len(ids)))]})
    return pd.DataFrame(rows),pd.DataFrame(erows)

def target_truth_from_registry(case,units,truth_arrays,target_registry):
    rows=[]
    byid={str(b):i for i,b in enumerate(units.block_id.astype(str))}
    # Separate localized A* sub-draws by target identity so ordering cannot change the draw.
    base_seed=int(units.iloc[0].replication)
    for tr in target_registry.itertuples(index=False):
        target=str(tr.target_block_id); require(target in byid,f'Target {target} absent from generated units'); i=byid[target]
        if case['primary_target_mode']=='observational': astar=float(units.loc[i,'A']); dose=np.nan; h=np.nan
        else:
            dose=float(case['primary_target_dose']); h=case.get('primary_bandwidth');
            seed=stable_seed('MDB-ASTAR-v1.1',case['case_id'],int(tr.replication),str(tr.MineID),int(tr.target_rank),target)
            rng=np.random.default_rng(seed)
            astar=sample_localized(rng,dose,None if h is None or (isinstance(h,float) and np.isnan(h)) else float(h),case['endpoint_structure']!='interior_only')
        mu=float(conditional_mean(case,np.array([truth_arrays['base'][i]]),np.array([truth_arrays['xpv'][i]]),np.array([astar]))[0]); ytrue=mu+float(truth_arrays['e'][i])
        me=float(measurement_error(case,np.array([astar]),np.array([units.loc[i,'ue_proxy']]),np.array([truth_arrays['substrate'][i]]),np.array([truth_arrays['measurement_z'][i]]))[0]); yobs=ytrue+me
        rows.append(dict(case_id=case['case_id'],replication=int(tr.replication),MineID=str(tr.MineID),target_rank=int(tr.target_rank),target_block_id=target,target_selection_mode=tr.target_selection_mode,target_design_sampling_weight=float(tr.target_design_sampling_weight),target_mode=case['primary_target_mode'],requested_dose=dose,bandwidth=h,A_star=astar,Y_true_A_star=ytrue,Y_observed_A_star=yobs,post_generation_truth_only=True))
    return pd.DataFrame(rows)
