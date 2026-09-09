from __future__ import annotations
import math
import numpy as np,pandas as pd
from .common import stable_seed

def _cluster_ci(df,value_col='covered',return_col='operational_return',B=2000,seed=20260811):
    if len(df)==0:return (np.nan,np.nan,np.nan,np.nan)
    x=df.copy();x['_cluster']=x.replication.astype(str)+'|'+x.MineID.astype(str);cl=[]
    for _,g in x.groupby('_cluster',sort=True):
        ret=g[return_col].astype(bool).to_numpy();cl.append((int(ret.sum()),int((g.loc[ret,value_col].astype(bool)).sum())))
    denom=sum(a for a,b in cl);num=sum(b for a,b in cl);est=num/denom if denom else np.nan
    if len(cl)<2 or denom==0:return (est,np.nan,np.nan,np.nan)
    rng=np.random.default_rng(int(seed));arr=np.asarray(cl,int);vals=[]
    for _ in range(int(B)):
        z=arr[rng.integers(0,len(arr),size=len(arr))];d=int(z[:,0].sum());vals.append(float(z[:,1].sum()/d) if d else np.nan)
    v=np.asarray(vals,float);v=v[np.isfinite(v)];return (est,float(np.std(v,ddof=1)) if len(v)>1 else np.nan,float(np.quantile(v,.025)) if len(v) else np.nan,float(np.quantile(v,.975)) if len(v) else np.nan)


def _cluster_mean_ci(df,value_col,return_col='operational_return',B=2000,seed=20260811):
    if len(df)==0:return (np.nan,np.nan,np.nan,np.nan)
    x=df.copy();x['_cluster']=x.replication.astype(str)+'|'+x.MineID.astype(str)
    clusters=[]
    for _,g in x.groupby('_cluster',sort=True):
        ret=g[return_col].astype(bool).to_numpy() if return_col in g else np.ones(len(g),bool)
        v=pd.to_numeric(g.loc[ret,value_col],errors='coerce').to_numpy(float);v=v[np.isfinite(v)]
        if len(v):clusters.append((float(v.sum()),int(len(v))))
    den=sum(n for _,n in clusters); est=sum(a for a,_ in clusters)/den if den else np.nan
    if len(clusters)<2 or den==0:return (est,np.nan,np.nan,np.nan)
    rng=np.random.default_rng(int(seed));arr=np.asarray(clusters,float);vals=[]
    for _ in range(int(B)):
        z=arr[rng.integers(0,len(arr),size=len(arr))];d=float(z[:,1].sum());vals.append(float(z[:,0].sum()/d) if d>0 else np.nan)
    v=np.asarray(vals,float);v=v[np.isfinite(v)];return (est,float(np.std(v,ddof=1)) if len(v)>1 else np.nan,float(np.quantile(v,.025)) if len(v) else np.nan,float(np.quantile(v,.975)) if len(v) else np.nan)


def _paired_cluster_effects(g,comparator,B,seed):
    z=g[g.method.isin(['M6',comparator])].copy();keys=['case_id','replication','MineID','target_rank','target_block_id'];
    ret=z.pivot_table(index=keys,columns='method',values='operational_return',aggfunc='first');cov=z.pivot_table(index=keys,columns='method',values='covered',aggfunc='first')
    if not {'M6',comparator}.issubset(ret.columns):return {'common_return_n':0,'coverage_difference_common_return':np.nan,'coverage_diff_ci_low':np.nan,'coverage_diff_ci_high':np.nan,'return_rate_difference':np.nan,'return_diff_ci_low':np.nan,'return_diff_ci_high':np.nan}
    w=ret[['M6',comparator]].astype(bool).copy();w['m6_cov']=cov['M6'].astype(bool);w['cmp_cov']=cov[comparator].astype(bool);w=w.reset_index();w['_cluster']=w.replication.astype(str)+'|'+w.MineID.astype(str)
    cl=[]
    for _,x in w.groupby('_cluster',sort=True):
        common=x.M6&x[comparator];cl.append((int(common.sum()),int(x.loc[common,'m6_cov'].sum()),int(x.loc[common,'cmp_cov'].sum()),len(x),int(x.M6.sum()),int(x[comparator].sum())))
    a=np.asarray(cl,float);cn=float(a[:,0].sum());cd=(float((a[:,1]-a[:,2]).sum()/cn) if cn>0 else np.nan);rn=float(a[:,3].sum());rd=float((a[:,4]-a[:,5]).sum()/rn) if rn>0 else np.nan
    if len(a)<2:return {'common_return_n':int(cn),'coverage_difference_common_return':cd,'coverage_diff_ci_low':np.nan,'coverage_diff_ci_high':np.nan,'return_rate_difference':rd,'return_diff_ci_low':np.nan,'return_diff_ci_high':np.nan}
    rng=np.random.default_rng(int(seed));cv=[];rv=[]
    for _ in range(int(B)):
        x=a[rng.integers(0,len(a),size=len(a))];d=x[:,0].sum();cv.append(float((x[:,1]-x[:,2]).sum()/d) if d>0 else np.nan);n=x[:,3].sum();rv.append(float((x[:,4]-x[:,5]).sum()/n) if n>0 else np.nan)
    cv=np.asarray(cv,float);cv=cv[np.isfinite(cv)];rv=np.asarray(rv,float);rv=rv[np.isfinite(rv)]
    return {'common_return_n':int(cn),'coverage_difference_common_return':cd,'coverage_diff_ci_low':float(np.quantile(cv,.025)) if len(cv) else np.nan,'coverage_diff_ci_high':float(np.quantile(cv,.975)) if len(cv) else np.nan,'return_rate_difference':rd,'return_diff_ci_low':float(np.quantile(rv,.025)) if len(rv) else np.nan,'return_diff_ci_high':float(np.quantile(rv,.975)) if len(rv) else np.nan}

def build_summaries(query,inversion,exact,endpoint,dose,lomo,contract):
    q=query.copy();q['dose_bin']=pd.cut(q.A_star,[-1e-12,.2,.4,.6,.8,1.0000001],labels=['[0,.2]','(.2,.4]','(.4,.6]','(.6,.8]','(.8,1]'],include_lowest=True).astype(str);q['spatial_stratum']=pd.cut(q.spatial_axis_rank,[.8,.85,.90,.95,1.0000001],labels=['T1','T2','T3','T4'],include_lowest=True).astype(str)
    cov=[]
    keys=['case_id','scenario_id','track','method','truth_scale']
    for k,g in q.groupby(keys,dropna=False,sort=True):
        ret=g.operational_return.astype(bool);est,se,lo,hi=_cluster_ci(g,B=int(contract['statistics']['bootstrap_repetitions']),seed=stable_seed(contract['statistics']['bootstrap_seed'],*k));cov.append(dict(zip(keys,k),n_queries=len(g),n_returned=int(ret.sum()),refusal_rate=float(1-ret.mean()),selective_coverage=est,coverage_mc_se=se,coverage_ci_low=lo,coverage_ci_high=hi,absolute_calibration_error=abs(est-(1-contract['alpha'])) if np.isfinite(est) else np.nan,false_support_rate=float(g.false_support.astype(bool).mean()),mean_support_ess=float(g.support_ess.mean()),mean_method_ess=float(g.method_ess.mean()),mean_n2_pinsker_tv=float(g.n2_pinsker_tv.mean(skipna=True)),mean_n3_diagnostic_lower_bound=float(g.n3_diagnostic_lower_bound.mean(skipna=True))))
    coverage=pd.DataFrame(cov)
    local=[]
    for dim in ['MineID','dose_bin','spatial_stratum']:
        for k,g in q.groupby(keys+[dim],dropna=False,sort=True):
            ret=g.operational_return.astype(bool);sel=g.loc[ret,'covered'];local.append({**dict(zip(keys+[dim],k)),'local_dimension':dim,'local_level':str(k[-1]),'n_queries':len(g),'n_returned':int(ret.sum()),'selective_coverage':float(sel.mean()) if len(sel) else np.nan,'refusal_rate':float(1-ret.mean())})
    local=pd.DataFrame(local)
    fifth=[]
    cell=local[local.local_dimension.isin(['MineID','spatial_stratum'])]
    for k,g in cell.groupby(keys,dropna=False,sort=True):
        v=g.selective_coverage.dropna().to_numpy(float);fifth.append({**dict(zip(keys,k)),'fifth_percentile_local_coverage':float(np.quantile(v,.05)) if len(v) else np.nan,'local_cells':len(v)})
    fifth=pd.DataFrame(fifth)
    refusal=q.groupby(['case_id','scenario_id','track','method','refusal_code'],dropna=False).size().reset_index(name='count');den=q.groupby(['case_id','scenario_id','track','method']).size().reset_index(name='total');refusal=refusal.merge(den,on=['case_id','scenario_id','track','method']);refusal['fraction']=refusal['count']/refusal['total']
    # Full inversion efficiency on the exact registered 50 S4 identities. Add closed-form M2/M5 from query.
    inv=inversion.copy();base=q[(q.track=='RF_ET_FG_PRIMARY')&(q.truth_scale=='observed')&(q.target_rank==1)&(q.case_id.isin(contract['full_inversion']['case_ids']))&(q.replication.isin(contract['full_inversion']['replications']))&(q.method.isin(['M2','M5']))].copy();base=base[['case_id','replication','MineID','target_rank','target_block_id','method','covered','operational_return','lower','upper','width','interval_score','truth_y']];base['spatial_rho']=base.case_id.map({f'MDB_S4_RHO{x:03d}':x/100 for x in [0,20,40,60,80]})
    dlo,dhi=map(float,contract['candidate_domain'])
    base['raw_interval_width']=base.width
    base['hull_width']=np.where(base.operational_return.astype(bool),np.maximum(0.0,np.minimum(base.upper.astype(float),dhi)-np.maximum(base.lower.astype(float),dlo)),np.nan)
    base['hull_wis']=base.interval_score # Never cap an unbounded comparator for WIS.
    base['covered_raw_set']=base.covered;base['width_representation']='physical_domain_intersection_for_efficiency_only'
    invsmall=inv[['case_id','spatial_rho','replication','MineID','target_rank','target_block_id','method','covered_raw_set','hull_width','hull_wis','operational_return','refusal_code']].copy();invsmall['width_representation']='finite_domain_outer_set_hull'
    effraw=pd.concat([base[['case_id','spatial_rho','replication','MineID','target_rank','target_block_id','method','covered_raw_set','hull_width','hull_wis','operational_return','width_representation']],invsmall],ignore_index=True)
    eff=[]
    for k,g in effraw.groupby(['case_id','spatial_rho','method'],sort=True):
        ret=g.operational_return.astype(bool)
        wcov=_cluster_ci(g,value_col='covered_raw_set',return_col='operational_return',B=int(contract['statistics']['bootstrap_repetitions']),seed=stable_seed(contract['statistics']['bootstrap_seed'],'effcov',*k))
        wwid=_cluster_mean_ci(g,'hull_width','operational_return',B=int(contract['statistics']['bootstrap_repetitions']),seed=stable_seed(contract['statistics']['bootstrap_seed'],'effwid',*k))
        eff.append({'case_id':k[0],'spatial_rho':k[1],'method':k[2],'n':len(g),'n_returned':int(ret.sum()),'coverage':wcov[0],'coverage_ci_low':wcov[2],'coverage_ci_high':wcov[3],'mean_width':wwid[0],'width_ci_low':wwid[2],'width_ci_high':wwid[3],'mean_wis':float(g.loc[ret,'hull_wis'].mean()) if ret.any() else np.nan})
    efficiency=pd.DataFrame(eff)
    matched=[]
    for rho,g in efficiency.groupby('spatial_rho'):
        m6=g[g.method=='M6'];m5=g[g.method=='M5']
        if len(m6) and len(m5):
            a=m6.iloc[0];b=m5.iloc[0];d=abs(a.coverage-b.coverage);matched.append({'spatial_rho':rho,'M6_coverage':a.coverage,'M5_coverage':b.coverage,'coverage_difference':d,'coverage_matched':bool(d<=contract['full_inversion']['coverage_match_tolerance']),'M6_mean_width':a.mean_width,'M5_mean_width':b.mean_width,'M6_minus_M5_width':a.mean_width-b.mean_width if d<=contract['full_inversion']['coverage_match_tolerance'] else np.nan,'efficiency_claim_allowed':bool(d<=contract['full_inversion']['coverage_match_tolerance'])})
    matched=pd.DataFrame(matched)
    # Hero, S9, S10, exact and endpoint summaries.
    hero=coverage[(coverage.track=='RF_ET_FG_PRIMARY')&(coverage.truth_scale=='observed')&(coverage.case_id.isin(contract['full_inversion']['case_ids']))].copy();hero['spatial_rho']=hero.case_id.map({f'MDB_S4_RHO{x:03d}':x/100 for x in [0,20,40,60,80]});hero=hero.merge(efficiency[['case_id','method','mean_width','width_ci_low','width_ci_high','mean_wis']],on=['case_id','method'],how='left')
    s9=coverage[(coverage.scenario_id=='S9')&(coverage.track=='RF_ET_FG_PRIMARY')].copy()
    s10=coverage[(coverage.scenario_id=='S10')&(coverage.track=='RF_ET_FG_PRIMARY')&(coverage.truth_scale=='observed')].copy()
    exsum=exact.groupby(['case_id','scenario_id','method','reference']).agg(n=('covered','size'),coverage=('covered','mean'),mean_pvalue=('pvalue','mean'),mean_pvalue_abs_diff=('pvalue_absolute_difference','mean'),mean_orbit_kl=('full_vs_sparse_orbit_kl','mean'),mean_pinsker_tv=('full_vs_sparse_pinsker_tv','mean')).reset_index() if len(exact) else pd.DataFrame()
    epsum=endpoint.groupby(['case_id','track','method','endpoint_dose']).agg(n=('operational_return','size'),return_rate=('operational_return','mean'),coverage=('covered_observed','mean'),expected_refusal=('expected_refusal','max')).reset_index() if len(endpoint) else pd.DataFrame()
    # Dose-response RMSE and shape RMSE (contrasts relative to dose 0).
    dm=[]
    if len(dose):
        for k,g in dose.groupby(['case_id','scenario_id','track'],sort=True):
            rmse=float(np.sqrt(np.mean(g.error.to_numpy(float)**2)));gg=g.copy();basep=gg[gg.dose==0][['replication','MineID','target_rank','prediction','Y_true_hard_dose']].rename(columns={'prediction':'p0','Y_true_hard_dose':'y0'});gg=gg.merge(basep,on=['replication','MineID','target_rank'],how='left');shape=(gg.prediction-gg.p0)-(gg.Y_true_hard_dose-gg.y0);dm.append({'case_id':k[0],'scenario_id':k[1],'track':k[2],'dose_response_rmse':rmse,'dose_response_contrast_rmse':float(np.sqrt(np.mean(shape.to_numpy(float)**2))),'n_predictions':len(g)})
    dose_metrics=pd.DataFrame(dm)
    lomo_summary=lomo.groupby(['case_id','scenario_id','method']).agg(n=('covered','size'),return_rate=('operational_return','mean'),selective_coverage=('covered',lambda x: float(x[lomo.loc[x.index,'operational_return']].mean()) if lomo.loc[x.index,'operational_return'].any() else np.nan),mean_width=('width','mean')).reset_index() if len(lomo) else pd.DataFrame()
    # Coverage-loss diagnostic error: observable sparse diagnostic vs empirical undercoverage for M6.
    diag=[]
    for k,g in q[(q.method=='M6')&(q.truth_scale=='observed')].groupby(['case_id','scenario_id','track'],sort=True):
        ret=g.operational_return.astype(bool);covv=float(g.loc[ret,'covered'].mean()) if ret.any() else np.nan;under=max(0.,.9-covv) if np.isfinite(covv) else np.nan;pred=float(g.n2_pinsker_tv.mean(skipna=True));diag.append({'case_id':k[0],'scenario_id':k[1],'track':k[2],'empirical_undercoverage':under,'mean_sparse_tv_diagnostic':pred,'absolute_diagnostic_error':abs(pred-under) if np.isfinite(pred) and np.isfinite(under) else np.nan})
    diagnostic=pd.DataFrame(diag)
    # Prespecified paired effect-size summary for the S4 hero family. No post-hoc p-value thresholding.
    pe=[];basehero=q[(q.track=='RF_ET_FG_PRIMARY')&(q.truth_scale=='observed')&(q.case_id.isin(contract['full_inversion']['case_ids']))]
    for case_id,g in basehero.groupby('case_id',sort=True):
        rho=float(case_id.replace('MDB_S4_RHO',''))/100.0
        for cmp in ['M2','M3','M4','M5']:
            z=_paired_cluster_effects(g,cmp,int(contract['statistics']['bootstrap_repetitions']),stable_seed(contract['statistics']['bootstrap_seed'],'paired',case_id,cmp));pe.append({'case_id':case_id,'spatial_rho':rho,'comparator':cmp,**z})
    primary_effects=pd.DataFrame(pe)
    return {'coverage':coverage,'local':local,'fifth':fifth,'refusal':refusal,'efficiency_raw':effraw,'efficiency':efficiency,'matched':matched,'hero':hero,'s9':s9,'s10':s10,'exact_summary':exsum,'endpoint_summary':epsum,'dose_metrics':dose_metrics,'lomo_summary':lomo_summary,'diagnostic':diagnostic,'primary_effects':primary_effects}
