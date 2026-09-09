from __future__ import annotations
import hashlib,time
import numpy as np,pandas as pd
from .common import require,stable_seed,conservative_split_quantile,coverage
from .models import RealOutcomeModel
from .intervals import m1_interval,m5_interval,m3_membership
from geodose_stage5b.spatial import graph_info,fit_spatial_pseudolikelihood

METHODS=['M1','M2','M3','M4','M5','M6']
NA_TREATMENT_CODE='R20_NO_AUTHENTIC_LONGITUDINAL_TREATMENT'

def _track_kind(track): return 'rf' if track=='RF_PRIMARY' else 'xgb'

def _target_sample_ids(test,scale,track,n_per_mine):
    ids=set()
    for mid,g in test.groupby('MineID',sort=True):
        q=g[['block_id']].copy()
        q['_h']=q.block_id.astype(str).map(lambda b:hashlib.sha256(f'Stage6A-v1|{scale}|{track}|{b}'.encode()).hexdigest())
        q=q.sort_values(['_h','block_id'],kind='mergesort');ids.update(q.head(min(int(n_per_mine),len(q))).block_id.astype(str))
    return ids

def fit_model_and_spatial(units,edges,track,scale,contract):
    features=contract['predictors'];outcome=contract['outcome']['column']
    train=units[units.benchmark_role=='nuisance_training'].copy().reset_index(drop=True);require(len(train)>0,'No nuisance-training rows')
    model=RealOutcomeModel(_track_kind(track),features,stable_seed('GeoDoseCP-Stage6A-model-v1',scale,track),contract).fit(train,outcome)
    # Preserve the full inventory for graph indexing, but do not impute or predict inactive rows.
    u=units.copy().reset_index(drop=True)
    active_roles={'nuisance_training','support_audit','calibration','test_target'}
    active=u.benchmark_role.isin(active_roles).to_numpy(bool);active_idx=np.flatnonzero(active);require(len(active_idx)>0,'No active primary-analysis rows')
    pred=np.full(len(u),np.nan,float);res=np.full(len(u),np.nan,float)
    pred[active_idx]=model.predict(u.iloc[active_idx]);res[active_idx]=u.iloc[active_idx][outcome].to_numpy(float)-pred[active_idx]
    require(np.isfinite(pred[active_idx]).all() and np.isfinite(res[active_idx]).all(),'Nonfinite active prediction/residual')
    g=graph_info(u,edges);spatial={}
    for mid,gm in u[u.benchmark_role=='nuisance_training'].groupby('MineID',sort=True):
        idx=gm.index.to_numpy(int);sp=fit_spatial_pseudolikelihood(res,idx,g,contract['spatial_route']['rho_grid'],contract['spatial_route']['sigma_bounds']);spatial[str(mid)]=sp
    fit={'track':track,'scale':scale,'fit_id':model.fit_id,'training_hash':model.training_hash,'training_rows':len(train),
         'prediction_rows':int(len(active_idx)),'prediction_role_contract':sorted(active_roles),'features':features,'hyperparameters':model.hp,'spatial_by_mine':spatial}
    imp=model.feature_importance();imp['track']=track;imp['scale']=scale
    return model,u,pred,res,g,fit,imp

def evaluate_mine(units,edges,track,scale,mine_id,contract,model,pred,residual,g,spatial):
    t0=time.time();outcome=contract['outcome']['column'];domain=contract['candidate_domain'];alpha=float(contract['alpha']);mid=str(mine_id)
    test=units[(units.benchmark_role=='test_target')&(units.MineID.astype(str)==mid)].copy();require(len(test)>0,f'No test targets {mid}')
    allcal=units[units.benchmark_role=='calibration'].copy();samecal=allcal[allcal.MineID.astype(str)==mid].copy();require(len(samecal)>=30,'Insufficient same-mine calibration')
    allcal_idx=allcal.index.to_numpy(int);cal_idx=samecal.index.to_numpy(int);m1scores=np.abs(residual[allcal_idx]);require(np.isfinite(m1scores).all(),'M1 calibration residual nonfinite')
    n=int(contract['candidate_inversion']['sample_targets_per_mine'][scale][track]);sample=_target_sample_ids(test,scale,track,n)
    rows=[];reductions=[];sp=spatial[mid]
    for r in test.itertuples():
        ti=int(r.Index);center=float(pred[ti]);y=float(getattr(r,outcome))
        base={'scale':scale,'track':track,'MineID':r.MineID,'MineN':r.MineN,'block_id':str(r.block_id),'observed_y':y,'center':center,
              'strict_quality_ue20':bool(r.strict_quality_ue20),'snapshot_exposure_descriptive':float(r.mapped_rehabilitation_fraction),'role':'test_target'}
        z=m1_interval(center,m1scores,alpha,domain)
        rows.append({**base,'method':'M1','returned':True,'refusal_code':'','covered_observed_product':coverage(y,z['lower'],z['upper']),'pvalue':np.nan,
                     'graph_safe_count':np.nan,'rho':np.nan,'sigma':np.nan,'n3_diagnostic_lower_bound':np.nan,'delta_sparse_pinsker_tv':np.nan,'interval_inverted':True,**z})
        for method in ['M2','M4','M6']:
            rows.append({**base,'method':method,'returned':False,'refusal_code':NA_TREATMENT_CODE,'covered_observed_product':np.nan,'pvalue':np.nan,
                         'graph_safe_count':np.nan,'rho':np.nan,'sigma':np.nan,'n3_diagnostic_lower_bound':np.nan,'delta_sparse_pinsker_tv':np.nan,
                         'interval_inverted':False,'lower':np.nan,'upper':np.nan,'width':np.nan,'raw_lower':np.nan,'raw_upper':np.nan,'quantile':np.nan,'calibration_count':0})
        m3=m3_membership(center,y,ti,cal_idx,residual,units,g,float(sp['rho']),float(sp['sigma']),contract,do_inversion=(str(r.block_id) in sample))
        rows.append({**base,'method':'M3','returned':bool(m3.get('returned',False)),'refusal_code':m3.get('refusal_code',''),
                     'covered_observed_product':m3.get('covered',np.nan),'pvalue':m3.get('pvalue',np.nan),'graph_safe_count':m3.get('graph_safe_count',np.nan),
                     'rho':m3.get('rho',float(sp['rho'])),'sigma':m3.get('sigma',float(sp['sigma'])),'n3_diagnostic_lower_bound':m3.get('n3_diagnostic_lower_bound',np.nan),
                     'delta_sparse_pinsker_tv':m3.get('delta_sparse_pinsker_tv',np.nan),'interval_inverted':bool(m3.get('interval_inverted',False)),
                     'lower':m3.get('lower',np.nan),'upper':m3.get('upper',np.nan),'width':m3.get('width',np.nan),'raw_lower':np.nan,'raw_upper':np.nan,'quantile':np.nan,
                     'calibration_count':len(cal_idx),'component_count':m3.get('component_count',np.nan),'raw_set_width':m3.get('raw_set_width',np.nan),
                     'hull_inflation':m3.get('hull_inflation',np.nan),'left_domain_truncated':m3.get('left_domain_truncated',False),'right_domain_truncated':m3.get('right_domain_truncated',False),
                     'components_json':m3.get('components_json',''),'error_type':m3.get('error_type',''),'error_message':m3.get('error_message','')})
        if m3.get('returned',False):
            reductions.append({'scale':scale,'track':track,'MineID':r.MineID,'MineN':r.MineN,'block_id':str(r.block_id),
                               'audit_selected_for_inversion':str(r.block_id) in sample,'M3_pvalue':float(m3['pvalue']),
                               'M4_zero_treatment_ratio_pvalue':float(m3['m4_zero_ratio_pvalue']),'M4_M3_abs_diff':float(m3['m4_m3_abs_diff']),
                               'M2_reduction_note':'Unit treatment ratio would reduce dose-only WCP to ordinary split CP; no NSW treatment model is fit.',
                               'M6_status':'NOT_APPLICABLE_NO_AUTHENTIC_LONGITUDINAL_TREATMENT'})
        m5=m5_interval(center,ti,cal_idx,residual,g,alpha,domain,contract['support_thresholds'])
        rows.append({**base,'method':'M5','returned':bool(m5.get('returned',False)),'refusal_code':m5.get('refusal_code',''),
                     'covered_observed_product':coverage(y,m5['lower'],m5['upper']) if m5.get('returned',False) else np.nan,'pvalue':np.nan,
                     'graph_safe_count':m5.get('graph_safe_count',np.nan),'rho':np.nan,'sigma':np.nan,'n3_diagnostic_lower_bound':np.nan,'delta_sparse_pinsker_tv':np.nan,
                     'interval_inverted':bool(m5.get('returned',False)),'lower':m5.get('lower',np.nan),'upper':m5.get('upper',np.nan),'width':m5.get('width',np.nan),
                     'raw_lower':m5.get('raw_lower',np.nan),'raw_upper':m5.get('raw_upper',np.nan),'quantile':m5.get('quantile',np.nan),
                     'calibration_count':m5.get('calibration_count',0),'ess':m5.get('ess',np.nan),'max_weight':m5.get('max_weight',np.nan)})
    df=pd.DataFrame(rows);red=pd.DataFrame(reductions)
    for method in ['M2','M4','M6']:require(not df.loc[df.method==method,'returned'].astype(bool).any(),f'{method} returned despite treatment gate')
    require(set(df.method)==set(METHODS),'Method registry incomplete')
    return df,red,{'scale':scale,'track':track,'MineID':mid,'MineN':test.MineN.iloc[0],'rows':len(df),'test_targets':len(test),
                   'inversion_sample_count':int(df[(df.method=='M3')&df.interval_inverted.astype(bool)].block_id.nunique()),'runtime_seconds':time.time()-t0}

def leakage_diagnostic(units90,contract):
    outcome=contract['outcome']['column'];features=contract['predictors'];alpha=float(contract['alpha']);domain=contract['candidate_domain'];rows=[]
    active=units90[units90.primary_component.astype(bool)&units90.real_baseline_eligible.astype(bool)].copy().reset_index(drop=True)
    h=active.block_id.astype(str).map(lambda b:int(hashlib.sha256(f'Stage6A-random-split|{b}'.encode()).hexdigest()[:16],16)/16**16)
    train=active[h<.50].copy();cal=active[(h>=.50)&(h<.75)].copy();test=active[h>=.75].copy();require(min(len(train),len(cal),len(test))>100,'Random diagnostic split insufficient')
    m=RealOutcomeModel('rf',features,stable_seed('Stage6A-leakage-random'),contract).fit(train,outcome);pc=m.predict(cal);pt=m.predict(test);q=conservative_split_quantile(np.abs(cal[outcome].to_numpy(float)-pc),alpha)
    lo=np.maximum(domain[0],pt-q);hi=np.minimum(domain[1],pt+q);yy=test[outcome].to_numpy(float)
    rows.append({'split':'random_block','held_out_mine':'','training_rows':len(train),'calibration_rows':len(cal),'test_rows':len(test),'coverage':float(np.mean((yy>=lo)&(yy<=hi))),
                 'mean_width':float(np.mean(hi-lo)),'rmse':float(np.sqrt(np.mean((yy-pt)**2))),'spatial_buffered':False,'causal_metric':False})
    train=active[active.benchmark_role=='nuisance_training'].copy();cal=active[active.benchmark_role=='calibration'].copy();test=active[active.benchmark_role=='test_target'].copy()
    m=RealOutcomeModel('rf',features,stable_seed('GeoDoseCP-Stage6A-model-v1','90m','RF_PRIMARY'),contract).fit(train,outcome);pc=m.predict(cal);pt=m.predict(test);q=conservative_split_quantile(np.abs(cal[outcome].to_numpy(float)-pc),alpha)
    lo=np.maximum(domain[0],pt-q);hi=np.minimum(domain[1],pt+q);yy=test[outcome].to_numpy(float)
    rows.append({'split':'buffered_spatial_block','held_out_mine':'','training_rows':len(train),'calibration_rows':len(cal),'test_rows':len(test),'coverage':float(np.mean((yy>=lo)&(yy<=hi))),
                 'mean_width':float(np.mean(hi-lo)),'rmse':float(np.sqrt(np.mean((yy-pt)**2))),'spatial_buffered':True,'causal_metric':False})
    frac=float(contract['leakage_diagnostic']['lomo_source_training_fraction'])
    for mid,gtest in active.groupby('MineID',sort=True):
        src=active[active.MineID.astype(str)!=str(mid)].copy();hv=src.block_id.astype(str).map(lambda b:int(hashlib.sha256(f'Stage6A-lomo-source|{mid}|{b}'.encode()).hexdigest()[:16],16)/16**16)
        tr=src[hv<frac].copy();ca=src[hv>=frac].copy();require(len(tr)>100 and len(ca)>50,'LOMO diagnostic source insufficient')
        m=RealOutcomeModel('rf',features,stable_seed('Stage6A-lomo',mid),contract).fit(tr,outcome);pc=m.predict(ca);pt=m.predict(gtest);q=conservative_split_quantile(np.abs(ca[outcome].to_numpy(float)-pc),alpha)
        lo=np.maximum(domain[0],pt-q);hi=np.minimum(domain[1],pt+q);yy=gtest[outcome].to_numpy(float)
        rows.append({'split':'leave_one_mine_out','held_out_mine':gtest.MineN.iloc[0],'training_rows':len(tr),'calibration_rows':len(ca),'test_rows':len(gtest),
                     'coverage':float(np.mean((yy>=lo)&(yy<=hi))),'mean_width':float(np.mean(hi-lo)),'rmse':float(np.sqrt(np.mean((yy-pt)**2))),
                     'spatial_buffered':False,'causal_metric':False,'local_orbit_methods_status':'M3/M4/M6 not transferable across disconnected mine graphs'})
    return pd.DataFrame(rows)
