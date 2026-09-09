from __future__ import annotations
from dataclasses import dataclass
import math,time
import numpy as np,pandas as pd
from .common import require,Stage5BError,stable_seed,interval_score
from .models import OutcomeModel,MixedPropensity,oracle_density,oracle_mean,robust_scale
from .spatial import graph_info,deterministic_graph_safe,fit_spatial_pseudolikelihood,nearest_calibration_block,localization_diagnostic,spatial_diagnostics,six_precision
from .orbit import eval_m3,eval_m4,eval_m6,invert_candidate,orbit_kl,eval_m3_m4_grid,eval_m6_grid,upper_tail_mask
from minedosebench.math import q_at,conditional_mean,measurement_error,sample_localized

METHODS=['M2','M3','M4','M5','M6']

@dataclass
class Track:
    name:str; kind:str; outcome:object; propensity:object; rho:float; sigma:float; power:float; fit_id:str; prop_id:str; theorem_certified:bool

def _sorted_calibration(task):
    c=task.units[task.units.benchmark_role=='calibration'].copy()
    frac=float(task.case.get('calibration_subsample_fraction',1.0))
    if frac<.999999:
        parts=[]
        for mid,g in c.groupby('MineID',sort=True):
            g=g.sort_values('block_id',kind='mergesort').reset_index(drop=True); n=max(5,int(round(frac*len(g)))); rng=np.random.default_rng(stable_seed('GeoDoseCP-Stage5B-CalSub-v1',task.case['case_id'],task.rep,mid));pick=np.sort(rng.choice(len(g),size=min(n,len(g)),replace=False));parts.append(g.iloc[pick])
        c=pd.concat(parts,ignore_index=True)
    return c.reset_index(drop=True)

def _q_values(a,dose,h,audited,observational=False):
    a=np.asarray(a,float)
    if observational:return np.ones(len(a),float)
    return q_at(a,float(dose),None if h is None or pd.isna(h) else float(h),bool(audited))

def _dose_logs(cal,target_row,dose,h,audited,track,observational=False):
    if observational:return np.zeros(len(cal)),0.0
    q=_q_values(cal.A.to_numpy(float),dose,h,audited,False);g=oracle_density(cal) if track.kind=='oracle' else track.propensity.density(cal);logs=np.full(len(cal),-np.inf);ok=(q>0)&(g>0)&np.isfinite(g);logs[ok]=np.log(q[ok])-np.log(g[ok]);qa=float(_q_values(np.array([target_row.A_star]),dose,h,audited,False)[0]);tdf=target_row.unit_df;gt=float(oracle_density(tdf,a=np.array([target_row.A_star]))[0]) if track.kind=='oracle' else float(track.propensity.density(tdf,a=np.array([target_row.A_star]))[0]);tl=float(np.log(qa)-np.log(gt)) if qa>0 and gt>0 and np.isfinite(gt) else -np.inf;return logs,tl

def _weight_diagnostics(logw,target_logw):
    x=np.asarray(logw,float);fin=np.isfinite(x)
    if not fin.any() or not np.isfinite(target_logw):return {'positive_weight_count':int(fin.sum()),'support_ess':0.,'support_max_weight':1.,'target_normalized_weight':1.,'support_status':'no_positive'}
    z=x[fin];m=float(z.max());w=np.exp(z-m);s=float(w.sum());ess=float(s*s/np.sum(w*w));maxw=float((w/s).max());tw=float(np.exp(target_logw-max(float(z.max()),float(target_logw))));cw=np.exp(z-max(float(z.max()),float(target_logw)));tn=float(tw/(tw+cw.sum()));return {'positive_weight_count':int(fin.sum()),'support_ess':ess,'support_max_weight':maxw,'target_normalized_weight':tn,'support_status':'positive'}

def _wcp(center,scores,logw,target_logw,alpha):
    scores=np.asarray(scores,float);logw=np.asarray(logw,float);order=np.argsort(scores,kind='mergesort');scores=scores[order];logw=logw[order];fin=np.isfinite(logw)&np.isfinite(scores)
    if not fin.any() or not np.isfinite(target_logw):return {'raw_status':'refused','lower':np.nan,'upper':np.nan,'quantile':np.nan}
    z=logw[fin];sc=scores[fin];m=max(float(z.max()),float(target_logw));w=np.exp(z-m);tw=float(np.exp(target_logw-m));total=float(w.sum()+tw);wn=w/total;level=1-alpha
    if float(wn.sum())+1e-15<level:return {'raw_status':'returned_infinite_target_pseudomass','lower':-np.inf,'upper':np.inf,'quantile':np.inf}
    k=min(int(np.searchsorted(np.cumsum(wn),level,side='left')),len(sc)-1);q=float(sc[k]);return {'raw_status':'returned_finite','lower':float(center-q),'upper':float(center+q),'quantile':q}

def _refusal(method,audited_endpoint,support_diag,graph_count,n3_lb,raw_ok,contract):
    # Exact Stage3F operational-gate mapping carried forward unchanged:
    # ESS/max-weight -> M2/M4/M5/M6; graph-safe count -> M3/M4/M5/M6;
    # N3 non-vacuity -> M6 only. M3 never inherits a dose-weight ESS gate.
    th=contract['support_thresholds']
    if not audited_endpoint:return 'R02_ENDPOINT_NOT_AUDITED'
    if support_diag['support_status']!='positive':return 'R03_NO_POSITIVE_INTERVENTION_SUPPORT'
    if not raw_ok:return 'R10_COMPUTATIONAL_FAILURE'
    if method in {'M2','M4','M5','M6'}:
        if float(support_diag['support_ess'])<float(th['minimum_ess']):return 'R04_ESS_BELOW_FROZEN_THRESHOLD'
        if float(support_diag['support_max_weight'])>float(th['maximum_normalized_weight'])+1e-15:return 'R05_MAX_WEIGHT_ABOVE_FROZEN_THRESHOLD'
    if method in {'M3','M4','M5','M6'} and int(graph_count)<int(th['minimum_graph_safe_count']):return 'R06_GRAPH_SAFE_COUNT_BELOW_FROZEN_THRESHOLD'
    if method=='M6' and (not np.isfinite(n3_lb) or float(n3_lb)<=0):return 'R08_N3_DIAGNOSTIC_VACUOUS'
    return ''

def _oracle_support_truth(method,audited,full_diag,safe_diag,graph_count,n3_lb,contract):
    # Benchmark-only oracle support label used solely to score false support.
    # It never enters target selection, nuisance fitting, interval construction, or estimated-track refusal.
    if not audited:return False
    th=contract['support_thresholds']; d=safe_diag if method=='M5' else full_diag
    if d.get('support_status')!='positive':return False
    if method in {'M2','M4','M5','M6'}:
        if float(d.get('support_ess',0.0))<float(th['minimum_ess']):return False
        if float(d.get('support_max_weight',1.0))>float(th['maximum_normalized_weight'])+1e-15:return False
    if method in {'M3','M4','M5','M6'} and int(graph_count)<int(th['minimum_graph_safe_count']):return False
    if method=='M6' and (not np.isfinite(n3_lb) or float(n3_lb)<=0):return False
    return True

def _row_with_unit(truth_row,unit_df):
    class O:pass
    o=O()
    for k,v in truth_row.items():setattr(o,k,v)
    o.unit_df=unit_df
    return o

def _track_center(track,case,df,a):
    if track.kind=='oracle':return np.asarray(oracle_mean(case,df,np.asarray(a,float)),float)
    return track.outcome.predict(df,np.asarray(a,float))

def _own_residuals(track,case,df):
    if track.kind=='oracle':mean=oracle_mean(case,df,df.A.to_numpy(float))
    else:mean=track.outcome.predict(df)
    return df.Y_observed_at_A.to_numpy(float)-mean

def _m6_matrices(track,case,slotdf,source_A,source_Y,dose,h,audited,observational):
    R=np.empty((6,6));T=np.full((6,6),-np.inf);J=np.zeros((6,6))
    for i in range(6):
        dfi=pd.concat([slotdf.iloc[[i]]]*6,ignore_index=True);means=_track_center(track,case,dfi,source_A);R[i,:]=source_Y-means
        if i==5 and not observational:
            q=_q_values(source_A,dose,h,audited,False);ok=q>0;T[i,ok]=np.log(q[ok])
        else:
            gd=oracle_density(dfi,a=source_A) if track.kind=='oracle' else track.propensity.density(dfi,a=source_A);ok=(gd>0)&np.isfinite(gd);T[i,ok]=np.log(gd[ok])
        # Frozen Stage5B outcome residual transformation uses s_i(a)=1. The inverse Jacobian is explicitly included as log(1)=0.
        J[i,:]=0.
    return R,T,J

def fit_tracks(task,contract):
    train=task.units[task.units.benchmark_role=='nuisance_training'].copy().reset_index(drop=True);support=task.units[task.units.benchmark_role=='support_audit'].copy().reset_index(drop=True);features=task.allowed_predictors;require(features,'No finite allowed predictors')
    prop=MixedPropensity(features,stable_seed('Stage5B-propensity',task.case['case_id'],task.rep)).fit(train)
    rf=OutcomeModel('rf',features,stable_seed('Stage5B-rf',task.case['case_id'],task.rep),contract).fit(train)
    g=graph_info(task.units,task.working_edges);idx={b:i for i,b in enumerate(task.units.block_id.astype(str))};nidx=np.array([idx[b] for b in train.block_id.astype(str)],int);rf_res=np.zeros(len(task.units));rf_res[nidx]=train.Y_observed_at_A.to_numpy(float)-rf.predict(train);sp=fit_spatial_pseudolikelihood(rf_res,nidx,g,contract['spatial_route']['rho_grid'],contract['spatial_route']['sigma_bounds'])
    tracks=[Track('RF_ET_FG_PRIMARY','estimated',rf,prop,float(sp['rho']),float(sp['sigma']),1.0,rf.fit_id,prop.fit_id,False)]
    # Oracle treatment/true graph/true conditional mean diagnostic. No nuisance estimation claim.
    power=1.5 if task.case['residual_law']=='transformed_gmrf_power_1_5' else 1.0;tracks.append(Track('ORACLE_OT_TG_DIAGNOSTIC','oracle',None,None,float(task.case['spatial_rho']),.04,power,'oracle_mean','oracle_g',bool(task.case['residual_law']!='transformed_gmrf_power_1_5')))
    if task.case['scenario_id'] in set(contract['confirmation_scenario_ids']):
        xgb=OutcomeModel('xgb',features,stable_seed('Stage5B-xgb',task.case['case_id'],task.rep),contract).fit(train);xr=train.Y_observed_at_A.to_numpy(float)-xgb.predict(train);xfull=np.zeros(len(task.units));xfull[nidx]=xr;xsp=fit_spatial_pseudolikelihood(xfull,nidx,g,contract['spatial_route']['rho_grid'],contract['spatial_route']['sigma_bounds']);tracks.append(Track('XGB_ET_FG_CONFIRMATION','estimated',xgb,prop,float(xsp['rho']),float(xsp['sigma']),1.0,xgb.fit_id,prop.fit_id,False))
    return tracks,g,train,support

def evaluate_task(task,contract):
    started=time.time();tracks,g,train,support=fit_tracks(task,contract);g_true=graph_info(task.units,task.true_edges);cal=_sorted_calibration(task);idx={b:i for i,b in enumerate(task.units.block_id.astype(str))};cal_idx=np.array([idx[b] for b in cal.block_id.astype(str)],int);support_idx=np.array([idx[b] for b in support.block_id.astype(str)],int)
    truth_lookup=task.target_truth.set_index(['MineID','target_rank']); target_rows=[]; exact_rows=[]; inv_rows=[]; endpoint_rows=[]; dose_rows=[];fit_rows=[];spatial_rows=[]; oracle_support_cache={}
    for track in tracks:
        # Fit audit and spatial diagnostics. Oracle is a diagnostic track, not a fitted nuisance theorem.
        if track.kind=='oracle':train_res=train.Y_observed_at_A.to_numpy(float)-oracle_mean(task.case,train,train.A.to_numpy(float))
        else:train_res=train.Y_observed_at_A.to_numpy(float)-track.outcome.predict(train)
        full_res=np.zeros(len(task.units));full_res[[idx[b] for b in train.block_id.astype(str)]]=train_res
        if track.kind=='oracle':support_res=support.Y_observed_at_A.to_numpy(float)-oracle_mean(task.case,support,support.A.to_numpy(float))
        else:support_res=support.Y_observed_at_A.to_numpy(float)-track.outcome.predict(support)
        full_support_res=np.zeros(len(task.units));full_support_res[support_idx]=support_res
        fit_rows.append({'case_id':task.case['case_id'],'scenario_id':task.case['scenario_id'],'replication':task.rep,'track':track.name,'predictor':'oracle' if track.kind=='oracle' else track.outcome.kind,'outcome_fit_id':track.fit_id,'propensity_fit_id':track.prop_id,'n_train':len(train),'n_calibration':len(cal),'feature_count':len(task.allowed_predictors),'feature_list':'|'.join(task.allowed_predictors),'spatial_rho':track.rho,'spatial_sigma':track.sigma,'empirical_nuisance_theorem_certified':False,'support_outcomes_used_for_fit':False,'calibration_outcomes_used_for_fit':False,'test_outcomes_used_for_fit':False})
        spatial_rows.extend(spatial_diagnostics(task.case['case_id'],task.rep,track.name,full_support_res,task.units,g,support_idx,stable_seed('Stage5B-spdiag',task.case['case_id'],task.rep,track.name)))
        owncal=_own_residuals(track,task.case,cal);cal_score=np.abs(owncal)
        for tr in task.targets.itertuples(index=False):
            ttruth=truth_lookup.loc[(str(tr.MineID),int(tr.target_rank))].to_dict();tid=str(tr.target_block_id);ti=idx[tid];tdf=task.units.iloc[[ti]].copy().reset_index(drop=True);tobj=_row_with_unit({**ttruth,'A_star':float(ttruth['A_star'])},tdf);dose=ttruth.get('requested_dose');h=ttruth.get('bandwidth');observational=task.case['primary_target_mode']=='observational';audited=bool(observational or task.case['endpoint_structure']!='interior_only' or (not pd.isna(dose) and float(dose) not in (0.,1.)))
            if observational:dose=float(task.units.iloc[ti].A);h=np.nan
            dlog,tlog=_dose_logs(cal,tobj,dose,h,audited,track,observational);sdiag=_weight_diagnostics(dlog,tlog)
            # M2/M5 center at realized A*.
            center=float(_track_center(track,task.case,tdf,np.array([float(ttruth['A_star'])]))[0]);m2=_wcp(center,cal_score,dlog,tlog,float(contract['alpha']))
            safe=deterministic_graph_safe(ti,cal_idx,g);safe_set=set(map(int,safe));safe_mask=np.array([i in safe_set for i in cal_idx]);m5_sdiag=_weight_diagnostics(dlog[safe_mask],tlog) if safe_mask.any() else {'support_status':'no_positive','support_ess':0.0,'support_max_weight':1.0,'positive_weight_count':0,'target_normalized_weight':1.0};m5=_wcp(center,cal_score[safe_mask],dlog[safe_mask],tlog,float(contract['alpha'])) if safe_mask.any() else {'raw_status':'refused','lower':np.nan,'upper':np.nan,'quantile':np.nan}
            # Local six-slot scalable orbit.
            try:
                bcal=nearest_calibration_block(ti,cal_idx,task.units,5);P,nd=localization_diagnostic(ti,bcal,cal_idx,task.units,g,track.rho,int(contract['spatial_route']['m']),int(contract['spatial_route']['sensitivity_m']));P=P/(track.sigma*track.sigma);bdf=task.units.iloc[np.r_[bcal,ti]].copy().reset_index(drop=True);calb=bdf.iloc[:5];src_res=np.r_[_own_residuals(track,task.case,calb),float(ttruth['Y_observed_A_star'])-center];m3obs=eval_m3(src_res,P,track.power);local_error_type='';local_error_message='';six_d=np.r_[dlog[[np.flatnonzero(cal.block_id.astype(str).to_numpy()==str(x))[0] for x in calb.block_id]],tlog];m4obs=eval_m4(m3obs,src_res,six_d);source_A=np.r_[calb.A.to_numpy(float),float(ttruth['A_star'])];source_Y_obs=np.r_[calb.Y_observed_at_A.to_numpy(float),float(ttruth['Y_observed_A_star'])];R,T,J=_m6_matrices(track,task.case,bdf,source_A,source_Y_obs,dose,h,audited,observational);m6obs=eval_m6(R,T,J,P,track.power);local_ok=True
            except Exception as exc:
                local_error_type=type(exc).__name__;local_error_message=str(exc)[:500];bcal=np.array([],int);nd={'n3_diagnostic_lower_bound':np.nan,'m64_count':0,'delta_sparse_localization_kl':np.nan,'delta_sparse_pinsker_tv':np.nan};m3obs=m4obs=m6obs=None;local_ok=False
            if tid not in oracle_support_cache:
                oracle_track=Track('ORACLE_SUPPORT_TRUTH','oracle',None,None,float(task.case['spatial_rho']),.04,1.0,'oracle','oracle',True)
                odlog,otlog=_dose_logs(cal,tobj,dose,h,audited,oracle_track,observational);ofull=_weight_diagnostics(odlog,otlog)
                true_safe=deterministic_graph_safe(ti,cal_idx,g_true);true_safe_set=set(map(int,true_safe));true_safe_mask=np.array([i in true_safe_set for i in cal_idx]);osafe=_weight_diagnostics(odlog[true_safe_mask],otlog) if true_safe_mask.any() else {'support_status':'no_positive','support_ess':0.0,'support_max_weight':1.0}
                try:
                    obcal=nearest_calibration_block(ti,cal_idx,task.units,5);_,ond=localization_diagnostic(ti,obcal,cal_idx,task.units,g_true,float(task.case['spatial_rho']),int(contract['spatial_route']['m']),int(contract['spatial_route']['sensitivity_m']));on3=float(ond.get('n3_diagnostic_lower_bound',np.nan))
                except Exception:on3=np.nan
                oracle_support_cache[tid]=(ofull,osafe,on3,int(len(true_safe)))
            ofull,osafe,on3,oracle_graph_count=oracle_support_cache[tid]
            for truth_scale in (['observed','latent'] if task.case['scenario_id']=='S9' else ['observed']):
                y=float(ttruth['Y_observed_A_star'] if truth_scale=='observed' else ttruth['Y_true_A_star'])
                orbitres={}
                if local_ok:
                    if truth_scale=='observed': orbitres={'M3':m3obs,'M4':m4obs,'M6':m6obs}
                    else:
                        src_res_lat=np.r_[_own_residuals(track,task.case,calb),float(ttruth['Y_true_A_star'])-center];m3l=eval_m3(src_res_lat,P,track.power);m4l=eval_m4(m3l,src_res_lat,six_d);source_Y_lat=np.r_[calb.Y_observed_at_A.to_numpy(float),float(ttruth['Y_true_A_star'])];Rl,Tl,Jl=_m6_matrices(track,task.case,bdf,source_A,source_Y_lat,dose,h,audited,observational);m6l=eval_m6(Rl,Tl,Jl,P,track.power);orbitres={'M3':m3l,'M4':m4l,'M6':m6l}
                for method in METHODS:
                    if method=='M2':raw=m2;raw_ok=raw['raw_status']!='refused';covered_raw=bool(raw_ok and raw['lower']<=y<=raw['upper']);meth_ess=sdiag['support_ess'];meth_max=sdiag['support_max_weight'];gcount=0;n3=np.nan
                    elif method=='M5':raw=m5;raw_ok=raw['raw_status']!='refused';covered_raw=bool(raw_ok and raw['lower']<=y<=raw['upper']);meth_ess=m5_sdiag['support_ess'];meth_max=m5_sdiag['support_max_weight'];gcount=len(safe);n3=np.nan
                    else:
                        rr=orbitres.get(method);raw_ok=rr is not None;covered_raw=bool(raw_ok and float(rr['pvalue'])>float(contract['alpha']));meth_ess=float(rr['ess']) if raw_ok else 0.;meth_max=float(rr['max_weight']) if raw_ok else 1.;gcount=len(safe);n3=(float(nd.get('n3_diagnostic_lower_bound',np.nan)) if method=='M6' else np.nan);raw={'raw_status':'pvalue_membership' if raw_ok else 'refused','lower':np.nan,'upper':np.nan,'quantile':np.nan}
                    gate_diag=m5_sdiag if method=='M5' else sdiag
                    code=_refusal(method,audited,gate_diag,gcount,n3,raw_ok,contract);oper=(code=='');covered=bool(oper and covered_raw);lo=raw.get('lower',np.nan);hi=raw.get('upper',np.nan);width=float(hi-lo) if np.isfinite(lo) and np.isfinite(hi) else (np.inf if np.isneginf(lo) and np.isposinf(hi) else np.nan);pv=float(orbitres[method]['pvalue']) if method in orbitres else np.nan
                    oracle_ok=_oracle_support_truth(method,audited,ofull,osafe,oracle_graph_count,on3,contract);false_support=bool(oper and not oracle_ok)
                    target_rows.append({'case_id':task.case['case_id'],'scenario_id':task.case['scenario_id'],'replication':task.rep,'MineID':str(tr.MineID),'target_rank':int(tr.target_rank),'target_block_id':tid,'track':track.name,'method':method,'truth_scale':truth_scale,'A_star':float(ttruth['A_star']),'requested_dose':np.nan if pd.isna(ttruth.get('requested_dose')) else float(ttruth.get('requested_dose')),'bandwidth':np.nan if pd.isna(ttruth.get('bandwidth')) else float(ttruth.get('bandwidth')),'truth_y':y,'center':center,'raw_status':raw['raw_status'],'raw_covered':covered_raw,'operational_return':oper,'refusal_code':code,'covered':covered,'lower':lo,'upper':hi,'width':width,'interval_score':interval_score(lo,hi,y,float(contract['alpha'])) if method in {'M2','M5'} and oper else np.nan,'candidate_pvalue':pv,'support_ess':float(gate_diag['support_ess']),'support_max_weight':float(gate_diag['support_max_weight']),'method_ess':float(meth_ess),'method_max_weight':float(meth_max),'graph_safe_or_local_count':int(gcount),'n2_localization_kl':float(nd.get('delta_sparse_localization_kl',np.nan)) if method in {'M3','M4','M6'} else np.nan,'n2_pinsker_tv':float(nd.get('delta_sparse_pinsker_tv',np.nan)) if method in {'M3','M4','M6'} else np.nan,'n3_diagnostic_lower_bound':n3,'n3_theorem_certified':False,'oracle_support_truth':bool(oracle_ok),'false_support':false_support,'computational_failure':bool(code=='R10_COMPUTATIONAL_FAILURE'),'local_error_type':(local_error_type if method in {'M3','M4','M6'} else ''),'local_error_message':(local_error_message if method in {'M3','M4','M6'} else ''),'target_selection_mode':str(tr.target_selection_mode),'target_design_sampling_weight':float(tr.target_design_sampling_weight),'spatial_axis_rank':float(task.units.iloc[ti].spatial_axis_rank) if pd.notna(task.units.iloc[ti].spatial_axis_rank) else np.nan,'method_family_status':'naive_non_theorem_comparator' if method=='M4' else ('GeoDose_CP' if method=='M6' else 'principal_comparator')})
            # Hard-dose predictor predictions, method-independent.
            htruth=task.hard_truth[(task.hard_truth.MineID.astype(str)==str(tr.MineID))&(task.hard_truth.target_rank==int(tr.target_rank))]
            for hr in htruth.itertuples(index=False):
                pred=float(_track_center(track,task.case,tdf,np.array([float(hr.dose)]))[0]);dose_rows.append({'case_id':task.case['case_id'],'scenario_id':task.case['scenario_id'],'replication':task.rep,'MineID':str(tr.MineID),'target_rank':int(tr.target_rank),'target_block_id':tid,'track':track.name,'dose':float(hr.dose),'prediction':pred,'Y_true_hard_dose':float(hr.Y_true_hard_dose),'error':pred-float(hr.Y_true_hard_dose)})
            # Full candidate inversion: S4 rho grid, reps 1/20, target rank1, RF track only, M3/M4/M6.
            fi=contract['full_inversion']
            if track.name=='RF_ET_FG_PRIMARY' and task.case['case_id'] in fi['case_ids'] and task.rep in fi['replications'] and int(tr.target_rank)==int(fi['target_rank']) and local_ok:
                # Candidate-independent terms are frozen/cached once per query. This preserves
                # the exact same p-value while avoiding thousands of repeated RF predictions.
                cal_res5=_own_residuals(track,task.case,calb)
                sy0=np.r_[calb.Y_observed_at_A.to_numpy(float),0.0]
                R0,T0,J0=_m6_matrices(track,task.case,bdf,source_A,sy0,dose,h,audited,observational)
                grid=np.linspace(float(contract['candidate_domain'][0]),float(contract['candidate_domain'][1]),int(fi['grid_points']))
                p3grid,p4grid=eval_m3_m4_grid(cal_res5,grid,center,P,track.power,six_d)
                p6grid=eval_m6_grid(R0,T0,J0,grid,P,track.power)
                for method in fi['methods']:
                    def evaluator(y):
                        yy=float(y);sr=np.r_[cal_res5,yy-center];m3=eval_m3(sr,P,track.power)
                        if method=='M3':return m3['pvalue']
                        if method=='M4':return eval_m4(m3,sr,six_d)['pvalue']
                        RR=R0.copy();RR[:,5]+=yy
                        return eval_m6(RR,T0,J0,P,track.power,keep_probs=False)['pvalue']
                    pg={'M3':p3grid,'M4':p4grid,'M6':p6grid}[method]
                    inv=invert_candidate(evaluator,*contract['candidate_domain'],int(fi['grid_points']),float(contract['alpha']),int(fi['boundary_bisection_iterations']),grid_pvalues=pg,boundary_abs_tol=float(contract['numerical_tolerances']['boundary_absolute_tolerance']),boundary_rel_tol=float(contract['numerical_tolerances']['boundary_relative_tolerance']));y=float(ttruth['Y_observed_A_star']);inside=any(a-1e-12<=y<=b+1e-12 for a,b in inv['components'])
                    # The raw numerical set is always archived, but publication efficiency uses it only when the
                    # frozen Stage3F operational gate returns the query. These gates are candidate-outcome blind.
                    inv_n3=float(nd.get('n3_diagnostic_lower_bound',np.nan)) if method=='M6' else np.nan
                    inv_gcount=len(safe)
                    inv_code=_refusal(method,audited,sdiag,inv_gcount,inv_n3,True,contract)
                    inv_rows.append({'case_id':task.case['case_id'],'spatial_rho':float(task.case['spatial_rho']),'replication':task.rep,'MineID':str(tr.MineID),'target_rank':int(tr.target_rank),'target_block_id':tid,'track':track.name,'method':method,'truth_y':y,'covered_raw_set':inside,'operational_return':bool(inv_code==''),'refusal_code':inv_code,**{k:(json_components(inv[k]) if k=='components' else v) for k,v in inv.items() if k!='components'},'components_json':json_components(inv['components']),'hull_wis':interval_score(inv['hull_lower'],inv['hull_upper'],y,float(contract['alpha'])) if inv['component_count'] else np.nan})
    # Exact structural full-graph vs sparse-m64 audit uses oracle nuisance/truth only.
    exact_rows.extend(evaluate_exact_audit(task,contract,g,cal_idx))
    endpoint_rows.extend(evaluate_endpoint_audit(task,contract,tracks,g,cal,cal_idx))
    return {'query_rows':target_rows,'exact_rows':exact_rows,'inversion_rows':inv_rows,'endpoint_rows':endpoint_rows,'dose_rows':dose_rows,'fit_rows':fit_rows,'spatial_rows':spatial_rows,'runtime_row':{'case_id':task.case['case_id'],'scenario_id':task.case['scenario_id'],'replication':task.rep,'seconds':time.time()-started,'computational_failure':False}}

def json_components(c):
    import json
    return json.dumps([[float(a),float(b)] for a,b in c],separators=(',',':'))

def _exact_full_precision_block(task,g,block_idx,rho,sigma):
    # GMRF exact graph-local factor for block + realized boundary is evaluated directly in exact evaluator; this helper returns local m64 precision only.
    pass

def _exact_spatial_logs(assign,task,g,block_idx,outside_res,rho,sigma,power,include_transform_jacobian=False):
    # Exact G2 pairwise GMRF factor: diagonal block terms + every graph edge touching B; far-field terms cancel.
    B=list(map(int,block_idx));bpos={n:i for i,n in enumerate(B)};E=np.asarray(assign,float)
    if power!=1.: Z=np.sign(E)*np.abs(E)**(1./power); zout=np.sign(outside_res)*np.abs(outside_res)**(1./power)
    else: Z=E;zout=np.asarray(outside_res,float)
    log=-.5*np.sum(Z*Z,axis=1)/(sigma*sigma)
    # For the full M6 joint-payload law under the transformed-GMRF DGP, the
    # residual-density transform |dz/de| is orbit-varying and must be included.
    # M3 follows the separately frozen D3 response-orbit comparator contract,
    # which uses the registered residual graph factor without this extra factor.
    if include_transform_jacobian and abs(float(power)-1.0)>1e-15:
        ae=np.abs(E); singular=(ae<1e-14).any(axis=1); ae=np.maximum(ae,1e-14)
        log+=np.sum((1./float(power)-1.)*np.log(ae)-math.log(float(power)),axis=1)
        # Values within the registered 1e-14 singular tolerance are clipped only
        # for density evaluation; their states are conservatively included in the tail below.
    bset=set(B); local_edges=set()
    # Only edges touching the six-slot block can vary across the orbit. Build
    # this exact graph boundary from adjacency in O(sum degree(B)), never scan
    # the entire ~100k-edge mine graph for every candidate/audit.
    for a in B:
        for b in g.adjacency[a]: local_edges.add((a,b) if a<b else (b,a))
    for a,b in sorted(local_edges):
        if g.degrees[a]<=0 or g.degrees[b]<=0:continue
        va=Z[:,bpos[a]] if a in bpos else zout[a];vb=Z[:,bpos[b]] if b in bpos else zout[b];log+=float(rho)/(sigma*sigma*math.sqrt(g.degrees[a]*g.degrees[b]))*va*vb
    return log

def _canonical_probability(pv:float,tol:float=1.0e-12)->float:
    v=float(pv);require(np.isfinite(v) and v>=-float(tol) and v<=1.0+float(tol),f'Probability outside roundoff tolerance: {v:.17g}');return float(np.clip(v,0.0,1.0))

def evaluate_exact_audit(task,contract,g,cal_idx):
    from .orbit import PERMS,normalize_logweights
    rows=[];emap=task.exact_map.set_index('target_block_id');idx={b:i for i,b in enumerate(task.units.block_id.astype(str))}
    # SOFTWARE PATCH v1.0.1 (scientific contract unchanged): the frozen exact-size6
    # registry was constructed on the full frozen calibration-role frame. S8_SMALL_CAL
    # subsamples calibration only for PRIMARY operational inference. The separate
    # structural exact-vs-sparse audit must therefore use the full frozen calibration
    # frame rather than inheriting the case-specific primary subsample.
    exact_cal_idx=np.flatnonzero(task.units.benchmark_role.astype(str).to_numpy()=='calibration').astype(int)
    exact_cal_set=set(map(int,exact_cal_idx))
    require(len(exact_cal_idx)>=5,'Frozen exact-audit calibration-role frame has fewer than five slots')
    for er in task.exact_registry.itertuples(index=False):
        if not bool(er.exact_audit_available):continue
        tid=str(er.exact_audit_target_block_id);require(tid in emap.index,'Frozen exact target not in exact-size6 map');m=emap.loc[tid];calids=[str(m[f'calibration_block_{k}']) for k in range(1,6)];block=np.array([idx[x] for x in calids]+[idx[tid]],int);extruth=__import__('geodose_stage5b.data',fromlist=['exact_audit_target_truth']).exact_audit_target_truth(task,pd.Series({'exact_audit_target_block_id':tid,'MineID':str(er.MineID)}));astar=float(extruth['A_star']);dose=extruth['requested_dose'];h=extruth['bandwidth'];observational=task.case['primary_target_mode']=='observational';audited=bool(observational or task.case['endpoint_structure']!='interior_only' or (not pd.isna(dose) and float(dose) not in (0.,1.)))
        bdf=task.units.iloc[block].copy().reset_index(drop=True);source_A=np.r_[bdf.iloc[:5].A.to_numpy(float),astar];source_Y=np.r_[bdf.iloc[:5].Y_true_at_A.to_numpy(float),float(extruth['Y_true_A_star'])]
        # Oracle M3 source residuals are latent factual residual truth plus exact target residual.
        center=float(oracle_mean(task.case,bdf.iloc[[5]],np.array([astar]))[0]);src_res=np.r_[bdf.iloc[:5].residual_truth.to_numpy(float),float(extruth['Y_true_A_star'])-center];assign=src_res[PERMS];lwfull=_exact_spatial_logs(assign,task,g,block,task.units.residual_truth.to_numpy(float),float(task.case['spatial_rho']),.04,1.5 if task.case['residual_law']=='transformed_gmrf_power_1_5' else 1.0);pfull=normalize_logweights(lwfull);score=np.abs(assign[:,5]);obs=abs(src_res[5]);m3_full_p=_canonical_probability(float(pfull[upper_tail_mask(score,obs)].sum()));src_full=np.bincount(PERMS[:,5],weights=pfull,minlength=6)
        # Sparse m64 structural reference, oracle rho.
        bcal=block[:5];require(set(map(int,bcal)).issubset(exact_cal_set),'Frozen exact-size6 calibration slot is not in the full frozen calibration-role frame');P64,_,_,_=six_precision(block[5],bcal,exact_cal_idx,task.units,g,float(task.case['spatial_rho']),int(contract['spatial_route']['m']));P64=P64/(.04*.04);m3s=eval_m3(src_res,P64,1.5 if task.case['residual_law']=='transformed_gmrf_power_1_5' else 1.0)
        # Oracle source dose ratios.
        if observational:dlog=np.zeros(6)
        else:
            q=_q_values(source_A,dose,h,audited,False);gd=np.r_[oracle_density(bdf.iloc[:5]),oracle_density(bdf.iloc[[5]],a=np.array([astar]))];dlog=np.full(6,-np.inf);ok=(q>0)&(gd>0);dlog[ok]=np.log(q[ok])-np.log(gd[ok])
        # M4 full and sparse.
        logs=np.full(6,-np.inf);pos=src_full>0;logs[pos]=np.log(src_full[pos])+dlog[pos];q4full=normalize_logweights(logs);m4full=_canonical_probability(float(q4full[upper_tail_mask(np.abs(src_res),obs)].sum()));m4s=eval_m4(m3s,src_res,dlog)
        # M6 full: direct assignment treatment/Jacobian/residual factors.
        dummy=Track('ORACLE','oracle',None,None,float(task.case['spatial_rho']),.04,1.5 if task.case['residual_law']=='transformed_gmrf_power_1_5' else 1.,'oracle','oracle',True);R,T,J=_m6_matrices(dummy,task.case,bdf,source_A,source_Y,dose,h,audited,observational);rassign=R[np.arange(6)[None,:],PERMS];tl=T[np.arange(6)[None,:],PERMS];jl=J[np.arange(6)[None,:],PERMS];lw6=np.sum(tl+jl,axis=1)+_exact_spatial_logs(rassign,task,g,block,task.units.residual_truth.to_numpy(float),float(task.case['spatial_rho']),.04,dummy.power,include_transform_jacobian=True);p6=normalize_logweights(lw6);sing6=((np.abs(rassign)<1e-14).any(axis=1) if abs(dummy.power-1.0)>1e-15 else np.zeros(len(rassign),bool));m6full=_canonical_probability(float(p6[(upper_tail_mask(np.abs(rassign[:,5]),abs(R[5,5]))|sing6)].sum()));m6s=eval_m6(R,T,J,P64,dummy.power)
        refs={'M3':(m3_full_p,pfull,m3s['pvalue'],m3s['orbit_probs']),'M4':(m4full,q4full,m4s['pvalue'],m4s['source_probs']),'M6':(m6full,p6,m6s['pvalue'],m6s['orbit_probs'])}
        for method,(pf,prob_f,ps,prob_s) in refs.items():
            kl=orbit_kl(prob_f,prob_s) if len(prob_f)==len(prob_s) else np.nan;tv=min(1.,math.sqrt(kl/2.)) if np.isfinite(kl) else 1.
            for ref,pv in [('full_graph_exact',pf),('sparse_m64',ps)]:rows.append({'case_id':task.case['case_id'],'scenario_id':task.case['scenario_id'],'replication':task.rep,'MineID':str(er.MineID),'target_block_id':tid,'method':method,'reference':ref,'pvalue':float(pv),'covered':bool(pv>contract['alpha']),'full_vs_sparse_orbit_kl':kl,'full_vs_sparse_pinsker_tv':tv,'pvalue_absolute_difference':abs(float(pf)-float(ps)),'exact_G2_G3_status':'full_graph_structural_oracle' if ref=='full_graph_exact' else 'registered_sparse_m64_audit','computational_failure':False})
    return rows

def evaluate_endpoint_audit(task,contract,tracks,g,cal,cal_idx):
    if task.case['case_id'] not in set(contract['endpoint_audit']['case_ids']):return []
    rows=[];idx={b:i for i,b in enumerate(task.units.block_id.astype(str))};trg=task.targets[task.targets.target_rank==1].copy()
    for track in [t for t in tracks if t.name in set(contract['endpoint_audit']['tracks'])]:
        cal_score=np.abs(_own_residuals(track,task.case,cal))
        for tr in trg.itertuples(index=False):
            ti=idx[str(tr.target_block_id)];tdf=task.units.iloc[[ti]].copy().reset_index(drop=True)
            for dose in contract['endpoint_audit']['doses']:
                audited=task.case['endpoint_structure']!='interior_only';expected_refusal=not audited
                if audited:
                    astar=float(dose);mu=float(conditional_mean(task.case,np.array([task.truth_arrays['base'][ti]]),np.array([task.truth_arrays['xpv'][ti]]),np.array([astar]))[0]);ytrue=mu+float(task.truth_arrays['e'][ti]);me=float(measurement_error(task.case,np.array([astar]),np.array([task.units.iloc[ti].ue_proxy]),np.array([task.truth_arrays['substrate'][ti]]),np.array([task.truth_arrays['measurement_z'][ti]]))[0]);yobs=ytrue+me
                else:astar=float(dose);yobs=np.nan;ytrue=np.nan
                tobj=_row_with_unit({'A_star':astar},tdf);dlog,tlog=_dose_logs(cal,tobj,dose,None,audited,track,False) if audited else (np.full(len(cal),-np.inf),-np.inf);sdiag=_weight_diagnostics(dlog,tlog);center=float(_track_center(track,task.case,tdf,np.array([astar]))[0])
                safe=deterministic_graph_safe(ti,cal_idx,g);safe_set=set(map(int,safe));safe_mask=np.array([x in safe_set for x in cal_idx]);m2=_wcp(center,cal_score,dlog,tlog,contract['alpha']) if audited else {'raw_status':'refused','lower':np.nan,'upper':np.nan,'quantile':np.nan};m5=_wcp(center,cal_score[safe_mask],dlog[safe_mask],tlog,contract['alpha']) if audited and safe_mask.any() else {'raw_status':'refused','lower':np.nan,'upper':np.nan,'quantile':np.nan}
                local_ok=False;local_error_type='';local_error_message='';nd={'m64_count':0,'n3_diagnostic_lower_bound':np.nan}
                if audited:
                    try:bcal=nearest_calibration_block(ti,cal_idx,task.units,5);P,nd=localization_diagnostic(ti,bcal,cal_idx,task.units,g,track.rho,64,32);P=P/(track.sigma*track.sigma);bdf=task.units.iloc[np.r_[bcal,ti]].copy().reset_index(drop=True);src=np.r_[_own_residuals(track,task.case,bdf.iloc[:5]),yobs-center];m3=eval_m3(src,P,track.power);calpos=[np.flatnonzero(cal.block_id.astype(str).to_numpy()==str(x))[0] for x in bdf.iloc[:5].block_id];sixd=np.r_[dlog[calpos],tlog];m4=eval_m4(m3,src,sixd);A=np.r_[bdf.iloc[:5].A.to_numpy(float),astar];Y=np.r_[bdf.iloc[:5].Y_observed_at_A.to_numpy(float),yobs];R,T,J=_m6_matrices(track,task.case,bdf,A,Y,dose,None,True,False);m6=eval_m6(R,T,J,P,track.power);local_ok=True
                    except Exception as exc:local_ok=False;local_error_type=type(exc).__name__;local_error_message=str(exc)[:500]
                for method in METHODS:
                    if not audited:raw_ok=False;cov=False;gc=0;n3=np.nan
                    elif method=='M2':raw_ok=m2['raw_status']!='refused';cov=raw_ok and m2['lower']<=yobs<=m2['upper'];gc=0;n3=np.nan
                    elif method=='M5':raw_ok=m5['raw_status']!='refused';cov=raw_ok and m5['lower']<=yobs<=m5['upper'];gc=len(safe);n3=np.nan
                    else:rr={'M3':m3,'M4':m4,'M6':m6}.get(method) if local_ok else None;raw_ok=rr is not None;cov=bool(raw_ok and rr['pvalue']>contract['alpha']);gc=len(safe);n3=(float(nd.get('n3_diagnostic_lower_bound',np.nan)) if method=='M6' else np.nan)
                    gate_sd=(_weight_diagnostics(dlog[safe_mask],tlog) if method=='M5' and safe_mask.any() else sdiag);code='R02_ENDPOINT_NOT_AUDITED' if not audited else _refusal(method,True,gate_sd,gc,n3,raw_ok,contract);rows.append({'case_id':task.case['case_id'],'replication':task.rep,'MineID':str(tr.MineID),'target_rank':int(tr.target_rank),'target_block_id':str(tr.target_block_id),'track':track.name,'method':method,'endpoint_dose':float(dose),'endpoint_audited':audited,'expected_refusal':expected_refusal,'operational_return':code=='','refusal_code':code,'covered_observed':bool(code=='' and cov),'Y_observed_A_star':yobs,'computational_failure':bool(code=='R10_COMPUTATIONAL_FAILURE'),'local_error_type':(local_error_type if method in {'M3','M4','M6'} else ''),'local_error_message':(local_error_message if method in {'M3','M4','M6'} else '')})
    return rows

def evaluate_lomo(task,contract,lomo_registry):
    if task.case['case_id'] not in set(contract['lomo']['case_ids']) or task.rep not in set(contract['lomo']['replications']):return []
    rows=[];features=task.allowed_predictors;g=graph_info(task.units,task.working_edges);idx={b:i for i,b in enumerate(task.units.block_id.astype(str))};truth_lookup=task.target_truth.set_index(['MineID','target_rank'])
    for fold,fr in lomo_registry.groupby('fold_id',sort=True):
        roles={str(r.MineID):str(r.role) for r in fr.itertuples(index=False)};test_mines=[m for m,r in roles.items() if r=='test_target'];cal_mines=[m for m,r in roles.items() if r=='calibration'];train_mines=[m for m,r in roles.items() if r=='nuisance_training'];require(len(test_mines)==1 and len(cal_mines)==1 and len(train_mines)==2,'LOMO registry malformed');tm=test_mines[0];cm=cal_mines[0]
        train=task.units[task.units.MineID.astype(str).isin(train_mines)].copy().reset_index(drop=True);cal=task.units[task.units.MineID.astype(str)==cm].copy().reset_index(drop=True);require(len(train)>20 and len(cal)>20,'LOMO training/calibration too small')
        prop=MixedPropensity(features,stable_seed('Stage5B-LOMO-prop',task.case['case_id'],task.rep,fold)).fit(train);model=OutcomeModel('rf',features,stable_seed('Stage5B-LOMO-rf',task.case['case_id'],task.rep,fold),contract).fit(train);scores=np.abs(cal.Y_observed_at_A.to_numpy(float)-model.predict(cal));cal_idx=np.array([idx[b] for b in cal.block_id.astype(str)],int)
        targets=task.targets[task.targets.MineID.astype(str)==tm]
        require(len(targets)==5,'LOMO held-out mine must have five frozen targets')
        for tr in targets.itertuples(index=False):
            tt=truth_lookup.loc[(tm,int(tr.target_rank))].to_dict();ti=idx[str(tr.target_block_id)];tdf=task.units.iloc[[ti]].copy().reset_index(drop=True);tobj=_row_with_unit({'A_star':float(tt['A_star'])},tdf);observational=task.case['primary_target_mode']=='observational';dose=float(tt['requested_dose']) if not pd.isna(tt.get('requested_dose')) else float(task.units.iloc[ti].A);h=tt.get('bandwidth');audited=bool(observational or task.case['endpoint_structure']!='interior_only' or dose not in (0.,1.));dummy=Track('LOMO_RF','estimated',model,prop,0.,model.scale_,1.,model.fit_id,prop.fit_id,False);dlog,tlog=_dose_logs(cal,tobj,dose,h,audited,dummy,observational);sd=_weight_diagnostics(dlog,tlog);center=float(model.predict(tdf,np.array([float(tt['A_star'])]))[0]);m2=_wcp(center,scores,dlog,tlog,contract['alpha']);safe=deterministic_graph_safe(ti,cal_idx,g);ss=set(map(int,safe));mask=np.array([x in ss for x in cal_idx]);m5=_wcp(center,scores[mask],dlog[mask],tlog,contract['alpha']) if mask.any() else {'raw_status':'refused','lower':np.nan,'upper':np.nan,'quantile':np.nan};y=float(tt['Y_observed_A_star'])
            for method in METHODS:
                if method in {'M3','M4','M6'}:code='R18_LOMO_DISCONNECTED_LOCAL_ORBIT';oper=False;cov=False;lo=hi=np.nan
                elif method=='M2':raw=m2;code=_refusal(method,audited,sd,len(cal),np.nan,raw['raw_status']!='refused',contract);oper=code=='';lo,hi=raw['lower'],raw['upper'];cov=bool(oper and lo<=y<=hi)
                else:raw=m5;sds=_weight_diagnostics(dlog[mask],tlog) if mask.any() else {'support_status':'no_positive','support_ess':0.,'support_max_weight':1.};code=_refusal(method,audited,sds,len(safe),np.nan,raw['raw_status']!='refused',contract);oper=code=='';lo,hi=raw['lower'],raw['upper'];cov=bool(oper and lo<=y<=hi)
                rows.append({'case_id':task.case['case_id'],'scenario_id':task.case['scenario_id'],'replication':task.rep,'fold_id':fold,'heldout_MineID':tm,'calibration_MineID':cm,'target_rank':int(tr.target_rank),'target_block_id':str(tr.target_block_id),'method':method,'operational_return':oper,'refusal_code':code,'covered':cov,'truth_y':y,'lower':lo,'upper':hi,'width':float(hi-lo) if np.isfinite(lo) and np.isfinite(hi) else (np.inf if np.isneginf(lo) and np.isposinf(hi) else np.nan),'computational_failure':False})
    return rows
