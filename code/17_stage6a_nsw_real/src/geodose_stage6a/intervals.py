from __future__ import annotations
import json
import numpy as np
from .common import require,conservative_split_quantile,interval_bounds,coverage
from geodose_stage5b.spatial import deterministic_graph_safe,nearest_calibration_block,localization_diagnostic
from geodose_stage5b.orbit import eval_m3,eval_m4,eval_m3_m4_grid,invert_candidate

def m1_interval(center,scores,alpha,domain):
    q=conservative_split_quantile(scores,alpha);rawlo,rawhi,lo,hi,w=interval_bounds(center,q,domain)
    return {'quantile':q,'raw_lower':rawlo,'raw_upper':rawhi,'lower':lo,'upper':hi,'width':w,'calibration_count':len(scores)}

def m5_interval(center,target_idx,cal_idx,residual,g,alpha,domain,thresholds):
    safe=deterministic_graph_safe(int(target_idx),np.asarray(cal_idx,int),g);n=len(safe)
    ess=float(n);maxw=float(1/n) if n else np.inf
    if n<int(thresholds['minimum_graph_safe_count']):return {'returned':False,'refusal_code':'R12_GRAPH_SAFE_COUNT_BELOW_FROZEN_THRESHOLD','graph_safe_count':n,'ess':ess,'max_weight':maxw}
    if ess<float(thresholds['minimum_ess']):return {'returned':False,'refusal_code':'R05_ESS_BELOW_FROZEN_THRESHOLD','graph_safe_count':n,'ess':ess,'max_weight':maxw}
    if maxw>float(thresholds['maximum_normalized_weight']):return {'returned':False,'refusal_code':'R06_MAX_WEIGHT_ABOVE_FROZEN_THRESHOLD','graph_safe_count':n,'ess':ess,'max_weight':maxw}
    scores=np.abs(np.asarray(residual,float)[safe]);q=conservative_split_quantile(scores,alpha);rawlo,rawhi,lo,hi,w=interval_bounds(center,q,domain)
    return {'returned':True,'refusal_code':'','graph_safe_count':n,'ess':ess,'max_weight':maxw,'quantile':q,'raw_lower':rawlo,'raw_upper':rawhi,'lower':lo,'upper':hi,'width':w,'calibration_count':n}

def m3_membership(center,y,target_idx,cal_idx,residual,units,g,rho,sigma,contract,do_inversion=False):
    th=contract['support_thresholds'];alpha=float(contract['alpha']);domain=contract['candidate_domain']
    try:
        safe=deterministic_graph_safe(int(target_idx),np.asarray(cal_idx,int),g);gcount=len(safe)
        if gcount<int(th['minimum_graph_safe_count']):return {'returned':False,'refusal_code':'R12_GRAPH_SAFE_COUNT_BELOW_FROZEN_THRESHOLD','graph_safe_count':gcount}
        bcal=nearest_calibration_block(int(target_idx),np.asarray(cal_idx,int),units,n_slots=int(contract['spatial_route']['local_calibration_slots']))
        P,diag=localization_diagnostic(int(target_idx),bcal,np.asarray(cal_idx,int),units,g,float(rho),m=int(contract['spatial_route']['m']),m_small=int(contract['spatial_route']['sensitivity_m']))
        P=P/(float(sigma)**2)
        src=np.r_[np.asarray(residual,float)[bcal],float(y)-float(center)]
        m3=eval_m3(src,P,1.0);m4=eval_m4(m3,src,np.zeros(6))
        out={'returned':True,'refusal_code':'','covered':bool(float(m3['pvalue'])>alpha),'pvalue':float(m3['pvalue']),'graph_safe_count':gcount,'rho':float(rho),'sigma':float(sigma),
             'delta_sparse_localization_kl':float(diag['delta_sparse_localization_kl']),'delta_sparse_pinsker_tv':float(diag['delta_sparse_pinsker_tv']),
             'n3_diagnostic_lower_bound':float(diag['n3_diagnostic_lower_bound']),'m4_zero_ratio_pvalue':float(m4['pvalue']),'m4_m3_abs_diff':abs(float(m4['pvalue'])-float(m3['pvalue'])),'interval_inverted':False}
        if do_inversion:
            grid=np.linspace(float(domain[0]),float(domain[1]),int(contract['candidate_inversion']['grid_points']))
            p3,_=eval_m3_m4_grid(src[:5],grid,float(center),P,1.0,np.zeros(6))
            def ev(v):
                ss=np.r_[src[:5],float(v)-float(center)];return float(eval_m3(ss,P,1.0,keep_probs=False)['pvalue'])
            inv=invert_candidate(ev,lo=float(domain[0]),hi=float(domain[1]),grid_points=len(grid),alpha=alpha,bisect_iter=int(contract['candidate_inversion']['bisection_iterations']),
                                 grid_pvalues=p3,boundary_abs_tol=float(contract['candidate_inversion']['boundary_abs_tol']),boundary_rel_tol=float(contract['candidate_inversion']['boundary_rel_tol']))
            out.update({'interval_inverted':True,'component_count':int(inv['component_count']),'lower':float(inv['hull_lower']) if np.isfinite(inv['hull_lower']) else np.nan,
                        'upper':float(inv['hull_upper']) if np.isfinite(inv['hull_upper']) else np.nan,'width':float(inv['hull_width']),'raw_set_width':float(inv['raw_set_width']),
                        'hull_inflation':float(inv['hull_inflation']),'left_domain_truncated':bool(inv['left_domain_truncated']),'right_domain_truncated':bool(inv['right_domain_truncated']),
                        'components_json':json.dumps(inv['components'],separators=(',',':'))})
        return out
    except Exception as e:
        return {'returned':False,'refusal_code':'R10_COMPUTATIONAL_FAILURE','graph_safe_count':np.nan,'error_type':type(e).__name__,'error_message':str(e),'interval_inverted':False}
