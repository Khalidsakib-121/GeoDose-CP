from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from geodose_stage6a.common import require,read_json,conservative_split_quantile,interval_bounds,stable_seed
from geodose_stage6a.authority import verify_authorities
from geodose_stage6a.data import build_90m,build_180m,validate_counts
from geodose_stage6a.evaluation import fit_model_and_spatial
from geodose_stage5b.orbit import eval_m3,eval_m4
P=[]
def ck(n,c,d=''):require(c,f'FAIL {n}: {d}');P.append(n);print('PASS',n)
def main():
    c=read_json(ROOT/'configs/stage6a_contract.json');checks=verify_authorities(ROOT,c);ck('authorities',len(checks)>=15,len(checks))
    cb=c['real_application_claim_boundary'];ck('no_authentic_treatment',cb['authentic_longitudinal_continuous_treatment_available'] is False);ck('snapshot_not_treatment',cb['mapped_rehabilitation_fraction_is_causal_treatment'] is False);ck('no_counterfactual_claim',cb['real_intervals_are_counterfactual_potential_outcome_intervals'] is False);ck('M6_disabled',cb['M6_treatment_route_operational'] is False);ck('bare_not_fabricated',cb['bare_ground_confirmatory_outcome_available_in_frozen_stage2b'] is False)
    ck('alpha',c['alpha']==.1);ck('domain',c['candidate_domain']==[-1.,1.]);ck('m64',c['spatial_route']['m']==64);ck('m32',c['spatial_route']['sensitivity_m']==32);ck('thresholds',c['support_thresholds']=={'minimum_ess':3.0,'maximum_normalized_weight':0.5,'minimum_graph_safe_count':20,'n3_diagnostic_lower_bound_must_be_finite_and_positive':True})
    ck('predictor_count',len(c['predictors'])==30);ck('no_forbidden',not set(c['predictors'])&set(c['forbidden_predictors']));ck('no_2025',not any('2025' in x for x in c['predictors']));ck('snapshot_not_predictor','mapped_rehabilitation_fraction' not in c['predictors']);ck('outcome_not_predictor',c['outcome']['column'] not in c['predictors'])
    q=conservative_split_quantile(np.arange(1,100),.1);ck('split_quantile',q==90,q);rawlo,rawhi,lo,hi,w=interval_bounds(.9,.5,[-1,1]);ck('physical_intersection',lo==.4 and hi==1 and abs(w-.6)<1e-12)
    e=np.array([-.4,-.2,.1,.3,.5,.15]);m3=eval_m3(e,np.eye(6),1.);m4=eval_m4(m3,e,np.zeros(6));ck('M4_zero_ratio_reduction',abs(m3['pvalue']-m4['pvalue'])<=1e-12)
    print('[DATA] Replaying complete frozen NSW substrate...')
    s90,e90,a90,sel=build_90m(ROOT,ROOT/'_STAGE6A_WORK',c);s90=s90.reset_index(drop=True);s180,e180,a180=build_180m(s90,c);s180=s180.reset_index(drop=True);validate_counts(s90,s180,c)
    ck('s90_counts',len(s90)==19950 and int(s90.real_baseline_eligible.sum())==17632 and int(s90.primary_component.sum())==13683 and int((s90.benchmark_role=='test_target').sum())==2736)
    mine=s90.groupby('MineN').real_baseline_eligible.sum().astype(int).to_dict();ck('mine_counts',mine=={'Bulga Complex':3403,'Hunter Valley Operations':7837,'Mt Arthur Coal':6392},mine)
    ck('outcome_physical',s90.loc[s90.real_baseline_eligible,c['outcome']['column']].between(-1,1).all());ck('predictors_finite',np.isfinite(s90.loc[s90.real_baseline_eligible,c['predictors']].to_numpy(float)).all());ck('strict_quality_nonvacuous',s90.loc[s90.real_baseline_eligible,'strict_quality_ue20'].mean()>.9)
    ck('s180_counts',len(s180)==5379 and int(s180.real_baseline_eligible.sum())==4352 and int(s180.primary_component.sum())==3506 and int((s180.benchmark_role=='test_target').sum())==701)
    ck('s180_child_rule',(s180.real_baseline_eligible.astype(bool)==((s180.eligible_child_count==s180.available_child_count)&(s180.available_child_count>0))).all())
    # Regression for inactive rows: no imputation/prediction.
    _,u,pred,res,g,fit,imp=fit_model_and_spatial(s90,e90,'RF_PRIMARY','90m',c);active=u.benchmark_role.isin(['nuisance_training','support_audit','calibration','test_target']).to_numpy(bool)
    ck('active_pred_finite',np.isfinite(pred[active]).all() and np.isfinite(res[active]).all());ck('inactive_quarantined',np.isnan(pred[~active]).all() and np.isnan(res[~active]).all());ck('prediction_rows',fit['prediction_rows']==int(active.sum()))
    # Forbidden spatial information adjacencies.
    forbidden={frozenset(('nuisance_training','support_audit')),frozenset(('nuisance_training','calibration')),frozenset(('nuisance_training','test_target')),frozenset(('support_audit','calibration')),frozenset(('support_audit','test_target'))}
    def bad(s,e):
        role=dict(zip(s.block_id.astype(str),s.benchmark_role.astype(str)));return sum(frozenset((role.get(str(a),''),role.get(str(b),''))) in forbidden for a,b in zip(e.source_block_id,e.target_block_id))
    ck('forbidden90',bad(s90,e90)==0,bad(s90,e90));ck('forbidden180',bad(s180,e180)==0,bad(s180,e180))
    cfg=json.loads((ROOT/'_STAGE6A_WORK/frozen_inputs/stage2b/stage2b_config.json').read_text());ck('selection_outcome_blind',cfg['selection_uses_pv_outcome'] is False);ck('UE20_frozen',20.0 in cfg['high_ue_sensitivity_thresholds'])
    src='\n'.join(p.read_text(encoding='utf-8',errors='ignore') for p in (ROOT/'src').rglob('*.py'));ck('no_network','requests.' not in src and 'urllib' not in src and 'http://' not in src and 'https://' not in src);ck('no_autotune','GridSearchCV' not in src and 'RandomizedSearchCV' not in src)
    print(f'STAGE6A UNIT/AUTHORITY TESTS PASSED ({len(P)}/{len(P)})')
if __name__=='__main__':main()
