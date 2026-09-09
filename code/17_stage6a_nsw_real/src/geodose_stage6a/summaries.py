from __future__ import annotations
import numpy as np,pandas as pd

def method_applicability(contract):
    rows=[
      ('M1','Standard split conformal prediction',True,'Observed-product predictive reference; pooled buffered spatial split.'),
      ('M2','Dose-only weighted conformal prediction',False,'Not applicable: authentic longitudinal rehabilitation dose is not constructible.'),
      ('M3','Spatial-only generalized conformal prediction',True,'Observed-product spatial uncertainty demonstration; treatment held absent.'),
      ('M4','Naive treatment × spatial product',False,'Operational treatment factor unavailable; zero-treatment-ratio reduction to M3 audited structurally only.'),
      ('M5','Conservative graph-safe conformal fallback',True,'Observed-product graph-safe fallback using frozen support gates.'),
      ('M6','Full GeoDose-CP',False,'Not operational in real NSW snapshot because authentic longitudinal treatment is unavailable; causal validity established elsewhere.')
    ]
    return pd.DataFrame(rows,columns=['method','description','operational_in_real_nsw','real_nsw_status'])

def sample_summary(s,scale):
    rows=[]
    for (mid,mn),g in s.groupby(['MineID','MineN'],sort=True):
        rows.append({'scale':scale,'MineID':mid,'MineN':mn,'inventory_rows':len(g),'eligible_rows':int(g.real_baseline_eligible.sum()),'primary_component_rows':int(g.primary_component.sum()),
                     'nuisance_rows':int((g.benchmark_role=='nuisance_training').sum()),'support_rows':int((g.benchmark_role=='support_audit').sum()),
                     'calibration_rows':int((g.benchmark_role=='calibration').sum()),'test_rows':int((g.benchmark_role=='test_target').sum()),
                     'buffer_rows':int((g.benchmark_role=='buffer_excluded').sum()),'secondary_rows':int((g.benchmark_role=='secondary_support_stratum').sum()),
                     'baseline_ineligible_rows':int((g.benchmark_role=='baseline_ineligible').sum()),
                     'strict_ue20_retention_among_eligible':float(g.loc[g.real_baseline_eligible.astype(bool),'strict_quality_ue20'].mean()),
                     'outcome_min':float(g.loc[g.real_baseline_eligible.astype(bool),'delta_pv_2024_2025_fraction'].min()),
                     'outcome_max':float(g.loc[g.real_baseline_eligible.astype(bool),'delta_pv_2024_2025_fraction'].max())})
    return pd.DataFrame(rows)

def _summ(g):
    returned=g.returned.astype(bool);r=g[returned];n=len(g);nr=len(r)
    cov=float(r.covered_observed_product.astype(float).mean()) if nr else np.nan
    # M3 interval width is only on deterministic pre-outcome inversion subset; for M1/M5 it is full.
    iw=r[r.interval_inverted.astype(bool)&r.width.notna()]
    return pd.Series({'queries':n,'returned':nr,'return_rate':nr/n if n else np.nan,'refusal_rate':1-nr/n if n else np.nan,
                      'selective_observed_product_coverage':cov,'mean_interval_width':float(iw.width.mean()) if len(iw) else np.nan,
                      'median_interval_width':float(iw.width.median()) if len(iw) else np.nan,'interval_width_sample_n':len(iw),
                      'mean_graph_safe_count':float(r.graph_safe_count.mean()) if nr and r.graph_safe_count.notna().any() else np.nan,
                      'mean_n3_diagnostic_lower_bound':float(r.n3_diagnostic_lower_bound.mean()) if nr and r.n3_diagnostic_lower_bound.notna().any() else np.nan,
                      'mean_sparse_pinsker_tv':float(r.delta_sparse_pinsker_tv.mean()) if nr and r.delta_sparse_pinsker_tv.notna().any() else np.nan})
def interval_summary(q):
    mine=q.groupby(['scale','track','method','MineID','MineN'],dropna=False,sort=True).apply(_summ,include_groups=False).reset_index()
    pooled=q.groupby(['scale','track','method'],dropna=False,sort=True).apply(_summ,include_groups=False).reset_index()
    return mine,pooled

def refusal_summary(q):
    z=q[~q.returned.astype(bool)].groupby(['scale','track','method','refusal_code'],dropna=False).size().rename('count').reset_index()
    totals=q.groupby(['scale','track','method']).size().rename('queries').reset_index()
    z=z.merge(totals,on=['scale','track','method'],how='left');z['fraction_of_queries']=z['count']/z['queries'];return z

def quality_sensitivity(q):
    rows=[]
    for keys,g in q[q.method.isin(['M1','M3','M5'])].groupby(['scale','track','method'],sort=True):
        for label,mask in [('all_frozen_eligible',np.ones(len(g),bool)),('strict_ue20',g.strict_quality_ue20.astype(bool).to_numpy())]:
            x=g.loc[mask];r=x[x.returned.astype(bool)]
            rows.append({'scale':keys[0],'track':keys[1],'method':keys[2],'quality_frame':label,'queries':len(x),'returned':len(r),
                         'return_rate':len(r)/len(x) if len(x) else np.nan,'selective_observed_product_coverage':float(r.covered_observed_product.astype(float).mean()) if len(r) else np.nan,
                         'mean_interval_width':float(r.loc[r.interval_inverted.astype(bool)&r.width.notna(),'width'].mean()) if len(r) else np.nan})
    return pd.DataFrame(rows)

def maup_summary(q):
    x=q[q.method.isin(['M1','M3','M5'])]
    return x.groupby(['scale','track','method'],sort=True).apply(_summ,include_groups=False).reset_index()

def reduction_summary(red):
    if len(red)==0:return pd.DataFrame()
    return red.groupby(['scale','track'],sort=True).agg(rows=('block_id','size'),max_M4_M3_abs_diff=('M4_M3_abs_diff','max'),
        mean_M4_M3_abs_diff=('M4_M3_abs_diff','mean'),inversion_audit_rows=('audit_selected_for_inversion','sum')).reset_index()
