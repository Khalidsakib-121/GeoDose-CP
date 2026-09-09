from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd
from .common import require,Stage5BError,stable_seed
from minedosebench.generator import generate_case_rep,target_truth_from_registry,working_edges
from minedosebench.math import conditional_mean,measurement_error,sample_localized,DOSES

@dataclass
class TaskData:
    case:dict; rep:int; substrate:gpd.GeoDataFrame; true_edges:pd.DataFrame; working_edges:pd.DataFrame
    units:pd.DataFrame; truth_arrays:dict; targets:pd.DataFrame; target_truth:pd.DataFrame; hard_truth:pd.DataFrame
    exact_registry:pd.DataFrame; exact_map:pd.DataFrame; allowed_predictors:list[str]

class BenchmarkData:
    def __init__(self,stage5a:Path):
        self.root=Path(stage5a)
        self.s90=gpd.read_file(self.root/'minedosebench_substrate_90m.gpkg')
        self.s180=gpd.read_file(self.root/'minedosebench_substrate_180m.gpkg')
        self.e90=pd.read_csv(self.root/'minedosebench_true_graph_90m.csv')
        self.e180=pd.read_csv(self.root/'minedosebench_true_graph_180m.csv')
        self.scenarios=pd.read_csv(self.root/'minedosebench_scenario_registry.csv')
        self.seeds=pd.read_csv(self.root/'minedosebench_seed_registry.csv')
        self.targets=pd.read_csv(self.root/'minedosebench_primary_target_registry.csv')
        self.exact_targets=pd.read_csv(self.root/'minedosebench_exact_audit_target_registry.csv')
        self.exact90=pd.read_csv(self.root/'minedosebench_exact_size6_blocks_90m.csv')
        self.exact180=pd.read_csv(self.root/'minedosebench_exact_size6_blocks_180m.csv')
        self.allowed=pd.read_csv(self.root/'minedosebench_allowed_predictors.csv')
        self.design_ratio=pd.read_csv(self.root/'minedosebench_target_design_ratio_oracle_90m.csv')
        self.lomo=pd.read_csv(self.root/'minedosebench_lomo_registry.csv')
        for s in [self.s90,self.s180]:
            s['block_id']=s.block_id.astype(str); s['MineID']=s.MineID.astype(str)
            # GeoPackage round-trip stores nullable booleans as text on the accepted Windows release.
            # Normalize them explicitly before reusing the frozen Stage5A generator.
            for bc in ['primary_component','benchmark_baseline_eligible','frozen_context_support_exclusion']:
                if bc in s.columns:
                    s[bc]=s[bc].map(lambda v: (str(v).strip().lower()=='true') if pd.notna(v) else False).astype(bool)
        for e in [self.e90,self.e180]:
            e['source_block_id']=e.source_block_id.astype(str);e['target_block_id']=e.target_block_id.astype(str);e['MineID']=e.MineID.astype(str)
        self.targets.target_block_id=self.targets.target_block_id.astype(str);self.targets.MineID=self.targets.MineID.astype(str)
        self.exact_targets.exact_audit_target_block_id=self.exact_targets.exact_audit_target_block_id.fillna('').astype(str);self.exact_targets.MineID=self.exact_targets.MineID.astype(str)
        self._validate()
    def _validate(self):
        require(len(self.scenarios)==27 and len(self.seeds)==540 and len(self.targets)==13500,'Frozen Stage5A registry count mismatch')
        require(int(self.s90.benchmark_baseline_eligible.astype(bool).sum())==23618,'Frozen 90m eligible count mismatch')
        require(len(self.s90)==27042 and len(self.s180)==7295,'Frozen support inventory mismatch')
        require(set(self.allowed.timing.astype(str))=={'pretreatment'},'Non-pretreatment predictor in allowed registry')
        forbidden={'mapped_rehabilitation_fraction','pv_median_2025','benchmark_role','hidden_u_truth','Y_true_at_A','Y_observed_at_A'}
        require(not (forbidden & set(self.allowed.predictor.astype(str))),'Forbidden predictor present')
    def case(self,case_id): return self.scenarios[self.scenarios.case_id==case_id].iloc[0].to_dict()
    def seedrow(self,case_id,rep): return self.seeds[(self.seeds.case_id==case_id)&(self.seeds.replication==rep)].iloc[0].to_dict()
    def support(self,case):
        if case['support_scale']=='180m_anchored_rerun': return self.s180.copy(),self.e180.copy(),self.exact180.copy()
        return self.s90.copy(),self.e90.copy(),self.exact90.copy()
    def build_task(self,case_id:str,rep:int)->TaskData:
        c=self.case(case_id); seed=self.seedrow(case_id,rep); sub,edges,exactmap=self.support(c)
        wed=working_edges(c,sub,edges,int(seed['graph_seed']))
        units,arr=generate_case_rep(c,int(rep),sub,edges,seed)
        # Attach only frozen public pretreatment fields; roles are retained for information separation, never as predictors.
        allow=[x for x in self.allowed.predictor.astype(str) if x in sub.columns]
        extra=['centroid_x','centroid_y','spatial_axis_rank','spatial_axis_score','primary_component']
        cols=['block_id']+sorted(set(allow+extra)-{'block_id'})
        units=units.merge(sub[cols],on='block_id',how='left',validate='1:1')
        # Every used predictor must be finite for active primary units. Drop columns structurally unavailable at 180m.
        good=[]
        for p in allow:
            v=pd.to_numeric(units[p],errors='coerce')
            if v.notna().all() and np.isfinite(v.to_numpy(float)).all(): good.append(p)
        trg=self.targets[(self.targets.case_id==case_id)&(self.targets.replication==rep)].copy().reset_index(drop=True)
        require(len(trg)==25,f'Expected 25 primary targets for {case_id}/rep{rep}, got {len(trg)}')
        truth=target_truth_from_registry(c,units,arr,trg)
        require(len(truth)==25,'Target truth row count mismatch')
        # Seven-dose latent truth for every frozen target.
        lookup=units.set_index('block_id',drop=False); hard=[]
        for tr in trg.itertuples(index=False):
            row=lookup.loc[str(tr.target_block_id)]
            i=int(np.flatnonzero(units.block_id.astype(str).to_numpy()==str(tr.target_block_id))[0])
            for d in DOSES:
                y=float(conditional_mean(c,np.array([arr['base'][i]]),np.array([arr['xpv'][i]]),np.array([float(d)]))[0]+arr['e'][i])
                hard.append({'case_id':case_id,'replication':int(rep),'MineID':str(tr.MineID),'target_rank':int(tr.target_rank),'target_block_id':str(tr.target_block_id),'dose':float(d),'Y_true_hard_dose':y})
        hard=pd.DataFrame(hard)
        # Physical-domain hard gate before methods inspect performance.
        for name,v in [('Y_true_at_A',units.Y_true_at_A),('Y_observed_at_A',units.Y_observed_at_A),('Y_true_A_star',truth.Y_true_A_star),('Y_observed_A_star',truth.Y_observed_A_star),('Y_true_hard_dose',hard.Y_true_hard_dose)]:
            a=pd.to_numeric(v,errors='coerce').to_numpy(float); require(np.isfinite(a).all(),f'R09_PHYSICAL_DOMAIN_VIOLATION nonfinite {name} {case_id}/rep{rep}'); require((a>=-1).all() and (a<=1).all(),f'R09_PHYSICAL_DOMAIN_VIOLATION {name} range=({a.min()},{a.max()}) {case_id}/rep{rep}')
        ex=self.exact_targets[(self.exact_targets.case_id==case_id)&(self.exact_targets.replication==rep)].copy().reset_index(drop=True)
        require(len(ex)==5,'Exact-audit registry should have 5 mine rows')
        return TaskData(c,int(rep),sub,edges,wed,units,arr,trg,truth,hard,ex,exactmap,good)

def exact_audit_target_truth(task:TaskData, row:pd.Series):
    c=task.case; target=str(row.exact_audit_target_block_id); units=task.units; idx={str(b):i for i,b in enumerate(units.block_id.astype(str))}; require(target in idx,'Exact target absent from generated units'); i=idx[target]
    if c['primary_target_mode']=='observational':
        astar=float(units.iloc[i].A); dose=np.nan; h=np.nan
    else:
        dose=float(c['primary_target_dose']); h=c.get('primary_bandwidth'); rng=np.random.default_rng(stable_seed('GeoDoseCP-Stage5B-Exact-Astar',c['case_id'],task.rep,str(row.MineID),target))
        astar=sample_localized(rng,dose,None if pd.isna(h) else float(h),c['endpoint_structure']!='interior_only')
    mu=float(conditional_mean(c,np.array([task.truth_arrays['base'][i]]),np.array([task.truth_arrays['xpv'][i]]),np.array([astar]))[0]); ytrue=mu+float(task.truth_arrays['e'][i]); me=float(measurement_error(c,np.array([astar]),np.array([units.iloc[i].ue_proxy]),np.array([task.truth_arrays['substrate'][i]]),np.array([task.truth_arrays['measurement_z'][i]]))[0])
    return {'target_block_id':target,'A_star':astar,'requested_dose':dose,'bandwidth':h,'Y_true_A_star':ytrue,'Y_observed_A_star':ytrue+me,'target_index':i}
