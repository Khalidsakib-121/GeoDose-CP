from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import Stage3BExternal, target_row
from .io import D4Error


@dataclass
class EstimatedTreatmentFit:
    model: Any
    source_scenario_id: str
    frozen_nuisance_seed: int
    derived_propensity_seed: int
    fit_id: str
    training_data_hash: str
    n_train: int


def import_verified_stage3c(source_root: Path):
    src=(Path(source_root)/'src').resolve()
    if not src.is_dir(): raise D4Error(f"Stage3C source src/ missing: {src}")
    src_s=str(src)
    if src_s not in sys.path: sys.path.insert(0,src_s)
    # Never reuse a same-named package imported from another checkout/environment.
    for name in list(sys.modules):
        if name=='geodose_stage3c' or name.startswith('geodose_stage3c.'):
            mod=sys.modules[name]; fp=getattr(mod,'__file__',None)
            if fp:
                try: Path(fp).resolve().relative_to(src)
                except Exception: del sys.modules[name]
    models=importlib.import_module('geodose_stage3c.models')
    runner=importlib.import_module('geodose_stage3c.runner')
    weights=importlib.import_module('geodose_stage3c.weights')
    for mod in (models,runner,weights):
        fp=Path(mod.__file__).resolve()
        try: fp.relative_to(src)
        except Exception as exc: raise D4Error(f"Stage3C import escaped verified source root: {fp}") from exc
    return models,runner,weights


def fit_estimated_treatment(stage3c_source_root: Path, stage3a_seeds: pd.DataFrame, case_units: pd.DataFrame, scenario_id: str) -> EstimatedTreatmentFit:
    models,runner,_=import_verified_stage3c(stage3c_source_root)
    source,base_seed=runner.frozen_nuisance_seed(stage3a_seeds,str(scenario_id),1)
    prop_seed=runner.derived_seed(base_seed,'MIXED_PROPENSITY_ESTIMATED')
    train=case_units[case_units['role'].astype(str)=='nuisance_training'].copy()
    if train.empty: raise D4Error('No nuisance_training units for estimated treatment fit')
    model=models.MixedPropensityModel('estimated',prop_seed).fit(train)
    return EstimatedTreatmentFit(model,source,int(base_seed),int(prop_seed),str(model.fit_id),str(model.training_data_hash),len(train))


def patch_case_units_with_estimated_treatment(case_units: pd.DataFrame, fit: EstimatedTreatmentFit) -> tuple[pd.DataFrame,dict[str,float]]:
    out=case_units.copy()
    probs,alpha,beta=fit.model.components(out)
    if probs.shape!=(len(out),3): raise D4Error('Estimated treatment component shape mismatch')
    if np.any(~np.isfinite(probs)) or np.any(probs<0): raise D4Error('Estimated treatment category probabilities invalid')
    sums=probs.sum(axis=1)
    if np.max(np.abs(sums-1.0))>2e-10: raise D4Error('Estimated treatment category probabilities do not normalize')
    if np.any(~np.isfinite(alpha)) or np.any(~np.isfinite(beta)) or np.any(alpha<=0) or np.any(beta<=0):
        raise D4Error('Estimated interior Beta parameters invalid')
    out['pi_atom_0']=probs[:,0]; out['pi_atom_1']=probs[:,1]; out['pi_interior']=probs[:,2]
    out['beta_alpha']=alpha; out['beta_beta']=beta
    if 'oracle_g_mixed_at_A' in out.columns:
        out['oracle_g_mixed_at_A']=fit.model.density(out)
    return out,{
        'max_category_sum_abs_error':float(np.max(np.abs(sums-1.0))),
        'minimum_beta_alpha':float(np.min(alpha)),'minimum_beta_beta':float(np.min(beta)),
    }


def source_rows(case_units: pd.DataFrame, block_nodes: tuple[int,...], target_draw: pd.Series) -> pd.DataFrame:
    lookup=case_units.set_index('node_index',drop=False)
    rows=[]
    for node in block_nodes[:-1]:
        r=lookup.loc[int(node)].copy(); r['A_eval']=float(r['A']); r['source_type']='calibration'; rows.append(r)
    r=lookup.loc[int(block_nodes[-1])].copy(); r['A_eval']=float(target_draw['A_star']); r['source_type']='target_candidate'; rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def oracle_log_dose_ratios(data: Stage3BExternal, case_id: str, target_dose: float, block_nodes: tuple[int,...]) -> np.ndarray:
    cu=data.units[data.units.case_id.astype(str)==str(case_id)].set_index('node_index',drop=False)
    target_unit=str(cu.loc[int(block_nodes[-1]),'unit_id'])
    tr=target_row(data,case_id,target_unit,float(target_dose))
    vals=[]
    for node in block_nodes[:-1]:
        uid=str(cu.loc[int(node),'unit_id'])
        sub=data.treatment_truth[
            (data.treatment_truth.case_id.astype(str)==str(case_id))
            & (data.treatment_truth.unit_id.astype(str)==uid)
            & (data.treatment_truth.target_dose.astype(float).sub(float(target_dose)).abs()<=1e-12)
            & (data.treatment_truth.role.astype(str)=='calibration')
        ]
        if len(sub)!=1: raise D4Error(f"Missing oracle treatment ratio for {case_id}/{uid}/{target_dose}")
        vals.append(float(sub.iloc[0]['log_oracle_treatment_ratio']))
    vals.append(float(tr['log_oracle_treatment_ratio_at_A_star']))
    arr=np.asarray(vals,dtype=float)
    if np.isnan(arr).any() or np.isposinf(arr).any(): raise D4Error('Invalid oracle dose log ratio')
    return arr


def estimated_log_dose_ratios(stage3c_source_root: Path, fit: EstimatedTreatmentFit, case_units: pd.DataFrame, target_draw: pd.Series, block_nodes: tuple[int,...], target_dose: float) -> np.ndarray:
    _,_,weights=import_verified_stage3c(stage3c_source_root)
    src=source_rows(case_units,block_nodes,target_draw)
    a=src['A_eval'].to_numpy(float)
    target_unit=case_units[case_units.node_index.astype(int)==int(block_nodes[-1])]
    endpoint_audited=bool(target_unit['endpoint_audited'].iloc[0])
    bw=None if float(target_dose) in (0.0,1.0) else float(target_draw['bandwidth'])
    q=weights.intervention_density(a,float(target_dose),bw,endpoint_audited)
    base=src.drop(columns=['A_eval','source_type']).copy()
    g=fit.model.density(base,a=a)
    logs=np.full(len(src),-np.inf,dtype=float)
    ok=(q>0)&(g>0)&np.isfinite(g)
    logs[ok]=np.log(q[ok])-np.log(g[ok])
    if np.isnan(logs).any() or np.isposinf(logs).any(): raise D4Error('Invalid estimated dose log ratio')
    return logs
