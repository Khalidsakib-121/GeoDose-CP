from __future__ import annotations

import copy
import importlib
import inspect
import json
import math
import sys
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from .data import Stage3BExternal, case_row
from .fixture import (
    build_exact_fixture_dict, canonical_edges, clone_data_object, discover_data_fields,
    edge_degrees, replace_fixture_mapping,
)
from .io import D4Error
from .nuisance import EstimatedTreatmentFit, patch_case_units_with_estimated_treatment


def _import_d2(source_root: Path) -> dict[str,Any]:
    src=Path(source_root)/'src'
    if not src.is_dir():
        raise D4Error(f'D2 source src/ missing: {src}')
    # Make sure we do not accidentally reuse a different geodose_stage3d package.
    src_s=str(src)
    if src_s not in sys.path: sys.path.insert(0,src_s)
    for name in list(sys.modules):
        if name=='geodose_stage3d' or name.startswith('geodose_stage3d.'):
            mod=sys.modules[name]
            f=str(getattr(mod,'__file__',''))
            if f and src_s.lower() not in f.lower():
                del sys.modules[name]
    mods={}
    for short in ['d2_io','exact_law','candidate_inversion','types']:
        mods[short]=importlib.import_module(f'geodose_stage3d.{short}')
        f=Path(mods[short].__file__).resolve()
        try: f.relative_to(src.resolve())
        except Exception as exc: raise D4Error(f'D2 import escaped verified source root: {short} -> {f}') from exc
    return mods


def _mutate_config(config: Any, **updates: Any) -> Any:
    """Return a shallow copy of a D1 fixture configuration with selected fields changed.

    Supports dict, dataclass, and normal attribute containers.  Unknown fields are
    not silently introduced for dataclasses; for dict/attribute objects they are
    added only when required by the accepted build function contract.
    """
    if isinstance(config,dict):
        out=copy.deepcopy(config); out.update(updates); return out
    if is_dataclass(config):
        names={f.name for f in config.__dataclass_fields__.values()}
        use={k:v for k,v in updates.items() if k in names}
        missing=[k for k in ('case_id','target_dose','fixture_source') if k not in names]
        if missing: raise D4Error(f'D2 fixture config dataclass lacks required fields: {missing}')
        return replace(config,**use)
    out=copy.copy(config)
    for k,v in updates.items():
        if k in ('case_id','target_dose','fixture_source') and not hasattr(out,k):
            raise D4Error(f'D2 fixture config lacks required attribute {k}')
        setattr(out,k,v)
    return out


def _signature_record(obj: Any) -> str:
    try: return str(inspect.signature(obj))
    except Exception: return '<unavailable>'


def _load_api_lock(package_root: Path) -> dict[str,Any]:
    path=Path(package_root)/'configs'/'d2_api_lock.json'
    if not path.is_file(): raise D4Error(f'D4 D2 API lock missing: {path}')
    obj=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj,dict): raise D4Error('D4 D2 API lock is not a mapping')
    return obj


def _validate_frozen_d2_contract(source_root: Path, d2_io: Any, exact_module: Any, candidate_module: Any, api_lock: dict[str,Any], alpha: float) -> dict[str,Any]:
    """Validate the exact accepted D2 v1.2 API/tolerance/randomization contract.

    The uploaded/local D2 tree is already hash-verified. This additional lock prevents
    D4 from inferring or guessing call semantics. Every callable and frozen constant
    used by the adapter must match the accepted D2 release exactly.
    """
    sigs={
        'stage3b_data':str(inspect.signature(d2_io.stage3b_data)),
        'load_d1_fixture_config':str(inspect.signature(d2_io.load_d1_fixture_config)),
        'build_d2_fixture':str(inspect.signature(d2_io.build_d2_fixture)),
        'frozen_orbit_seed':str(inspect.signature(d2_io.frozen_orbit_seed)),
        'derive_seed':str(inspect.signature(d2_io.derive_seed)),
        'prepare_orbit':str(inspect.signature(exact_module.prepare_orbit)),
        'CandidateEvaluator':str(inspect.signature(candidate_module.CandidateEvaluator)),
        'CandidateEvaluator.evaluate':str(inspect.signature(candidate_module.CandidateEvaluator.evaluate)),
        'acceptance':str(inspect.signature(candidate_module.acceptance)),
    }
    expected=api_lock.get('signatures',{})
    bad={k:{'expected':expected.get(k),'observed':v} for k,v in sigs.items() if expected.get(k)!=v}
    if bad: raise D4Error(f'Accepted D2 API signature lock mismatch: {bad}')

    tol_path=Path(source_root)/'configs'/'stage3d_d2_tolerances.yaml'
    cand_path=Path(source_root)/'configs'/'stage3d_d2_candidate_contract.yaml'
    frozen_tol=yaml.safe_load(tol_path.read_text(encoding='utf-8'))
    cand=yaml.safe_load(cand_path.read_text(encoding='utf-8'))
    for key,val in api_lock.get('frozen_tolerances',{}).items():
        if key not in frozen_tol or not math.isclose(float(frozen_tol[key]),float(val),rel_tol=0.0,abs_tol=0.0):
            raise D4Error(f'Accepted D2 frozen tolerance mismatch for {key}: expected {val!r}, observed {frozen_tol.get(key)!r}')
    if not math.isclose(float(alpha),float(api_lock.get('alpha_primary')),rel_tol=0.0,abs_tol=0.0):
        raise D4Error(f'D4 alpha {alpha} differs from accepted D2 alpha {api_lock.get("alpha_primary")}')
    rc=api_lock.get('randomization_contract',{})
    observed_rc={k:cand['randomization'][k] for k in rc}
    if observed_rc!=rc: raise D4Error(f'Accepted D2 randomization contract mismatch: expected={rc}, observed={observed_rc}')
    return {'pass':True,'signatures':sigs,'frozen_tolerances':{k:frozen_tol[k] for k in api_lock['frozen_tolerances']},'randomization_contract':observed_rc}


def _construct_candidate_evaluator(cls: Any, prepared: Any, alpha: float, tie_uniform: float, tolerances: dict[str,float]) -> tuple[Any,dict[str,Any]]:
    """Instantiate the *exact frozen D2 v1.2* CandidateEvaluator contract.

    No semantic aliases or guessed parameters are allowed. All five keyword-only
    values are read from the byte-verified accepted D2 tolerance registry.
    """
    sig=inspect.signature(cls)
    required=['score_tie_abs_tolerance','score_tie_rel_tolerance','non_gaussian_min_abs_residual']
    missing=[k for k in required if k not in tolerances]
    if missing: raise D4Error(f'Accepted D2 tolerance registry missing CandidateEvaluator inputs: {missing}')
    kwargs={
        'alpha':float(alpha),
        'tie_uniform':float(tie_uniform),
        'score_abs_tolerance':float(tolerances['score_tie_abs_tolerance']),
        'score_rel_tolerance':float(tolerances['score_tie_rel_tolerance']),
        'non_gaussian_min_abs_residual':float(tolerances['non_gaussian_min_abs_residual']),
    }
    try: sig.bind(prepared,**kwargs)
    except TypeError as exc: raise D4Error(f'Accepted D2 CandidateEvaluator exact binding failed; signature={sig}: {exc}') from exc
    try: ev=cls(prepared,**kwargs)
    except Exception as exc: raise D4Error(f'Could not instantiate accepted D2 CandidateEvaluator signature={sig}: {exc}') from exc
    return ev,{'signature':str(sig),'bound_parameters':['prepared',*kwargs.keys()],'bound_values':{k:(float(v) if isinstance(v,(int,float,np.number)) else v) for k,v in kwargs.items()}}


def _preflight_d2_callable_signatures(d2_io: Any, exact_module: Any, candidate_module: Any, source_root: Path, base_data: Any, tolerances: dict[str,Any], api_lock: dict[str,Any], alpha: float) -> dict[str,Any]:
    # Exact lock comparison catches required positional/keyword-only arguments in one pass.
    exact=_validate_frozen_d2_contract(source_root,d2_io,exact_module,candidate_module,api_lock,alpha)
    threshold=float(tolerances['minimum_precision_eigenvalue'])
    # Exercise Signature.bind on the exact calls D4 will make.
    checks={}
    def bind(name: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        sig=inspect.signature(fn)
        try: sig.bind(*args,**kwargs)
        except TypeError as exc: raise D4Error(f'Accepted D2 API preflight failed for {name}; signature={sig}: {exc}') from exc
        checks[name]=str(sig)
    bind('stage3b_data',d2_io.stage3b_data,source_root)
    bind('load_d1_fixture_config',d2_io.load_d1_fixture_config,source_root,'G6_GAUSS_INTERIOR_TRUTH')
    bind('build_d2_fixture',d2_io.build_d2_fixture,base_data,source_root,'G6_GAUSS_INTERIOR_TRUTH')
    bind('frozen_orbit_seed',d2_io.frozen_orbit_seed,source_root,'S4',1)
    bind('derive_seed',d2_io.derive_seed,1,'D2_G3_TIE_UNIFORM','fixture')
    bind('prepare_orbit',exact_module.prepare_orbit,object(),minimum_precision_eigenvalue_required=threshold)
    bind('CandidateEvaluator',candidate_module.CandidateEvaluator,object(),alpha=alpha,tie_uniform=0.5,score_abs_tolerance=float(tolerances['score_tie_abs_tolerance']),score_rel_tolerance=float(tolerances['score_tie_rel_tolerance']),non_gaussian_min_abs_residual=float(tolerances['non_gaussian_min_abs_residual']))
    bind('CandidateEvaluator.evaluate',candidate_module.CandidateEvaluator.evaluate,object(),0.0)
    bind('acceptance',candidate_module.acceptance,object(),'conservative')
    return {'pass':True,'callables':checks,'minimum_precision_eigenvalue_required':threshold,'exact_lock':exact}


def _prepare_orbit_with_frozen_precision(exact_module: Any, fixture: Any, tolerances: dict[str,Any]) -> tuple[Any,dict[str,Any]]:
    """Call the accepted D2 ``prepare_orbit`` using its frozen precision threshold.

    The accepted D2 v1.2 source requires the keyword-only argument
    ``minimum_precision_eigenvalue_required``.  D4 must not guess this value: it is
    read from the byte-verified accepted D2 tolerance registry
    ``configs/stage3d_d2_tolerances.yaml`` under the key
    ``minimum_precision_eigenvalue``.  Any signature or tolerance mismatch fails
    closed before scientific computation.
    """
    fn=exact_module.prepare_orbit
    sig=inspect.signature(fn)
    pname='minimum_precision_eigenvalue_required'
    if pname not in sig.parameters:
        raise D4Error(
            'Accepted D2 prepare_orbit signature is incompatible with the frozen D4 adapter: '
            f'expected keyword-only parameter {pname!r}; signature={sig}. '
            'D4 refuses rather than guessing.'
        )
    param=sig.parameters[pname]
    if param.kind not in (inspect.Parameter.KEYWORD_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD):
        raise D4Error(
            f'Accepted D2 prepare_orbit parameter {pname!r} has unsupported kind {param.kind}; '
            f'signature={sig}.'
        )
    if 'minimum_precision_eigenvalue' not in tolerances:
        raise D4Error('Accepted D2 tolerance registry lacks minimum_precision_eigenvalue')
    try:
        threshold=float(tolerances['minimum_precision_eigenvalue'])
    except Exception as exc:
        raise D4Error('Accepted D2 minimum_precision_eigenvalue is not numeric') from exc
    if not math.isfinite(threshold) or threshold<=0.0:
        raise D4Error(f'Accepted D2 minimum_precision_eigenvalue must be finite and positive, got {threshold!r}')
    try:
        prepared=fn(fixture,minimum_precision_eigenvalue_required=threshold)
    except Exception as exc:
        raise D4Error(
            'Accepted D2 prepare_orbit rejected the fixture when called with the frozen '
            f'minimum precision eigenvalue {threshold:.17g}; signature={sig}: {exc}'
        ) from exc
    return prepared,{
        'signature':str(sig),
        'minimum_precision_eigenvalue_required':threshold,
        'threshold_source':'accepted_d2/configs/stage3d_d2_tolerances.yaml:minimum_precision_eigenvalue',
    }


def _get_numeric(obj: Any, names: list[str]) -> float|None:
    for n in names:
        if hasattr(obj,n):
            v=getattr(obj,n)
            if callable(v):
                try: v=v()
                except TypeError: continue
            try:
                x=float(v)
                if math.isfinite(x): return x
            except Exception: pass
    return None


def _get_bool(obj: Any, names: list[str]) -> bool|None:
    for n in names:
        if hasattr(obj,n):
            v=getattr(obj,n)
            if callable(v):
                try: v=v()
                except TypeError: continue
            if isinstance(v,(bool,np.bool_)): return bool(v)
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value,(bool,np.bool_)): return bool(value)
    if isinstance(value,(int,np.integer)): return bool(int(value))
    if isinstance(value,str):
        x=value.strip().lower()
        if x in {'true','1','yes','y'}: return True
        if x in {'false','0','no','n'}: return False
    raise D4Error(f'Cannot interpret boolean value safely: {value!r}')


def _candidate_result_summary(result: Any, acceptance_fn: Callable[...,Any], alpha: float) -> dict[str,Any]:
    """Extract the exact accepted D2 CandidateResult fields; no alias fallback."""
    required=['conservative_p','randomized_p','conservative_accept','randomized_accept','tie_uniform','singular_conservative_inclusion','distinct_states']
    missing=[n for n in required if not hasattr(result,n)]
    if missing: raise D4Error(f'Accepted D2 CandidateResult missing required fields {missing}')
    cp=float(result.conservative_p); rp=float(result.randomized_p); u=float(result.tie_uniform)
    if not (-1e-12<=cp<=1+1e-12 and -1e-12<=rp<=1+1e-12 and 0.0<=u<=1.0):
        raise D4Error(f'Invalid accepted D2 p/tie values: conservative={cp}, randomized={rp}, U={u}')
    try:
        ca=acceptance_fn(result,'conservative'); ra=acceptance_fn(result,'randomized')
    except Exception as exc: raise D4Error(f'Accepted D2 acceptance(result, mode) API failed: {exc}') from exc
    if not isinstance(ca,(bool,np.bool_)) or not isinstance(ra,(bool,np.bool_)):
        raise D4Error('Accepted D2 acceptance(result, mode) returned non-boolean')
    if bool(ca)!=bool(result.conservative_accept) or bool(ra)!=bool(result.randomized_accept):
        raise D4Error('Accepted D2 acceptance helper disagrees with CandidateResult flags')
    return {
        'conservative_p':cp,'randomized_p':rp,'tie_uniform':u,
        'conservative_accept':bool(ca),'randomized_accept':bool(ra),
        'singular_conservative_inclusion':bool(result.singular_conservative_inclusion),
        'distinct_states':int(result.distinct_states),
        'result_type':f'{type(result).__module__}.{type(result).__name__}',
    }


def _tie_uniform(d2_io: Any, source_root: Path, fixture_id: str, *, replication: int=1, label: str='D2_G3_TIE_UNIFORM') -> tuple[float,dict[str,Any]]:
    """Reproduce the accepted D2 randomization seed exactly.

    D2 freezes *all* exact fixtures to the S4 replication-1 orbit seed, then derives
    one fixture-specific sub-seed using (label, fixture_id). The uniform is held fixed
    across candidate y.
    """
    base=int(d2_io.frozen_orbit_seed(source_root,'S4',int(replication)))
    derived=int(d2_io.derive_seed(base,str(label),str(fixture_id)))
    u=float(np.random.default_rng(derived).random())
    return u,{'orbit_seed':base,'derived_tie_seed':derived,'seed_scenario':'S4','seed_replication':int(replication),'derived_subseed_label':str(label),'tie_fixture_id':str(fixture_id)}


class D2SourceAdapter:
    def __init__(self, source_root: Path, alpha: float, tolerances: dict[str,float], api_lock: dict[str,Any]):
        self.source_root=Path(source_root)
        self.alpha=float(alpha)
        tol_path=self.source_root/'configs'/'stage3d_d2_tolerances.yaml'
        if not tol_path.is_file(): raise D4Error(f'Accepted D2 tolerance file missing: {tol_path}')
        frozen=yaml.safe_load(tol_path.read_text(encoding='utf-8'))
        if not isinstance(frozen,dict): raise D4Error('Accepted D2 tolerance registry is not a mapping')
        self.tolerances=dict(frozen)
        # D4 may tighten only adapter-level aliases; accepted D2 values remain authoritative.
        for k,v in tolerances.items():
            if k not in self.tolerances: self.tolerances[k]=v
        self.mods=_import_d2(self.source_root)
        self.d2_io=self.mods['d2_io']; self.exact=self.mods['exact_law']; self.cinv=self.mods['candidate_inversion']
        self.api={
            'CandidateEvaluator':_signature_record(self.cinv.CandidateEvaluator),
            'CandidateEvaluator.evaluate':_signature_record(self.cinv.CandidateEvaluator.evaluate),
            'build_d2_fixture':_signature_record(self.d2_io.build_d2_fixture),
            'load_d1_fixture_config':_signature_record(self.d2_io.load_d1_fixture_config),
            'stage3b_data':_signature_record(self.d2_io.stage3b_data),
            'prepare_orbit':_signature_record(self.exact.prepare_orbit),
        }
        self.base_data=self.d2_io.stage3b_data(self.source_root)
        self.fields=discover_data_fields(self.base_data)
        self.api_lock=dict(api_lock)
        self.api['preflight']=_preflight_d2_callable_signatures(
            self.d2_io,self.exact,self.cinv,self.source_root,self.base_data,self.tolerances,self.api_lock,self.alpha
        )

    def _make_custom_data(self, external: Stage3BExternal, case_id: str, target_dose: float, block_nodes: tuple[int,...], treatment_mode: str, graph_mode: str, fit: EstimatedTreatmentFit|None) -> tuple[Any,dict[str,Any]]:
        units=external.units.copy()
        cases=external.cases.copy()
        true_edges=external.true_edges.copy()
        fitted_edges=external.fitted_edges.copy()
        fixture_edges=(true_edges if graph_mode=='TG' else fitted_edges)
        fixture_edges_case=fixture_edges[fixture_edges.case_id.astype(str)==str(case_id)][['source_node','target_node']].copy()
        if fixture_edges_case.empty: raise D4Error(f'No {graph_mode} graph edges for {case_id}')

        # Exact D2 GMRF specialization also represents iid Gaussian residuals at rho=0.
        # Stage3B names those rows iid_continuous; use an explicit computational alias only.
        representation_override=False
        mask_case=cases.case_id.astype(str)==str(case_id)
        if int(mask_case.sum())!=1: raise D4Error(f'Case registry row missing for {case_id}')
        crow=cases.loc[mask_case].iloc[0]
        if str(crow['residual_law'])=='iid_continuous':
            if abs(float(crow['spatial_rho']))>1e-15: raise D4Error('iid_continuous alias allowed only at rho=0')
            cases.loc[mask_case,'residual_law']='gaussian_gmrf'; representation_override=True

        fit_diag={}
        if treatment_mode=='ET':
            if fit is None: raise D4Error('ET variant missing estimated treatment fit')
            cu=units[units.case_id.astype(str)==str(case_id)].copy()
            patched,fit_diag=patch_case_units_with_estimated_treatment(cu,fit)
            mask=units.case_id.astype(str)==str(case_id)
            # Update only treatment-law fields; factual A/Y and oracle outcome fields stay fixed.
            for col in ['pi_atom_0','pi_atom_1','pi_interior','beta_alpha','beta_beta']:
                units.loc[mask,col]=patched[col].to_numpy()
            if 'oracle_g_mixed_at_A' in units.columns and 'oracle_g_mixed_at_A' in patched.columns:
                units.loc[mask,'oracle_g_mixed_at_A']=patched['oracle_g_mixed_at_A'].to_numpy()
        elif treatment_mode!='OT': raise D4Error(f'Unknown treatment mode {treatment_mode}')

        if graph_mode=='FG':
            # D1/D2 exact code reads the field identified as the true graph. Replace only
            # the selected case with the accepted Stage3B fitted edge set. This is a D4
            # misspecification diagnostic; rho and scale intentionally remain oracle.
            true_edges=true_edges[true_edges.case_id.astype(str)!=str(case_id)].copy()
            add=external.fitted_edges[external.fitted_edges.case_id.astype(str)==str(case_id)].copy()
            # Align extra columns conservatively to the true-edge table schema.
            for c in true_edges.columns:
                if c not in add.columns: add[c]=np.nan
            add=add[[c for c in true_edges.columns]]
            true_edges=pd.concat([true_edges,add],ignore_index=True)
            # Keep graph-derived slot metadata consistent with the graph used by exact law.
            cu_mask=units.case_id.astype(str)==str(case_id)
            n=int(units.loc[cu_mask,'node_index'].max())+1
            deg=edge_degrees(fixture_edges_case,n)
            nodes=units.loc[cu_mask,'node_index'].astype(int).to_numpy()
            units.loc[cu_mask,'graph_degree_true']=deg[nodes]

        fixture=build_exact_fixture_dict(external,case_id,target_dose,block_nodes,fixture_edges_case)
        attrs={}
        # External accepted Stage3B tables are schema-compatible with the D2-embedded
        # Stage3B data. Preserve fields not needed by D4 from the accepted base object.
        attrs[self.fields['units']]=units
        attrs[self.fields['cases']]=cases
        attrs[self.fields['true_edges']]=true_edges
        if 'fitted_edges' in self.fields:
            attrs[self.fields['fitted_edges']]=fitted_edges
        base_map=getattr(self.base_data,self.fields['fixtures']) if not isinstance(self.base_data,dict) else self.base_data[self.fields['fixtures']]
        attrs[self.fields['fixtures']]=replace_fixture_mapping(base_map,fixture,case_id,target_dose)
        custom=clone_data_object(self.base_data,attrs)
        return custom,{
            'residual_representation_override_iid_as_rho0_gmrf':representation_override,
            'treatment_fit_diagnostics':fit_diag,'graph_mode':graph_mode,'treatment_mode':treatment_mode,
            'block_nodes':list(map(int,block_nodes)),'fixture_distinct_permutations':int(fixture['distinct_permutations']),
        }

    def _build_prepared(self, data_obj: Any, case_id: str, target_dose: float) -> tuple[Any,dict[str,Any]]:
        # Preserve the accepted endpoint/interior configuration family while swapping
        # only the Stage3B case, dose and custom six-node fixture source.  This avoids
        # silently forcing endpoint cases through an interior-only D1 configuration.
        if abs(float(target_dose)-0.0)<=1e-15:
            base_id='G6_GAUSS_ENDPOINT0'
        elif abs(float(target_dose)-1.0)<=1e-15:
            base_id='G6_GAUSS_ENDPOINT1'
        else:
            base_id='G6_GAUSS_INTERIOR_TRUTH'
        original=self.d2_io.load_d1_fixture_config
        base_cfg=original(self.source_root,base_id)
        custom_cfg=_mutate_config(base_cfg,case_id=str(case_id),target_dose=float(target_dose),fixture_source='size6_unique',candidate_rule='stored_simulation_truth')
        def patched(package_root: Path, d1_evaluation_id: str):
            if str(d1_evaluation_id)==base_id: return custom_cfg
            return original(package_root,d1_evaluation_id)
        self.d2_io.load_d1_fixture_config=patched
        try:
            fixture=self.d2_io.build_d2_fixture(data_obj,self.source_root,base_id)
        except Exception as exc:
            raise D4Error(f'Accepted D2 build_d2_fixture rejected new D4 case {case_id}/dose={target_dose}: {exc}') from exc
        finally:
            self.d2_io.load_d1_fixture_config=original
        prepared,prepare_meta=_prepare_orbit_with_frozen_precision(self.exact,fixture,self.tolerances)
        return prepared,{'base_evaluation_id':base_id,'custom_config_repr':repr(custom_cfg),'fixture_type':f'{type(fixture).__module__}.{type(fixture).__name__}','prepared_type':f'{type(prepared).__module__}.{type(prepared).__name__}','prepare_orbit':prepare_meta}

    def _evaluator(self, prepared: Any, fixture_id: str) -> tuple[Any,dict[str,Any]]:
        label=str(self.api_lock['randomization_contract']['derived_subseed_label'])
        u,seed_meta=_tie_uniform(self.d2_io,self.source_root,fixture_id,replication=1,label=label)
        ev,ctor=_construct_candidate_evaluator(self.cinv.CandidateEvaluator,prepared,self.alpha,u,self.tolerances)
        return ev,{'tie_uniform':u,**seed_meta,**ctor}

    def accepted_replay_fixture(
        self, accepted_d2_trace: pd.DataFrame, d1_evaluation_id: str, d2_fixture_id: str,
        candidates: list[float], scenario_id: str='S4'
    ) -> tuple[pd.DataFrame,dict[str,Any]]:
        data=self.d2_io.stage3b_data(self.source_root)
        fixture=self.d2_io.build_d2_fixture(data,self.source_root,str(d1_evaluation_id))
        prepared,prepare_meta=_prepare_orbit_with_frozen_precision(self.exact,fixture,self.tolerances)
        ev,meta=self._evaluator(prepared,str(d2_fixture_id))
        rows=[]
        ref=accepted_d2_trace[accepted_d2_trace.fixture_id.astype(str)==str(d2_fixture_id)].copy()
        if ref.empty: raise D4Error(f'Accepted D2 trace lacks fixture {d2_fixture_id}')
        for y in candidates:
            try: result=ev.evaluate(float(y))
            except Exception as exc: raise D4Error(f'Accepted D2 CandidateEvaluator replay failed for {d2_fixture_id} at y={y}: {exc}') from exc
            ss=_candidate_result_summary(result,self.cinv.acceptance,self.alpha)
            h=float(y).hex(); rr=ref[ref.candidate_hex.astype(str)==h]
            if len(rr)!=1: raise D4Error(f'Accepted D2 reference candidate missing for {d2_fixture_id}: {h}')
            r=rr.iloc[0]
            accepted_bool=_as_bool(r['conservative_accept'])
            rows.append({
                'd1_evaluation_id':str(d1_evaluation_id),'fixture_id':str(d2_fixture_id),
                'candidate_y':float(y),'candidate_hex':h,
                'source_conservative_p':ss['conservative_p'],'accepted_conservative_p':float(r['conservative_p']),
                'abs_p_error':abs(ss['conservative_p']-float(r['conservative_p'])),
                'source_randomized_p':ss['randomized_p'],'accepted_randomized_p':float(r['randomized_p']),
                'abs_randomized_p_error':abs(ss['randomized_p']-float(r['randomized_p'])),
                'source_tie_uniform':ss['tie_uniform'],'accepted_tie_uniform':float(r['tie_uniform']),
                'abs_tie_uniform_error':abs(ss['tie_uniform']-float(r['tie_uniform'])),
                'source_accept':bool(ss['conservative_accept']),'accepted_accept':accepted_bool,
                'accept_match':bool(ss['conservative_accept'])==accepted_bool,
                'source_randomized_accept':bool(ss['randomized_accept']),'accepted_randomized_accept':_as_bool(r['randomized_accept']),
                'randomized_accept_match':bool(ss['randomized_accept'])==_as_bool(r['randomized_accept']),
                'source_distinct_states':int(ss['distinct_states']),'accepted_distinct_states':int(r['distinct_states']),
                'distinct_states_match':int(ss['distinct_states'])==int(r['distinct_states']),
                'result_type':ss['result_type'],
            })
        out=pd.DataFrame(rows)
        return out,{'adapter_api':self.api,'prepare_orbit':prepare_meta,'candidate_evaluator':meta,'data_fields':self.fields,
                    'd1_evaluation_id':str(d1_evaluation_id),'d2_fixture_id':str(d2_fixture_id)}

    def accepted_replay(self, accepted_d2_trace: pd.DataFrame, candidates: list[float]) -> tuple[pd.DataFrame,dict[str,Any]]:
        return self.accepted_replay_fixture(
            accepted_d2_trace,'G6_GAUSS_INTERIOR_TRUTH','G6_GAUSS_INTERIOR',candidates,'S4'
        )

    def prepare_new_case(self, external: Stage3BExternal, case_id: str, scenario_id: str, target_dose: float, block_nodes: tuple[int,...], treatment_mode: str, graph_mode: str, fit: EstimatedTreatmentFit|None) -> tuple[Any,dict[str,Any]]:
        custom,meta=self._make_custom_data(external,case_id,target_dose,block_nodes,treatment_mode,graph_mode,fit)
        prepared,bmeta=self._build_prepared(custom,case_id,target_dose)
        fixture_id=f'D4::{case_id}::dose={float(target_dose).hex()}::{treatment_mode}_{graph_mode}'
        ev,emeta=self._evaluator(prepared,fixture_id)
        return ev,{**meta,**bmeta,'d4_fixture_id':fixture_id,'candidate_evaluator':emeta}

    def evaluate_new_candidate(self, evaluator: Any, candidate_y: float) -> dict[str,Any]:
        try: result=evaluator.evaluate(float(candidate_y))
        except Exception as exc: raise D4Error(f'Accepted D2 CandidateEvaluator failed on new candidate y={candidate_y}: {exc}') from exc
        return _candidate_result_summary(result,self.cinv.acceptance,self.alpha)
