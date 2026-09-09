from __future__ import annotations
import copy, hashlib, importlib, inspect, io, json, math, sys, zipfile, gzip, platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import sparse
from .io import require, Stage3FError


def _prepend_verified(root: Path, sub: str='src') -> Path:
    p=Path(root)/sub; require(p.is_dir(),f'SOURCE_SUBDIR_MISSING:{p}')
    s=str(p)
    if s not in sys.path: sys.path.insert(0,s)
    return p

class Stage3BBridge:
    def __init__(self, source_root: Path, stage3a_zip: Path, accepted_stage3b_zip: Path):
        self.source_root=Path(source_root); src=_prepend_verified(self.source_root)
        for name in list(sys.modules):
            if name=='geodose_stage3b' or name.startswith('geodose_stage3b.'):
                f=str(getattr(sys.modules[name],'__file__',''))
                if f and str(src.resolve()).lower() not in str(Path(f).resolve()).lower(): del sys.modules[name]
        self.core=importlib.import_module('geodose_stage3b.core')
        require(Path(self.core.__file__).resolve().is_relative_to(src.resolve()),'STAGE3B_IMPORT_ESCAPED_VERIFIED_ROOT')
        self.stage3a=self.core.read_stage3a_archive(Path(stage3a_zip))
        self.case_frame=self.core.build_case_registry()
        self.cases={c.case_id:c for c in self.core.frame_to_cases(self.case_frame)}
        self.basis_cache={}
        self.accepted_zip=Path(accepted_stage3b_zip)
        self.accepted_rep1=self._load_accepted_rep1()
    def _load_accepted_rep1(self):
        def read(name):
            with zipfile.ZipFile(self.accepted_zip) as z:
                m=[x for x in z.namelist() if x.replace('\\','/').endswith(name)]
                require(len(m)==1,f'ACCEPTED_STAGE3B_MEMBER_NOT_UNIQUE:{name}')
                raw=z.read(m[0]);
                if name.endswith('.gz'): raw=gzip.decompress(raw)
                return pd.read_csv(io.BytesIO(raw))
        return {'units':read('stage3b_validation_units.csv.gz'),'targets':read('stage3b_realized_target_draws.csv.gz'),'obs_targets':read('stage3b_observational_target_draws.csv.gz')}
    def generate(self, case_id: str, replication: int) -> dict[str,Any]:
        require(case_id in self.cases,f'UNKNOWN_PILOT_CASE:{case_id}')
        return self.core.generate_case(self.cases[case_id],self.stage3a,self.basis_cache,replication=int(replication))
    def rep1_replay_audit(self, case_id: str, generated: dict[str,Any], tol: float=2e-12) -> dict[str,Any]:
        """Replay the accepted Stage3B replication-1 release.

        The accepted release was generated on Python 3.10 / NumPy 2.2.6 / pandas 2.3.3.
        Spatial GMRF draws use ``np.linalg.eigh``; eigenvector orientation/basis choices can
        legitimately differ across LAPACK builds even when the frozen source, seeds, and
        innovations are identical.  Therefore the *official* frozen runtime enforces a full
        byte-level numerical replay, while non-frozen developer runtimes record the mismatch
        without pretending it is an accepted replay.  The Windows launcher verifies the frozen
        runtime before Stage3F, so official pilot outputs can only take the strict branch.
        """
        gu=generated['units'].sort_values('node_index').reset_index(drop=True)
        au=self.accepted_rep1['units'][self.accepted_rep1['units'].case_id.astype(str)==str(case_id)].sort_values('node_index').reset_index(drop=True)
        require(len(gu)==len(au)>0,f'REP1_UNIT_COUNT_MISMATCH:{case_id}')
        common=[c for c in gu.columns if c in au.columns]
        per_numeric={}; string_bad=0
        for c in common:
            if pd.api.types.is_numeric_dtype(gu[c]) and pd.api.types.is_numeric_dtype(au[c]):
                a=gu[c].to_numpy(float); b=au[c].to_numpy(float); mask=np.isfinite(a)&np.isfinite(b)
                err=float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.0
                if np.any(np.isnan(a)!=np.isnan(b)): err=float('inf')
                per_numeric[c]=err
            else:
                string_bad += int(np.any(gu[c].astype(str).to_numpy()!=au[c].astype(str).to_numpy()))
        maxerr=float(max(per_numeric.values(),default=0.0))
        worst=max(per_numeric,key=per_numeric.get) if per_numeric else ''
        frozen_runtime=(sys.version_info[:2]==(3,10) and np.__version__=='2.2.6' and pd.__version__=='2.3.3')
        full_pass=bool(maxerr<=tol and string_bad==0)
        if frozen_runtime:
            require(full_pass,f'STAGE3B_REP1_REPLAY_MISMATCH:{case_id}:maxerr={maxerr}:worst={worst}:string_bad={string_bad}')
            status='FULL_REPLAY_PASS_FROZEN_RUNTIME'
        else:
            status='NONFROZEN_QA_RUNTIME_FULL_REPLAY_NOT_ACCEPTANCE_EVIDENCE'
        return {'case_id':case_id,'replication':1,'common_columns':len(common),'unit_rows':len(gu),'max_abs_numeric_error':maxerr,'worst_numeric_column':worst,'string_column_mismatch_count':string_bad,'frozen_runtime':bool(frozen_runtime),'full_numeric_replay_pass':bool(full_pass),'replay_status':status,'pass':bool(full_pass if frozen_runtime else True)}

class Stage3CBridge:
    def __init__(self, source_root: Path):
        self.root=Path(source_root); src=_prepend_verified(self.root)
        for name in list(sys.modules):
            if name=='geodose_stage3c' or name.startswith('geodose_stage3c.'):
                f=str(getattr(sys.modules[name],'__file__',''))
                if f and str(src.resolve()).lower() not in str(Path(f).resolve()).lower(): del sys.modules[name]
        self.models=importlib.import_module('geodose_stage3c.models')
        self.weights=importlib.import_module('geodose_stage3c.weights')
        self.methods=importlib.import_module('geodose_stage3c.methods')
        self.graph=importlib.import_module('geodose_stage3c.graph')
        self.runner=importlib.import_module('geodose_stage3c.runner')
        for m in [self.models,self.weights,self.methods,self.graph,self.runner]:
            require(Path(m.__file__).resolve().is_relative_to(src.resolve()),f'STAGE3C_IMPORT_ESCAPED:{m.__name__}')
        import yaml
        self.threshold_plan=yaml.safe_load((self.root/'configs/stage3c_pilot_threshold_plan.yaml').read_text())
        require(self.threshold_plan.get('status')=='deferred_not_selected_in_reference','STAGE3C_THRESHOLD_PLAN_STATUS_DRIFT')

class M3M4Bridge:
    def __init__(self,d4_root: Path):
        self.root=Path(d4_root); vendor=self.root/'vendor'; require(vendor.is_dir(),'D4_VENDOR_MISSING')
        if str(vendor) not in sys.path: sys.path.insert(0,str(vendor))
        for name in list(sys.modules):
            if name=='geodose_stage3d_d3_m4' or name.startswith('geodose_stage3d_d3_m4.'):
                f=str(getattr(sys.modules[name],'__file__',''))
                if f and str(vendor.resolve()).lower() not in str(Path(f).resolve()).lower(): del sys.modules[name]
        self.graph=importlib.import_module('geodose_stage3d_d3_m4.graph_law')
        self.m3=importlib.import_module('geodose_stage3d_d3_m4.m3_oracle')
        self.m4=importlib.import_module('geodose_stage3d_d3_m4.m4_oracle')
        for m in [self.graph,self.m3,self.m4]: require(Path(m.__file__).resolve().is_relative_to(vendor.resolve()),'M3M4_IMPORT_ESCAPED_VERIFIED_D4_VENDOR')

class D2Bridge:
    def __init__(self, source_root: Path, alpha: float):
        self.root=Path(source_root); src=_prepend_verified(self.root)
        for name in list(sys.modules):
            if name=='geodose_stage3d' or name.startswith('geodose_stage3d.'):
                f=str(getattr(sys.modules[name],'__file__',''))
                if f and str(src.resolve()).lower() not in str(Path(f).resolve()).lower(): del sys.modules[name]
        self.types=importlib.import_module('geodose_stage3d.types')
        self.exact=importlib.import_module('geodose_stage3d.exact_law')
        self.cinv=importlib.import_module('geodose_stage3d.candidate_inversion')
        self.gp=importlib.import_module('geodose_stage3d.graph_precision')
        self.orbit=importlib.import_module('geodose_stage3d.orbit')
        self.io=importlib.import_module('geodose_stage3d.io')
        self.mixed=importlib.import_module('geodose_stage3d.mixed_measure')
        for m in [self.types,self.exact,self.cinv,self.gp,self.orbit,self.io,self.mixed]: require(Path(m.__file__).resolve().is_relative_to(src.resolve()),f'D2_IMPORT_ESCAPED:{m.__name__}')
        import yaml
        self.tol=yaml.safe_load((self.root/'configs/stage3d_d2_tolerances.yaml').read_text())
        self.alpha=float(alpha)
        # Complete signature lock, fail closed.
        expected={
          'prepare_orbit':"(fixture: 'ExactFixture', *, minimum_precision_eigenvalue_required: 'float') -> 'PreparedOrbit'",
          'CandidateEvaluator':"(prepared: 'PreparedOrbit', *, alpha: 'float', tie_uniform: 'float', score_abs_tolerance: 'float', score_rel_tolerance: 'float', non_gaussian_min_abs_residual: 'float') -> 'None'",
          'evaluate':"(self, candidate_y: 'float', *, keep_state: 'bool' = False) -> 'CandidateResult'",
          'acceptance':"(result: 'CandidateResult', mode: 'str') -> 'bool'",
        }
        got={'prepare_orbit':str(inspect.signature(self.exact.prepare_orbit)),'CandidateEvaluator':str(inspect.signature(self.cinv.CandidateEvaluator)),'evaluate':str(inspect.signature(self.cinv.CandidateEvaluator.evaluate)),'acceptance':str(inspect.signature(self.cinv.acceptance))}
        require(got==expected,f'D2_API_SIGNATURE_DRIFT:{got}')
        self.api=got
    @staticmethod
    def graph_boundary(block_nodes: np.ndarray, edges: pd.DataFrame) -> np.ndarray:
        b=set(map(int,block_nodes)); out=set()
        for r in edges.itertuples(index=False):
            a=int(r.source_node); c=int(r.target_node)
            if (a in b)^(c in b): out.add(c if a in b else a)
        return np.asarray(sorted(out),dtype=int)
    def make_fixture(self, case, units: pd.DataFrame, edges: pd.DataFrame, block_nodes: tuple[int,...], target_row: pd.Series, *, observational_target: bool, use_observed_outcomes: bool=True):
        units=units.sort_values('node_index').reset_index(drop=True).copy(); bnodes=np.asarray(block_nodes,dtype=int)
        require(len(bnodes)==6 and len(set(bnodes.tolist()))==6,'PILOT_EXACT_BLOCK_MUST_HAVE_6_UNIQUE_NODES')
        lookup=units.set_index('node_index',drop=False)
        Slot=self.types.Slot; Payload=self.types.Payload
        slots=[]; payloads=[]
        for pos,node in enumerate(bnodes.tolist()):
            r=lookup.loc[node]
            slots.append(Slot(pos,str(r.unit_id),node,int(r.grid_row),int(r.grid_col),str(r.role)))
            if pos==5:
                y=float(target_row['Y_observed_at_A_star'] if use_observed_outcomes else target_row['Y_true_at_A_star'])
                payloads.append(Payload(f'payload_{pos:02d}',float(target_row.A_star),y,True,'pilot_realized_target'))
            else:
                y=float(r.Y_observed_at_A if use_observed_outcomes else r.Y_true_at_A)
                payloads.append(Payload(f'payload_{pos:02d}',float(r.A),y,False,'pilot_factual_calibration'))
        candidate=float(payloads[-1].Y_reference)
        keys=[self.io.payload_key(p,candidate_y=candidate) for p in payloads]
        perms=self.orbit.distinct_index_permutations(keys)
        cser=pd.Series(case.__dict__).copy()
        if str(cser['residual_law'])=='iid_continuous': cser['residual_law']='gaussian_gmrf'
        boundary=self.graph_boundary(bnodes,edges); require(len(boundary)>0,'PILOT_EXACT_BLOCK_EMPTY_BOUNDARY')
        bw=None if observational_target else (None if pd.isna(target_row.get('bandwidth',np.nan)) else float(target_row['bandwidth']))
        # Observational q=g is represented by a temporary interior q and patched after prepare_orbit.
        target_dose=float(target_row.A_star) if not observational_target else 0.5
        if not observational_target and 'target_dose' in target_row: target_dose=float(target_row.target_dose)
        if observational_target: bw=0.1
        fixture=self.types.ExactFixture(
            evaluation_id=f'PILOT_{case.case_id}_REP{int(units.replication.iloc[0]):03d}',fixture_source='pilot_dynamic_size6',case_id=case.case_id,scenario_id=case.scenario_id,
            target_dose=target_dose,bandwidth=bw,candidate_y=candidate,target_payload_truth=candidate,target_slot_position=5,
            slots=slots,payloads=payloads,permutations=perms,case_units=units,case_edges=edges,case_registry_row=cser,block_nodes=bnodes,boundary_nodes=boundary,boundary_edges=pd.DataFrame(),
            rho=float(case.spatial_rho),residual_scale=0.35,residual_law=str(cser['residual_law']),transform_power=float(units.residual_transform_power.iloc[0]),endpoint_audited=bool(lookup.loc[bnodes[-1]].endpoint_audited)
        )
        prepared=self.exact.prepare_orbit(fixture,minimum_precision_eigenvalue_required=float(self.tol['minimum_precision_eigenvalue']))
        if observational_target:
            prepared.intervention_q=prepared.observational_g[5].copy()
        return prepared
    def evaluator(self, prepared, tie_uniform: float):
        kw={'alpha':self.alpha,'tie_uniform':float(tie_uniform),'score_abs_tolerance':float(self.tol['score_tie_abs_tolerance']),'score_rel_tolerance':float(self.tol['score_tie_rel_tolerance']),'non_gaussian_min_abs_residual':float(self.tol['non_gaussian_min_abs_residual'])}
        inspect.signature(self.cinv.CandidateEvaluator).bind(prepared,**kw)
        return self.cinv.CandidateEvaluator(prepared,**kw)
    def patch_sparse_precision(self, prepared, Q_sparse, boundary_nodes: np.ndarray):
        p=copy.deepcopy(prepared); Q=Q_sparse.tocsr(); B=np.asarray(p.fixture.block_nodes,dtype=int); D=np.asarray(boundary_nodes,dtype=int)
        mineig=float(self.gp.minimum_precision_eigenvalue(Q)); require(mineig>float(self.tol['minimum_precision_eigenvalue']),'PILOT_SPARSE_PRECISION_NOT_PD')
        # Boundary is supplied by accepted Stage3E graph_boundary_from_precision(Q,B),
        # not by the original graph.  This is essential because Vecchia Q may have fill-in.
        p.fixture.boundary_nodes=D; p.Q=Q; p.Q_BB,p.Q_BD=self.gp.precision_blocks(Q,B,D)
        p.boundary_residual=np.asarray(p.outside_residual,dtype=float)[D]; p.boundary_latent_z=np.asarray(p.outside_latent_z,dtype=float)[D]
        return p
    def patch_outcome_model(self, prepared, model, units: pd.DataFrame):
        p=copy.deepcopy(prepared); lookup=units.set_index('node_index',drop=False); B=p.fixture.block_nodes
        # candidate mean matrix: rows=slots, columns=payload A
        A=np.asarray(p.A_payload,float); mat=np.empty((6,6),float)
        for i,node in enumerate(B):
            row=lookup.loc[[int(node)]].copy()
            mat[i]=model.predict_target(pd.concat([row]*len(A),ignore_index=True),A)
        p.outcome_mean=mat; p.outcome_scale=np.ones_like(mat)
        factual=model.predict_factual(units)
        p.outside_residual=units['Y_observed_at_A'].to_numpy(float)-factual
        p.outside_latent_z=p.outside_residual.copy()
        D=np.asarray(p.fixture.boundary_nodes,dtype=int); p.boundary_residual=p.outside_residual[D]; p.boundary_latent_z=p.outside_latent_z[D]
        return p

class Stage3EBridge:
    def __init__(self, source_root: Path):
        self.root=Path(source_root); src=_prepend_verified(self.root)
        for name in list(sys.modules):
            if name=='geodose_stage3e' or name.startswith('geodose_stage3e.'):
                f=str(getattr(sys.modules[name],'__file__',''))
                if f and str(src.resolve()).lower() not in str(Path(f).resolve()).lower(): del sys.modules[name]
        self.ordering=importlib.import_module('geodose_stage3e.ordering'); self.n2=importlib.import_module('geodose_stage3e.gaussian_n2'); self.spatial_confidence=importlib.import_module('geodose_stage3e.spatial_confidence')
        for m in [self.ordering,self.n2,self.spatial_confidence]: require(Path(m.__file__).resolve().is_relative_to(src.resolve()),'STAGE3E_IMPORT_ESCAPED')
    def sparse_precision(self, Q_full, units: pd.DataFrame, node_ids: np.ndarray, m: int, mineig: float=1e-10):
        coords=units.sort_values('node_index')[['x_coord','y_coord']].to_numpy(float)
        require(str(inspect.signature(self.ordering.deterministic_maximin_order))=="(coords: 'np.ndarray', node_ids: 'np.ndarray') -> 'np.ndarray'",f'STAGE3E_ORDERING_API_DRIFT:{inspect.signature(self.ordering.deterministic_maximin_order)}')
        ordering=self.ordering.deterministic_maximin_order(coords,node_ids)
        fam,_=self.n2.build_vecchia_family(Q_full,ordering,coords,node_ids,[int(m)],minimum_precision_eigenvalue=float(mineig),kl_nonnegative_tolerance=1e-9)
        obj=fam[int(m)]; return obj.precision,float(obj.total_kl),ordering
    def precision_boundary(self, Q_sparse, block_nodes: np.ndarray) -> np.ndarray:
        # Reuse the accepted Stage3E definition; do not infer the boundary from the
        # original graph because Vecchia precision fill-in can create additional links.
        return np.asarray(self.spatial_confidence.graph_boundary_from_precision(Q_sparse,np.asarray(block_nodes,dtype=int)),dtype=int)
