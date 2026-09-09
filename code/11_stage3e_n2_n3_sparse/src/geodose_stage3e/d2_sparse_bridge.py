from __future__ import annotations
import copy, importlib, inspect, json, math, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from .io import require, Stage3EError
from .spatial_confidence import graph_boundary_from_precision

class D2SparseBridge:
    """Read-only bridge to the byte-verified accepted D2 G1/G2/G3 engine.

    Stage3E changes only the residual reference precision supplied to the
    already-accepted PreparedOrbit. It does not reimplement orbit algebra,
    treatment factors, Jacobians, score partitioning, tie randomization, or G3.
    """
    def __init__(self, source_root: Path, api_lock_path: Path, alpha: float):
        self.root=Path(source_root); self.alpha=float(alpha)
        src=self.root/'src'; require(src.is_dir(),f'D2_SOURCE_SRC_MISSING:{src}')
        s=str(src)
        if s not in sys.path: sys.path.insert(0,s)
        for name in list(sys.modules):
            if name=='geodose_stage3d' or name.startswith('geodose_stage3d.'):
                f=str(getattr(sys.modules[name],'__file__',''))
                if f and s.lower() not in f.lower(): del sys.modules[name]
        self.io=importlib.import_module('geodose_stage3d.d2_io')
        self.exact=importlib.import_module('geodose_stage3d.exact_law')
        self.cinv=importlib.import_module('geodose_stage3d.candidate_inversion')
        self.gp=importlib.import_module('geodose_stage3d.graph_precision')
        self.types=importlib.import_module('geodose_stage3d.types')
        for m in [self.io,self.exact,self.cinv,self.gp,self.types]:
            try: Path(m.__file__).resolve().relative_to(src.resolve())
            except Exception as exc: raise Stage3EError(f'D2_IMPORT_ESCAPED_VERIFIED_ROOT:{m.__name__}->{m.__file__}') from exc
        self.lock=json.loads(Path(api_lock_path).read_text(encoding='utf-8'))
        self.tol=yaml.safe_load((self.root/'configs'/'stage3d_d2_tolerances.yaml').read_text(encoding='utf-8'))
        self._validate_api()
        self.data=self.io.stage3b_data(self.root)

    def _validate_api(self) -> None:
        sigs={
            'stage3b_data':str(inspect.signature(self.io.stage3b_data)),
            'load_d1_fixture_config':str(inspect.signature(self.io.load_d1_fixture_config)),
            'build_d2_fixture':str(inspect.signature(self.io.build_d2_fixture)),
            'frozen_orbit_seed':str(inspect.signature(self.io.frozen_orbit_seed)),
            'derive_seed':str(inspect.signature(self.io.derive_seed)),
            'prepare_orbit':str(inspect.signature(self.exact.prepare_orbit)),
            'CandidateEvaluator':str(inspect.signature(self.cinv.CandidateEvaluator)),
            'CandidateEvaluator.evaluate':str(inspect.signature(self.cinv.CandidateEvaluator.evaluate)),
            'acceptance':str(inspect.signature(self.cinv.acceptance)),
        }
        expected=self.lock['signatures']; bad={k:(expected.get(k),v) for k,v in sigs.items() if expected.get(k)!=v}
        require(not bad,f'D2_API_LOCK_MISMATCH:{bad}')
        for k,v in self.lock['frozen_tolerances'].items():
            require(k in self.tol and float(self.tol[k])==float(v),f'D2_TOLERANCE_LOCK_MISMATCH:{k}')
        require(self.alpha==float(self.lock['alpha_primary']),'D2_ALPHA_LOCK_MISMATCH')
        self.api_record={'signatures':sigs,'frozen_tolerances':{k:self.tol[k] for k in self.lock['frozen_tolerances']}}

    def _tie_uniform(self, fixture_id: str) -> tuple[float,dict[str,int|str|float]]:
        label=str(self.lock['randomization_contract']['derived_subseed_label'])
        base=int(self.io.frozen_orbit_seed(self.root,'S4',1)); derived=int(self.io.derive_seed(base,label,str(fixture_id)))
        u=float(np.random.default_rng(derived).random())
        return u,{'orbit_seed':base,'derived_tie_seed':derived,'tie_fixture_id':str(fixture_id),'tie_uniform':u}

    def _evaluator(self, prepared: Any, fixture_id: str):
        u,seed=self._tie_uniform(fixture_id)
        kwargs={'alpha':self.alpha,'tie_uniform':u,'score_abs_tolerance':float(self.tol['score_tie_abs_tolerance']),'score_rel_tolerance':float(self.tol['score_tie_rel_tolerance']),'non_gaussian_min_abs_residual':float(self.tol['non_gaussian_min_abs_residual'])}
        inspect.signature(self.cinv.CandidateEvaluator).bind(prepared,**kwargs)
        return self.cinv.CandidateEvaluator(prepared,**kwargs),seed

    def accepted_exact_prepared(self) -> Any:
        fixture=self.io.build_d2_fixture(self.data,self.root,'G6_GAUSS_INTERIOR_TRUTH')
        return self.exact.prepare_orbit(fixture,minimum_precision_eigenvalue_required=float(self.tol['minimum_precision_eigenvalue']))

    def patch_precision(self, prepared: Any, Q_sparse) -> tuple[Any,dict[str,object]]:
        p=copy.deepcopy(prepared); Q=Q_sparse.tocsr(); B=np.asarray(p.fixture.block_nodes,dtype=int)
        mineig=float(self.gp.minimum_precision_eigenvalue(Q)); require(mineig>float(self.tol['minimum_precision_eigenvalue']),f'SPARSE_D2_PRECISION_NOT_PD:{mineig}')
        D=graph_boundary_from_precision(Q,B)
        p.Q=Q
        p.Q_BB,p.Q_BD=self.gp.precision_blocks(Q,B,D)
        p.fixture.boundary_nodes=D
        # CandidateEvaluator only consumes the prepared block matrices and boundary residual.
        p.boundary_residual=np.asarray(p.outside_residual,dtype=float)[D]
        p.boundary_latent_z=np.asarray(p.outside_latent_z,dtype=float)[D]
        return p,{'block_nodes':B.tolist(),'boundary_nodes':D.tolist(),'boundary_size':len(D),'minimum_precision_eigenvalue':mineig,'q_nnz':int(Q.nnz)}

    def evaluate(self, prepared: Any, fixture_id: str, candidates: list[float]) -> tuple[pd.DataFrame,dict[str,object]]:
        ev,seed=self._evaluator(prepared,fixture_id); rows=[]
        for y in candidates:
            r=ev.evaluate(float(y))
            rows.append({'candidate_y':float(y),'candidate_hex':float(y).hex(),'conservative_p':float(r.conservative_p),'randomized_p':float(r.randomized_p),'conservative_accept':bool(self.cinv.acceptance(r,'conservative')),'randomized_accept':bool(self.cinv.acceptance(r,'randomized')),'distinct_states':int(r.distinct_states),'tie_uniform':float(r.tie_uniform)})
        return pd.DataFrame(rows),seed

    def full_invert(self, prepared: Any, fixture_id: str, *, independent_grid_audit: bool = True) -> dict[str,Any]:
        """Run the accepted D2 finite-domain G3 inversion on a supplied PreparedOrbit.

        Stage3E does not rewrite numerical inversion.  It calls the accepted D2
        landmark/grid, boundary refinement, component extraction, and acceptance
        routines using the byte-verified D2 candidate contract and tolerances.
        """
        cc=yaml.safe_load((self.root/'configs'/'stage3d_d2_candidate_contract.yaml').read_text(encoding='utf-8'))
        require(isinstance(cc,dict),'D2_CANDIDATE_CONTRACT_SCHEMA_INVALID')
        domain=cc['candidate_domain']; lower=float(domain['lower']); upper=float(domain['upper'])
        landmarks=self.cinv.candidate_landmarks(prepared,lower,upper)
        initial,source_map=self.cinv.initial_candidate_points(
            lower,upper,landmarks,
            coarse_points=int(cc['initial_grid']['coarse_points']),
            staggered_points=int(cc['initial_grid']['staggered_midpoints']),
            certification_points=int(cc['initial_grid']['certification_grid_points']),
            flank_relative=float(cc['initial_grid']['landmark_flank_relative_to_domain']),
            flank_minimum=float(cc['initial_grid']['landmark_flank_minimum']),
        )
        evaluator,seed=self._evaluator(prepared,fixture_id)
        for y in initial.tolist(): evaluator.evaluate(float(y))
        boundaries={}
        boundary_rows=[]
        for mode in ('randomized','conservative'):
            bs=self.cinv.refine_boundaries(
                evaluator,initial,mode=mode,
                absolute_tolerance=float(self.tol['boundary_absolute_tolerance']),
                relative_tolerance=float(self.tol['boundary_relative_tolerance']),
                maximum_depth=int(self.tol['boundary_maximum_depth']),
            )
            boundaries[mode]=bs
            for idx,b in enumerate(bs,1): boundary_rows.append({'tie_mode':mode,'boundary_id':idx,**b.__dict__})
        exact_points=set(float(v) for v in landmarks.get('transform_singularity',[]))
        comps={mode:self.cinv.outer_components_from_cache(evaluator,lower,upper,mode=mode,certified_exact_points=exact_points) for mode in ('randomized','conservative')}
        comps['randomized']=self.cinv.intersect_component_sets(comps['randomized'],comps['conservative'],tie_mode='randomized')
        for r in evaluator.cached_results():
            require(not r.randomized_accept or r.conservative_accept,'D2_RANDOMIZED_NOT_SUBSET_CONSERVATIVE')
        summary={mode:self.cinv.summarize_components(comps[mode],lower,upper) for mode in ('randomized','conservative')}
        # Independent offset grid: no accepted point may be excluded by the returned outer set.
        audit={'performed':bool(independent_grid_audit),'points':0,'conservative_accepted_points':0,'conservative_false_negatives':0,'randomized_accepted_points':0,'randomized_false_negatives':0}
        if independent_grid_audit:
            n=int(self.tol['independent_grid_points']); step=(upper-lower)/n
            grid=lower+(np.arange(n,dtype=float)+0.5)*step
            fnc=fnr=accc=accr=0
            for y in grid:
                r=evaluator.evaluate(float(y))
                ac=bool(self.cinv.acceptance(r,'conservative')); ar=bool(self.cinv.acceptance(r,'randomized'))
                if ac:
                    accc+=1
                    if not self.cinv.point_in_components(float(y),comps['conservative']): fnc+=1
                if ar:
                    accr+=1
                    if not self.cinv.point_in_components(float(y),comps['randomized']): fnr+=1
            require(fnc==0 and fnr==0,f'STAGE3E_SPARSE_INVERSION_FALSE_NEGATIVES:conservative={fnc},randomized={fnr}')
            audit={'performed':True,'points':n,'conservative_accepted_points':accc,'conservative_false_negatives':fnc,'randomized_accepted_points':accr,'randomized_false_negatives':fnr}
        trace=[]
        for r in evaluator.cached_results():
            trace.append({'candidate_y':float(r.candidate_y),'candidate_hex':float(r.candidate_y).hex(),'conservative_p':float(r.conservative_p),'randomized_p':float(r.randomized_p),'conservative_accept':bool(r.conservative_accept),'randomized_accept':bool(r.randomized_accept),'distinct_states':int(r.distinct_states),'tie_uniform':float(r.tie_uniform)})
        component_rows=[]
        for mode in ('randomized','conservative'):
            for c in comps[mode]: component_rows.append({'tie_mode':mode,**c})
        return {'fixture_id':str(fixture_id),'domain_lower':lower,'domain_upper':upper,'seed':seed,'landmarks':landmarks,'initial_unique_points':int(len(initial)),'candidate_evaluation_count':int(len(evaluator.cached_results())),'summaries':summary,'components':component_rows,'boundaries':boundary_rows,'independent_grid_audit':audit,'trace':trace,'returned_set_representation':'coverage_preserving_outer_numerical_set_intersect_registered_domain','full_real_line_claim':False}
