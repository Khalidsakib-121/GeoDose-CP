from __future__ import annotations
import importlib, json, sys
from pathlib import Path
from typing import Any
import yaml
from .io import require, Stage3EError

class D4NewDataBridge:
    """Read-only bridge to the accepted D4 v1.1.0 new-data preparation adapter.

    Stage3E reuses D4 only to construct the already-audited six-slot exact PreparedOrbit
    for a new Stage3B case.  It then replaces only the residual precision with the oracle
    sparse/Vecchia precision and delegates candidate evaluation to the accepted D2 G3
    engine through :class:`D2SparseBridge`.
    """
    def __init__(self, d4_source_root: Path, d2_source_root: Path, stage3b_output_zip: Path, alpha: float):
        self.d4_root=Path(d4_source_root); self.d2_root=Path(d2_source_root); self.stage3b_zip=Path(stage3b_output_zip); self.alpha=float(alpha)
        src=self.d4_root/'src'; vendor=self.d4_root/'vendor'
        require(src.is_dir(),f'D4_SOURCE_SRC_MISSING:{src}')
        for path in [src,vendor]:
            s=str(path)
            if path.is_dir() and s not in sys.path: sys.path.insert(0,s)
        # Refuse an accidentally imported different D4 package.
        for name in list(sys.modules):
            if name=='geodose_stage3d_d4' or name.startswith('geodose_stage3d_d4.'):
                f=str(getattr(sys.modules[name],'__file__',''))
                if f and str(src.resolve()).lower() not in str(Path(f).resolve()).lower():
                    del sys.modules[name]
        try:
            self.data_mod=importlib.import_module('geodose_stage3d_d4.data')
            self.fixture_mod=importlib.import_module('geodose_stage3d_d4.fixture')
            self.runtime_mod=importlib.import_module('geodose_stage3d_d4.d2_runtime')
        except Exception as exc:
            raise Stage3EError(f'D4_SOURCE_IMPORT_FAILED:{exc}') from exc
        for m in [self.data_mod,self.fixture_mod,self.runtime_mod]:
            try: Path(m.__file__).resolve().relative_to(src.resolve())
            except Exception as exc: raise Stage3EError(f'D4_IMPORT_ESCAPED_VERIFIED_ROOT:{m.__name__}->{m.__file__}') from exc
        api_lock=json.loads((self.d4_root/'configs'/'d2_api_lock.json').read_text(encoding='utf-8'))
        tolerances=yaml.safe_load((self.d4_root/'configs'/'tolerances.yaml').read_text(encoding='utf-8'))
        require(isinstance(api_lock,dict) and isinstance(tolerances,dict),'D4_CONFIG_SCHEMA_INVALID')
        self.external=self.data_mod.load_stage3b(self.stage3b_zip)
        self.adapter=self.runtime_mod.D2SourceAdapter(self.d2_root,self.alpha,tolerances,api_lock)
        self.api_record={
            'd4_source_root':str(self.d4_root),
            'data_module':str(self.data_mod.__file__),
            'fixture_module':str(self.fixture_mod.__file__),
            'runtime_module':str(self.runtime_mod.__file__),
            'd2_api_preflight':self.adapter.api.get('preflight'),
        }

    def prepare_oracle_true_graph(self, case_id: str, target_dose: float) -> tuple[Any,dict[str,Any]]:
        case_id=str(case_id); dose=float(target_dose)
        crow=self.data_mod.case_row(self.external,case_id)
        block=tuple(self.fixture_mod.choose_block(self.external,case_id,'frozen_stage3b_size6_coordinates'))
        # Reuse the accepted D4 preparation logic but replace the D2-embedded target-draw
        # table with the currently accepted external Stage3B target table. D4 v1.1.0 did not
        # need this override for its registered smoke cases; Stage3E includes additional
        # cases (notably S8) and must not fall back to a stale/duplicated embedded target row.
        custom,meta=self.adapter._make_custom_data(
            self.external,case_id,dose,block,treatment_mode='OT',graph_mode='TG',fit=None,
        )
        target_field=self.adapter.fields.get('targets')
        require(target_field is not None,'D4_D2_DATA_TARGET_FIELD_NOT_DISCOVERED')
        # Validate the exact frozen target unit/draw through the accepted D4 Stage3B reader.
        cu=self.external.units[self.external.units.case_id.astype(str)==case_id].copy()
        target_unit=self.data_mod.frozen_target_unit(cu,320)
        target_row=self.data_mod.target_row(self.external,case_id,str(target_unit['unit_id']),dose)
        targets=self.external.target_draws.copy()
        sel=(targets.case_id.astype(str)==case_id) & (targets.unit_id.astype(str)==str(target_unit['unit_id'])) & (targets.target_dose.astype(float).sub(dose).abs()<=1e-12)
        require(int(sel.sum())==1,f'STAGE3E_EXTERNAL_TARGET_ROW_NOT_UNIQUE:{case_id}:{dose}')
        original_scope=str(targets.loc[sel,'target_scope'].iloc[0])
        # Accepted D2's generic new-fixture builder filters to target_scope='registered_grid'.
        # Stage3B S8 uses the equally frozen label 'scenario_primary_extra' for dose 0.98.
        # Relabel only this *local clone* after the D4 support/draw validation above; A*, Y*,
        # bandwidth and all scientific truth values are unchanged. Upstream Stage3B is untouched.
        scope_alias_applied=original_scope!='registered_grid'
        if scope_alias_applied: targets.loc[sel,'target_scope']='registered_grid'
        custom=self.fixture_mod.clone_data_object(custom,{target_field:targets})
        prepared,bmeta=self.adapter._build_prepared(custom,case_id,dose)
        return prepared,{
            'case_id':case_id,'scenario_id':str(crow['scenario_id']),'target_dose':dose,
            'block_nodes':list(map(int,block)),'treatment_mode':'OT','graph_mode':'TG',
            'stage3e_external_target_table_override':True,'target_field_name':str(target_field),'target_scope_original':original_scope,'target_scope_local_D2_alias_applied':bool(scope_alias_applied),
            **meta,**bmeta,
        }
