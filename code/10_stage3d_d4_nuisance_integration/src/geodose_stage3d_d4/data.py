from __future__ import annotations

import io
import json
import gzip
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io import D4Error


@dataclass(frozen=True)
class Stage3BExternal:
    cases: pd.DataFrame
    units: pd.DataFrame
    target_draws: pd.DataFrame
    treatment_truth: pd.DataFrame
    true_edges: pd.DataFrame
    fitted_edges: pd.DataFrame
    exact_fixtures: dict
    feature_contract: dict


def _read_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    matches=[n for n in zf.namelist() if n.replace('\\','/').endswith(name)]
    if len(matches)!=1:
        raise D4Error(f"Expected one {name}, found {len(matches)}")
    raw=zf.read(matches[0])
    if name.endswith('.gz'):
        raw=gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw))


def _read_json(zf: zipfile.ZipFile, name: str) -> dict:
    matches=[n for n in zf.namelist() if n.replace('\\','/').endswith(name)]
    if len(matches)!=1:
        raise D4Error(f"Expected one {name}, found {len(matches)}")
    return json.loads(zf.read(matches[0]).decode('utf-8'))


def load_stage3b(path: Path) -> Stage3BExternal:
    with zipfile.ZipFile(path) as zf:
        cases=_read_csv(zf,'stage3b_case_registry.csv')
        units=_read_csv(zf,'stage3b_validation_units.csv.gz')
        targets=_read_csv(zf,'stage3b_realized_target_draws.csv.gz')
        tt=_read_csv(zf,'stage3b_treatment_transport_truth.csv.gz')
        te=_read_csv(zf,'stage3b_true_graph_edges_by_case.csv.gz')
        fe=_read_csv(zf,'stage3b_fitted_graph_edges.csv.gz')
        fixtures=_read_json(zf,'stage3b_exact_orbit_fixtures.json')
        feature=_read_json(zf,'stage3b_method_feature_contract.json')
    required_units={
        'case_id','scenario_id','replication','unit_id','node_index','grid_row','grid_col','role','A',
        'pi_atom_0','pi_atom_1','pi_interior','beta_alpha','beta_beta','outcome_baseline',
        'conditional_mean_at_A','shared_spatial_residual','Y_true_at_A','Y_observed_at_A','endpoint_audited',
        'X_spatial_1','X_spatial_2','X_nonlinear_1','X_nonlinear_2','x_coord','y_coord','residual_latent_gaussian','residual_transform_power'
    }
    if not required_units.issubset(units.columns):
        raise D4Error(f"Stage3B units schema missing {sorted(required_units-set(units.columns))}")
    if units['replication'].nunique()!=1 or int(units['replication'].iloc[0])!=1:
        raise D4Error("D4 source gate is frozen to Stage3B validation replication 1")
    required_targets={'case_id','unit_id','target_dose','A_star','Y_true_at_A_star','Y_observed_at_A_star','conditional_mean_at_A_star','target_supported_by_generator','draw_status','bandwidth','log_oracle_treatment_ratio_at_A_star'}
    if not required_targets.issubset(targets.columns):
        raise D4Error(f"Stage3B target schema missing {sorted(required_targets-set(targets.columns))}")
    return Stage3BExternal(cases,units,targets,tt,te,fe,fixtures,feature)



def frozen_target_unit(case_units: pd.DataFrame, target_node: int=320) -> pd.Series:
    """Return the single frozen D4 target slot (node 320) from one Stage3B case.

    Stage3B intentionally contains 125 ``test_target`` units per case.  D4 exact
    size-6 fixtures use one fixed target slot, node 320, so selecting by role alone
    is ambiguous and scientifically wrong.  This helper freezes the exact target
    identity and fails closed if the role/node contract is not unique.
    """
    required={'node_index','role','unit_id'}
    missing=required-set(case_units.columns)
    if missing:
        raise D4Error(f"Stage3B case units missing target-selector fields {sorted(missing)}")
    x=case_units[(case_units['node_index'].astype(int)==int(target_node)) & (case_units['role'].astype(str)=='test_target')]
    if len(x)!=1:
        raise D4Error(f"Expected exactly one frozen target node {int(target_node)} with role test_target, found {len(x)}")
    return x.iloc[0]

def load_stage3a_seed_registry(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        matches=[n for n in zf.namelist() if n.replace('\\','/').endswith('seed_registry.csv')]
        if len(matches)!=1:
            raise D4Error(f"Expected one seed_registry.csv, found {len(matches)}")
        raw=zf.read(matches[0])
    return pd.read_csv(io.BytesIO(raw))


def case_row(data: Stage3BExternal, case_id: str) -> pd.Series:
    x=data.cases[data.cases['case_id'].astype(str)==str(case_id)]
    if len(x)!=1:
        raise D4Error(f"Expected one Stage3B case row for {case_id}, found {len(x)}")
    return x.iloc[0]


def target_row(data: Stage3BExternal, case_id: str, unit_id: str, target_dose: float) -> pd.Series:
    # Query the registered target first, then distinguish an unsupported/refused
    # target from a genuinely missing/duplicated registry row.  This preserves
    # fail-closed support semantics and makes endpoint-refusal audits explicit.
    x=data.target_draws[
        (data.target_draws['case_id'].astype(str)==str(case_id))
        & (data.target_draws['unit_id'].astype(str)==str(unit_id))
        & (data.target_draws['target_dose'].astype(float).sub(float(target_dose)).abs() <= 1e-12)
    ]
    if len(x)!=1:
        raise D4Error(f"Expected one registered target row for {case_id}/{unit_id}/dose={target_dose}, found {len(x)}")
    row=x.iloc[0]
    supported=bool(row['target_supported_by_generator'])
    drawn=str(row['draw_status'])=='drawn'
    if not supported or not drawn:
        raise D4Error(
            f"Generator does not support requested D4 target: {case_id}/{unit_id}/{target_dose}; "
            f"draw_status={row['draw_status']}; target_supported_by_generator={supported}"
        )
    return row
