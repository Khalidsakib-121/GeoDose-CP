from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class Payload:
    payload_id: str
    A: float
    Y_reference: float
    is_target_payload: bool
    origin: str


@dataclass(frozen=True)
class Slot:
    slot_position: int
    unit_id: str
    node_index: int
    grid_row: int
    grid_col: int
    role: str


@dataclass
class ExactFixture:
    evaluation_id: str
    fixture_source: str
    case_id: str
    scenario_id: str
    target_dose: float
    bandwidth: float | None
    candidate_y: float
    target_payload_truth: float
    target_slot_position: int
    slots: list[Slot]
    payloads: list[Payload]
    permutations: np.ndarray
    case_units: pd.DataFrame
    case_edges: pd.DataFrame
    case_registry_row: pd.Series
    block_nodes: np.ndarray
    boundary_nodes: np.ndarray
    boundary_edges: pd.DataFrame
    rho: float
    residual_scale: float
    residual_law: str
    transform_power: float
    endpoint_audited: bool


@dataclass
class PreparedOrbit:
    fixture: ExactFixture
    slot_unit_rows: pd.DataFrame
    A_payload: np.ndarray
    Y_payload_reference: np.ndarray
    target_payload_index: int
    outcome_mean: np.ndarray
    outcome_scale: np.ndarray
    observational_g: np.ndarray
    intervention_q: np.ndarray
    Q: sparse.csr_matrix
    Q_BB: np.ndarray
    Q_BD: np.ndarray
    outside_residual: np.ndarray
    outside_latent_z: np.ndarray
    boundary_residual: np.ndarray
    boundary_latent_z: np.ndarray


@dataclass
class OrbitEvaluation:
    evaluation_id: str
    candidate_y: float
    permutations: np.ndarray
    log_treatment: np.ndarray
    log_outcome_jacobian: np.ndarray
    log_residual: np.ndarray
    log_total: np.ndarray
    probability: np.ndarray
    residual_matrix: np.ndarray
    latent_matrix: np.ndarray | None
    target_payload_positions: np.ndarray
    target_slot_payload_indices: np.ndarray
    diagnostics: dict[str, Any]
