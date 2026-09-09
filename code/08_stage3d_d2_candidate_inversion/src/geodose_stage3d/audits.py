from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .orbit import assignment_string
from .types import ExactFixture, OrbitEvaluation


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def write_csv_gz(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.17g")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(buffer.getvalue().encode("utf-8"))


def state_audit_frame(fixture: ExactFixture, evaluation: OrbitEvaluation) -> pd.DataFrame:
    target_slot = fixture.target_slot_position
    target_slot_payload = evaluation.target_slot_payload_indices
    payload_A = np.asarray([payload.A for payload in fixture.payloads], dtype=float)
    payload_origin = np.asarray([payload.origin for payload in fixture.payloads], dtype=object)
    return pd.DataFrame(
        {
            "evaluation_id": fixture.evaluation_id,
            "case_id": fixture.case_id,
            "fixture_source": fixture.fixture_source,
            "candidate_y": evaluation.candidate_y,
            "state_index": np.arange(len(evaluation.permutations), dtype=int),
            "assignment": [assignment_string(row) for row in evaluation.permutations],
            "target_payload_position": evaluation.target_payload_positions,
            "target_slot_payload_index": target_slot_payload,
            "target_slot_A": payload_A[target_slot_payload],
            "target_slot_payload_origin": payload_origin[target_slot_payload],
            "log_treatment_factor": evaluation.log_treatment,
            "log_outcome_jacobian_factor": evaluation.log_outcome_jacobian,
            "log_residual_factor": evaluation.log_residual,
            "log_total_weight": evaluation.log_total,
            "normalized_probability": evaluation.probability,
        }
    )


def boundary_audit_frame(fixture: ExactFixture) -> pd.DataFrame:
    frame = fixture.boundary_edges.copy()
    frame.insert(0, "evaluation_id", fixture.evaluation_id)
    frame.insert(1, "case_id", fixture.case_id)
    frame["block_node_count"] = len(fixture.block_nodes)
    frame["boundary_node_count"] = len(fixture.boundary_nodes)
    return frame
