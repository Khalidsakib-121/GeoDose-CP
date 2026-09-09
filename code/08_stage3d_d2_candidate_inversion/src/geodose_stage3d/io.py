from __future__ import annotations

import gzip
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .provenance import sha256_file
from .types import ExactFixture, Payload, Slot


class Stage3DError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage3DError(message)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    require(isinstance(value, dict), f"YAML root must be a mapping: {path}")
    return value


def verify_input_hashes(package_root: str | Path) -> dict[str, dict[str, str]]:
    root = Path(package_root)
    expected = json.loads((root / "configs" / "input_hashes.json").read_text(encoding="utf-8"))
    location_by_name = {
        "GeoDose_Stage3A_OUTPUTS.zip": root / "inputs" / "upstream" / "GeoDose_Stage3A_OUTPUTS.zip",
        "GeoDose_Stage3B_Controlled_Generator_v1_2_FREEZE.zip": root / "inputs" / "upstream" / "GeoDose_Stage3B_Controlled_Generator_v1_2_FREEZE.zip",
        "GeoDose_Stage3B_OUTPUTS.zip": root / "inputs" / "upstream" / "GeoDose_Stage3B_OUTPUTS.zip",
        "GeoDose_Stage3C_GateC1_v1_1_1_EXACT_FREEZE.zip": root / "inputs" / "upstream" / "GeoDose_Stage3C_GateC1_v1_1_1_EXACT_FREEZE.zip",
        "GeoDose_Stage3C_REFERENCE_OUTPUTS.zip": root / "inputs" / "upstream" / "GeoDose_Stage3C_REFERENCE_OUTPUTS.zip",
        "Unified math.docx": root / "inputs" / "math" / "Unified math.docx",
        "Concise math.docx": root / "inputs" / "math" / "Concise math.docx",
        "proposal.docx": root / "inputs" / "math" / "proposal.docx",
        "files_updated_aligned.zip": root / "inputs" / "reference" / "files_updated_aligned.zip",
    }
    audit: dict[str, dict[str, str]] = {}
    for name, expected_hash in expected.items():
        path = location_by_name[name]
        require(path.is_file(), f"D1_INPUT_HASH_MISMATCH: missing {name}")
        observed = sha256_file(path)
        require(observed == expected_hash, f"D1_INPUT_HASH_MISMATCH: {name} expected {expected_hash}, got {observed}")
        audit[name] = {"path": str(path.relative_to(root)), "expected_sha256": expected_hash, "observed_sha256": observed}
    return audit


class Stage3BArchive:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        require(self.path.is_file(), f"Missing Stage 3B archive: {self.path}")
        self._zip = zipfile.ZipFile(self.path)
        bad = self._zip.testzip()
        require(bad is None, f"Stage 3B archive CRC failure: {bad}")

    def close(self) -> None:
        self._zip.close()

    def read_json(self, name: str) -> Any:
        return json.loads(self._zip.read(name).decode("utf-8"))

    def read_csv(self, name: str) -> pd.DataFrame:
        raw = self._zip.read(name)
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        return pd.read_csv(io.BytesIO(raw))


class Stage3BData:
    def __init__(self, archive_path: str | Path):
        archive = Stage3BArchive(archive_path)
        try:
            self.case_registry = archive.read_csv("stage3b_case_registry.csv")
            self.validation_units = archive.read_csv("stage3b_validation_units.csv.gz")
            self.realized_targets = archive.read_csv("stage3b_realized_target_draws.csv.gz")
            self.true_edges = archive.read_csv("stage3b_true_graph_edges_by_case.csv.gz")
            self.fixtures = archive.read_json("stage3b_exact_orbit_fixtures.json")
            self.generator_contract = archive.read_json("stage3b_generator_contract.json")
            self.verification = archive.read_json("STAGE3B_VERIFICATION.json")
        finally:
            archive.close()
        require(self.verification.get("status") == "verified_complete", "Stage 3B input is not verified complete")
        require(self.generator_contract.get("methods_implemented") is False, "Unexpected Stage 3B method status")

    def case_row(self, case_id: str) -> pd.Series:
        rows = self.case_registry.loc[self.case_registry["case_id"] == case_id]
        require(len(rows) == 1, f"Case registry row not unique: {case_id}")
        return rows.iloc[0]

    def case_units(self, case_id: str) -> pd.DataFrame:
        frame = self.validation_units.loc[self.validation_units["case_id"] == case_id].copy()
        require(len(frame) > 0, f"No units for case {case_id}")
        frame = frame.sort_values("node_index").reset_index(drop=True)
        expected = np.arange(len(frame), dtype=int)
        require(np.array_equal(frame["node_index"].to_numpy(dtype=int), expected), f"Node index gap for {case_id}")
        return frame

    def case_edges(self, case_id: str) -> pd.DataFrame:
        frame = self.true_edges.loc[self.true_edges["case_id"] == case_id, ["source_node", "target_node"]].copy()
        frame = frame.astype(int).sort_values(["source_node", "target_node"]).drop_duplicates().reset_index(drop=True)
        require(len(frame) > 0, f"No true graph edges for case {case_id}")
        return frame

    def target_row(self, case_id: str, node_index: int, target_dose: float) -> pd.Series:
        units = self.case_units(case_id)
        unit_rows = units.loc[units["node_index"] == int(node_index)]
        require(len(unit_rows) == 1, f"Target unit not unique: {case_id}/{node_index}")
        unit_id = str(unit_rows.iloc[0]["unit_id"])
        rows = self.realized_targets.loc[
            (self.realized_targets["case_id"] == case_id)
            & (self.realized_targets["unit_id"] == unit_id)
            & np.isclose(self.realized_targets["target_dose"].astype(float), float(target_dose), atol=1e-12, rtol=0.0)
            & (self.realized_targets["target_scope"] == "registered_grid")
        ]
        require(len(rows) == 1, f"Target draw not unique: {case_id}/{unit_id}/{target_dose}")
        return rows.iloc[0]


def payload_key(payload: Payload, *, candidate_y: float | None = None) -> tuple[float, float]:
    """Return the theorem-level movable-payload key ``(A,Y)``.

    The candidate source index is tracked only so its response can be replaced
    before an orbit evaluation.  Source origin (target versus calibration) is
    *not* part of payload equality under G1.  Consequently, when a candidate
    pair numerically coincides with a calibration pair, the quotient orbit must
    collapse those duplicate states.
    """
    y_value = float(candidate_y) if payload.is_target_payload and candidate_y is not None else float(payload.Y_reference)
    return (float(payload.A), y_value)


def distinct_state_count(payloads: list[Payload], *, candidate_y: float | None = None) -> int:
    counts = Counter(payload_key(p, candidate_y=candidate_y) for p in payloads)
    value = math.factorial(len(payloads))
    for count in counts.values():
        value //= math.factorial(count)
    return value


def _slots_from_fixture(raw: dict[str, Any], case_units: pd.DataFrame) -> list[Slot]:
    slots: list[Slot] = []
    for position, item in enumerate(raw["fixed_slots"]):
        node = int(item["node_index"])
        unit_row = case_units.loc[case_units["node_index"] == node]
        require(len(unit_row) == 1, f"Fixed slot node missing: {node}")
        row = unit_row.iloc[0]
        require(str(row["role"]) == str(item["role"]), f"Fixed slot role mismatch at node {node}")
        slots.append(
            Slot(
                slot_position=position,
                unit_id=str(row["unit_id"]),
                node_index=node,
                grid_row=int(row["grid_row"]),
                grid_col=int(row["grid_col"]),
                role=str(item["role"]),
            )
        )
    return slots


def _source_payloads(raw: dict[str, Any]) -> list[Payload]:
    result: list[Payload] = []
    target_count = 0
    for idx, item in enumerate(raw["movable_payloads"]):
        origin = str(item["payload_origin"])
        is_target = origin == "realized_localized_target_truth_reference"
        target_count += int(is_target)
        result.append(
            Payload(
                payload_id=f"payload_{idx:02d}",
                A=float(item["A"]),
                Y_reference=float(item["Y"]),
                is_target_payload=is_target,
                origin=origin,
            )
        )
    require(target_count == 1, "D1_TARGET_SLOT_INVALID: fixture must have one target payload")
    return result


def _derived_payloads(
    data: Stage3BData,
    raw_base: dict[str, Any],
    case_id: str,
    target_dose: float,
) -> tuple[list[Payload], float, float | None]:
    case_units = data.case_units(case_id)
    target_position = int(raw_base["target_slot_position"])
    payloads: list[Payload] = []
    for idx, fixed in enumerate(raw_base["fixed_slots"]):
        node = int(fixed["node_index"])
        unit = case_units.loc[case_units["node_index"] == node].iloc[0]
        if idx == target_position:
            target = data.target_row(case_id, node, target_dose)
            payloads.append(
                Payload(
                    payload_id=f"payload_{idx:02d}",
                    A=float(target["A_star"]),
                    Y_reference=float(target["Y_true_at_A_star"]),
                    is_target_payload=True,
                    origin="realized_localized_target_truth_reference",
                )
            )
            target_truth = float(target["Y_true_at_A_star"])
            bandwidth = None if pd.isna(target["bandwidth"]) else float(target["bandwidth"])
        else:
            payloads.append(
                Payload(
                    payload_id=f"payload_{idx:02d}",
                    A=float(unit["A"]),
                    Y_reference=float(unit["Y_true_at_A"]),
                    is_target_payload=False,
                    origin="factual_calibration_observation",
                )
            )
    return payloads, target_truth, bandwidth


def graph_boundary(block_nodes: np.ndarray, edges: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    block = set(int(x) for x in block_nodes.tolist())
    boundary: set[int] = set()
    records: list[tuple[int, int]] = []
    for row in edges.itertuples(index=False):
        source, target = int(row.source_node), int(row.target_node)
        source_in, target_in = source in block, target in block
        if source_in ^ target_in:
            outside = target if source_in else source
            boundary.add(outside)
            records.append((source, target))
    frame = pd.DataFrame(sorted(set(records)), columns=["source_node", "target_node"])
    return np.asarray(sorted(boundary), dtype=int), frame


def build_fixture(
    data: Stage3BData,
    evaluation: dict[str, Any],
    *,
    candidate_shift: float,
) -> ExactFixture:
    evaluation_id = str(evaluation["evaluation_id"])
    fixture_source = str(evaluation["fixture_source"])
    case_id = str(evaluation["case_id"])
    target_dose = float(evaluation["target_dose"])
    base_name = fixture_source.replace("_derived", "")
    require(base_name in data.fixtures, f"Unknown exact fixture: {base_name}")
    raw = data.fixtures[base_name]
    case_units = data.case_units(case_id)
    case_edges = data.case_edges(case_id)
    case_row = data.case_row(case_id)
    slots = _slots_from_fixture(raw, case_units)
    target_slot = int(raw["target_slot_position"])
    require(sum(slot.role == "test_target" for slot in slots) == 1, "D1_TARGET_SLOT_INVALID: fixed target role count")
    require(slots[target_slot].role == "test_target", "D1_TARGET_SLOT_INVALID: target position role mismatch")

    if fixture_source.endswith("_derived"):
        payloads, target_truth, bandwidth = _derived_payloads(data, raw, case_id, target_dose)
    else:
        payloads = _source_payloads(raw)
        target_truth = float(raw["target_truth_reference"]["Y_true_at_A_star"])
        bandwidth = None if raw["target_truth_reference"].get("bandwidth") is None else float(raw["target_truth_reference"]["bandwidth"])

    require(len(slots) == len(payloads), "Slot/payload length mismatch")
    require(sum(p.is_target_payload for p in payloads) == 1, "D1_TARGET_SLOT_INVALID: target payload count")

    block_nodes = np.asarray([slot.node_index for slot in slots], dtype=int)
    boundary_nodes, boundary_edges = graph_boundary(block_nodes, case_edges)
    require(len(boundary_nodes) > 0, "D1_BOUNDARY_PAYLOAD_MISSING: empty boundary")

    # Stage 3B exact fixture graph identities are frozen and must reproduce from
    # the accepted true graph, not merely look plausible.
    raw_internal = {tuple(sorted((int(edge["source_node"]), int(edge["target_node"])))) for edge in raw["internal_edges"]}
    computed_internal = {
        tuple(sorted((int(row.source_node), int(row.target_node))))
        for row in case_edges.itertuples(index=False)
        if int(row.source_node) in set(block_nodes.tolist()) and int(row.target_node) in set(block_nodes.tolist())
    }
    require(raw_internal == computed_internal, f"Frozen internal-edge mismatch for {base_name}/{case_id}")
    raw_boundary = {tuple(sorted((int(edge["source_node"]), int(edge["target_node"])))) for edge in raw["boundary_edges"]}
    computed_boundary = {tuple(sorted((int(row.source_node), int(row.target_node)))) for row in boundary_edges.itertuples(index=False)}
    require(raw_boundary == computed_boundary, f"Frozen boundary-edge mismatch for {base_name}/{case_id}")

    # Unique source fixtures must agree exactly with accepted Stage 3B factual
    # and realized-target tables. The deliberate duplicate fixture is a
    # quotient-orbit stress object and is exempt from slotwise factual identity.
    if fixture_source in {"size6_unique", "size8_unique"}:
        for index, payload in enumerate(payloads):
            node = slots[index].node_index
            unit = case_units.loc[case_units["node_index"] == node].iloc[0]
            if payload.is_target_payload:
                target = data.target_row(case_id, node, target_dose)
                require(abs(payload.A - float(target["A_star"])) <= 1e-14, "Frozen target A_star mismatch")
                require(abs(payload.Y_reference - float(target["Y_true_at_A_star"])) <= 1e-14, "Frozen target truth mismatch")
            else:
                require(abs(payload.A - float(unit["A"])) <= 1e-14, "Frozen calibration treatment mismatch")
                require(abs(payload.Y_reference - float(unit["Y_true_at_A"])) <= 1e-14, "Frozen calibration outcome mismatch")
    endpoint_audited = bool(case_units.loc[case_units["node_index"] == slots[target_slot].node_index, "endpoint_audited"].iloc[0])

    candidate_rule = str(evaluation["candidate_rule"])
    if candidate_rule == "stored_simulation_truth":
        candidate_y = target_truth
    elif candidate_rule == "stored_simulation_truth_plus_0.75":
        candidate_y = target_truth + float(candidate_shift)
    else:
        raise Stage3DError(f"Unknown candidate rule: {candidate_rule}")

    # G1 quotients by the numerical movable payloads (A,Y), not by their
    # source labels.  Because the target response is replaced by the candidate
    # before conditioning on the augmented orbit, the distinct state space is
    # candidate-specific at exact collision values.
    from .orbit import distinct_index_permutations

    candidate_keys = [payload_key(payload, candidate_y=float(candidate_y)) for payload in payloads]
    permutations = distinct_index_permutations(candidate_keys)
    expected_states = distinct_state_count(payloads, candidate_y=float(candidate_y))
    require(expected_states == len(permutations), f"D1_ENUMERATION_COUNT_MISMATCH: expected {expected_states}, got {len(permutations)}")

    transform_power = float(case_units["residual_transform_power"].iloc[0])
    require(np.allclose(case_units["residual_transform_power"].astype(float), transform_power), f"Residual transform power varies in {case_id}")
    return ExactFixture(
        evaluation_id=evaluation_id,
        fixture_source=fixture_source,
        case_id=case_id,
        scenario_id=str(case_row["scenario_id"]),
        target_dose=target_dose,
        bandwidth=bandwidth,
        candidate_y=float(candidate_y),
        target_payload_truth=float(target_truth),
        target_slot_position=target_slot,
        slots=slots,
        payloads=payloads,
        permutations=permutations,
        case_units=case_units,
        case_edges=case_edges,
        case_registry_row=case_row,
        block_nodes=block_nodes,
        boundary_nodes=boundary_nodes,
        boundary_edges=boundary_edges,
        rho=float(case_row["spatial_rho"]),
        residual_scale=0.35,
        residual_law=str(case_row["residual_law"]),
        transform_power=transform_power,
        endpoint_audited=endpoint_audited,
    )
