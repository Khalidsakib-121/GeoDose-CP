from __future__ import annotations

import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import Stage3BData, Stage3DError, build_fixture, read_yaml, require
from .provenance import sha256_file


D1_SOURCE_PREFIX = "GeoDose_Stage3D_D0_D1_ExactOrbit_Algebra_v1_1_FREEZE/"


def verify_d2_inputs(package_root: str | Path) -> dict[str, dict[str, str]]:
    root = Path(package_root)
    expected = json.loads((root / "configs" / "stage3d_d2_input_hashes.json").read_text(encoding="utf-8"))
    locations = {
        "GeoDose_Stage3A_OUTPUTS.zip": root / "inputs" / "upstream" / "GeoDose_Stage3A_OUTPUTS.zip",
        "GeoDose_Stage3B_OUTPUTS.zip": root / "inputs" / "upstream" / "GeoDose_Stage3B_OUTPUTS.zip",
        "GeoDose_Stage3D_D0_D1_ExactOrbit_Algebra_v1_1_FREEZE.zip": root / "inputs" / "upstream" / "GeoDose_Stage3D_D0_D1_ExactOrbit_Algebra_v1_1_FREEZE.zip",
        "GeoDose_Stage3D_D1_OUTPUTS_ACCEPTED.zip": root / "inputs" / "upstream" / "GeoDose_Stage3D_D1_OUTPUTS_ACCEPTED.zip",
        "Unified math.docx": root / "inputs" / "math" / "Unified math.docx",
        "Concise math.docx": root / "inputs" / "math" / "Concise math.docx",
        "proposal.docx": root / "inputs" / "math" / "proposal.docx",
    }
    audit: dict[str, dict[str, str]] = {}
    for name, expected_hash in expected.items():
        path = locations[name]
        require(path.is_file(), f"D2_INPUT_HASH_MISMATCH: missing {name}")
        observed = sha256_file(path)
        require(observed == expected_hash, f"D2_INPUT_HASH_MISMATCH: {name}")
        audit[name] = {
            "path": path.relative_to(root).as_posix(),
            "expected_sha256": expected_hash,
            "observed_sha256": observed,
        }
    _verify_d1_source_inheritance(root)
    _verify_d1_output(root)
    return audit


def _verify_d1_source_inheritance(root: Path) -> None:
    archive_path = root / "inputs" / "upstream" / "GeoDose_Stage3D_D0_D1_ExactOrbit_Algebra_v1_1_FREEZE.zip"
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        require(bad is None, f"D2_D1_SOURCE_MISMATCH: CRC {bad}")
        for local in sorted((root / "src" / "geodose_stage3d").glob("*.py")):
            if local.name.startswith("d2_") or local.name in {"candidate_inversion.py", "topology_validation.py"}:
                continue
            member = D1_SOURCE_PREFIX + "src/geodose_stage3d/" + local.name
            require(member in archive.namelist(), f"D2_D1_SOURCE_MISMATCH: missing {member}")
            expected = hashlib.sha256(archive.read(member)).hexdigest()
            observed = sha256_file(local)
            require(expected == observed, f"D2_D1_SOURCE_MISMATCH: {local.name}")


def _verify_d1_output(root: Path) -> None:
    path = root / "inputs" / "upstream" / "GeoDose_Stage3D_D1_OUTPUTS_ACCEPTED.zip"
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        require(bad is None, f"D2_D1_OUTPUT_NOT_VERIFIED: CRC {bad}")
        verification = json.loads(archive.read("STAGE3D_D1_VERIFICATION.json").decode("utf-8"))
        require(verification.get("status") == "verified_complete", "D2_D1_OUTPUT_NOT_VERIFIED")
        require(int(verification.get("verification_check_count", verification.get("check_count", 69))) == 69 or len(verification.get("checks", {})) == 69,
                "D2_D1_OUTPUT_NOT_VERIFIED: unexpected check count")
        require(verification.get("G3_implemented") is False, "D2_D1_OUTPUT_NOT_VERIFIED: D1 scope changed")


def read_stage3a_seed_registry(package_root: str | Path) -> pd.DataFrame:
    path = Path(package_root) / "inputs" / "upstream" / "GeoDose_Stage3A_OUTPUTS.zip"
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("seed_registry.csv")]
        require(len(names) == 1, "D2_CONTRACT_MISMATCH: seed registry not unique")
        return pd.read_csv(io.BytesIO(archive.read(names[0])))


def frozen_orbit_seed(package_root: str | Path, scenario_id: str = "S4", replication: int = 1) -> int:
    seeds = read_stage3a_seed_registry(package_root)
    rows = seeds.loc[
        (seeds["phase"] == "pilot")
        & (seeds["scenario_id"] == scenario_id)
        & (seeds["replication"].astype(int) == int(replication))
    ]
    require(len(rows) == 1, "D2_CONTRACT_MISMATCH: frozen orbit seed not unique")
    return int(rows.iloc[0]["orbit_seed"])


def derive_seed(base_seed: int, *labels: str) -> int:
    payload = "|".join([str(int(base_seed)), *[str(label) for label in labels]]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def load_d1_fixture_config(package_root: str | Path, d1_evaluation_id: str) -> dict[str, Any]:
    registry = read_yaml(Path(package_root) / "inputs" / "d1_contracts" / "stage3d_fixture_registry.yaml")
    rows = [row for row in registry["fixtures"] if str(row["evaluation_id"]) == str(d1_evaluation_id)]
    require(len(rows) == 1, f"D2_CONTRACT_MISMATCH: D1 fixture {d1_evaluation_id}")
    return dict(rows[0])


def build_d2_fixture(data: Stage3BData, package_root: str | Path, d1_evaluation_id: str):
    d1_cfg = load_d1_fixture_config(package_root, d1_evaluation_id)
    d1_tolerances = read_yaml(Path(package_root) / "inputs" / "d1_contracts" / "stage3d_tolerance_registry.yaml")
    return build_fixture(data, d1_cfg, candidate_shift=float(d1_tolerances["candidate_algebra_shift"]))


def registered_candidate_domain(contract: dict[str, Any], case_id: str) -> dict[str, float | int | str | bool]:
    """Return the a-priori finite D2 inversion domain.

    The controlled latent outcome has Gaussian support and therefore no finite
    physical range. D2 consequently returns a *domain-truncated* numerical
    approximation on a fixed interval chosen before reading any outcomes. It
    never labels the result unbounded or equal to the full real-line set.
    """
    domain = contract["candidate_domain"]
    require(domain["domain_type"] == "fixed_registered_domain_truncated_reference", "D2_CANDIDATE_DOMAIN_INVALID")
    lower = float(domain["lower"])
    upper = float(domain["upper"])
    require(np.isfinite(lower) and np.isfinite(upper) and lower < upper, "D2_CANDIDATE_DOMAIN_INVALID")
    require(bool(domain["target_truth_allowed"]) is False, "D2_TARGET_TRUTH_LEAKAGE")
    require(bool(domain["observed_outcomes_allowed"]) is False, "D2_TARGET_TRUTH_LEAKAGE")
    return {
        "case_id": str(case_id),
        "domain_type": str(domain["domain_type"]),
        "domain_source": str(domain["source"]),
        "lower": lower,
        "upper": upper,
        "center": 0.5 * (lower + upper),
        "width": upper - lower,
        "domain_truncated_set": True,
        "full_real_line_claim": False,
        "unbounded_claim": False,
        "target_truth_used": False,
        "calibration_outcomes_used": False,
        "support_audit_outcomes_used": False,
        "nuisance_training_outcomes_used": False,
    }


def stage3b_data(package_root: str | Path) -> Stage3BData:
    return Stage3BData(Path(package_root) / "inputs" / "upstream" / "GeoDose_Stage3B_OUTPUTS.zip")
