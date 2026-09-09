from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class D3M4Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise D3M4Error(f"Invalid ZIP: {path}") from exc
    if bad is not None:
        raise D3M4Error(f"ZIP integrity failure in {path}: {bad}")


def verify_upstream(paths: dict[str, Path], expected: dict[str, str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    all_pass = True
    for key, digest in expected.items():
        if key not in paths:
            raise D3M4Error(f"Required upstream path key missing: {key}")
        p = Path(paths[key])
        rec: dict[str, Any] = {"path": str(p), "expected_sha256": digest}
        if not p.is_file():
            rec.update({"exists": False, "pass": False, "error": "missing"})
            all_pass = False
        else:
            observed = sha256_file(p)
            rec.update({"exists": True, "observed_sha256": observed})
            try:
                verify_zip(p)
                zip_ok = True
            except Exception as exc:
                zip_ok = False
                rec["zip_error"] = str(exc)
            rec["zip_integrity"] = "pass" if zip_ok else "fail"
            rec["pass"] = bool(observed == digest and zip_ok)
            all_pass &= rec["pass"]
        rows[key] = rec
    if not all_pass:
        raise D3M4Error(f"Frozen upstream verification failed: {rows}")
    return {"all_upstream_pass": True, "artifacts": rows}


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    try:
        return zf.read(name)
    except KeyError as exc:
        raise D3M4Error(f"Required ZIP member missing: {name}") from exc


def _csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    raw = _read_member(zf, name)
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw))


def _json(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return json.loads(_read_member(zf, name).decode("utf-8"))


@dataclass(frozen=True)
class Stage3BOracleData:
    units: pd.DataFrame
    realized_targets: pd.DataFrame
    observational_targets: pd.DataFrame
    treatment_truth: pd.DataFrame
    true_edges: pd.DataFrame
    cases: pd.DataFrame
    exact_fixtures: dict[str, Any]
    verification: dict[str, Any]
    generator_contract: dict[str, Any]


@dataclass(frozen=True)
class D2ReferenceData:
    candidate_states: pd.DataFrame
    candidate_trace: pd.DataFrame
    grid_registry: pd.DataFrame
    verification: dict[str, Any]
    contract: dict[str, Any]
    source_hashes: dict[str, Any]


@dataclass(frozen=True)
class M3ReferenceData:
    candidate_trace: pd.DataFrame
    source_marginals: pd.DataFrame
    verification: dict[str, Any]
    contract: dict[str, Any]
    source_hashes: dict[str, Any]


def load_stage3b_oracle(path: Path) -> Stage3BOracleData:
    with zipfile.ZipFile(path) as zf:
        units = _csv(zf, "stage3b_validation_units.csv.gz")
        realized = _csv(zf, "stage3b_realized_target_draws.csv.gz")
        observational = _csv(zf, "stage3b_observational_target_draws.csv.gz")
        treatment = _csv(zf, "stage3b_treatment_transport_truth.csv.gz")
        edges = _csv(zf, "stage3b_true_graph_edges_by_case.csv.gz")
        cases = _csv(zf, "stage3b_case_registry.csv")
        fixtures = _json(zf, "stage3b_exact_orbit_fixtures.json")
        verification = _json(zf, "STAGE3B_VERIFICATION.json")
        contract = _json(zf, "stage3b_generator_contract.json")
    if verification.get("status") != "verified_complete":
        raise D3M4Error("Stage 3B is not verified_complete")
    if fixtures.get("source_case_id") != "S4_RHO060":
        raise D3M4Error("Stage 3B exact fixture source case changed")
    return Stage3BOracleData(units, realized, observational, treatment, edges, cases, fixtures, verification, contract)


def load_d2_reference(path: Path) -> D2ReferenceData:
    with zipfile.ZipFile(path) as zf:
        states = _csv(zf, "stage3d_d2_candidate_state_audit.csv.gz")
        trace = _csv(zf, "stage3d_d2_candidate_pvalue_trace.csv.gz")
        grid = _csv(zf, "stage3d_d2_candidate_grid_registry.csv")
        verification = _json(zf, "STAGE3D_D2_VERIFICATION.json")
        contract = _json(zf, "stage3d_d2_contract_summary.json")
        source_hashes = _json(zf, "stage3d_d2_source_hashes.json")
    if verification.get("status") != "verified_complete" or int(verification.get("failed_checks", -1)) != 0:
        raise D3M4Error("Accepted D2 output is not verified_complete")
    if verification.get("G3_candidate_pvalues_implemented") is not True:
        raise D3M4Error("D2 reference does not report G3 candidate p-values")
    if verification.get("M4_implemented") is not False or verification.get("M6_implemented") is not False:
        raise D3M4Error("D2 method boundary changed unexpectedly")
    return D2ReferenceData(states, trace, grid, verification, contract, source_hashes)


def load_m3_reference(path: Path) -> M3ReferenceData:
    with zipfile.ZipFile(path) as zf:
        trace = _csv(zf, "stage3d_d3_m3_candidate_trace.csv.gz")
        source = _csv(zf, "stage3d_d3_m3_source_marginals.csv.gz")
        verification = _json(zf, "STAGE3D_D3_M3_VERIFICATION.json")
        contract = _json(zf, "stage3d_d3_m3_contract_summary.json")
        source_hashes = _json(zf, "stage3d_d3_m3_source_hashes.json")
    if verification.get("status") != "verified_complete" or int(verification.get("failed_checks", -1)) != 0:
        raise D3M4Error("Accepted M3 output is not verified_complete")
    if contract.get("M3_implemented") is not True or contract.get("M4_implemented") is not False:
        raise D3M4Error("M3 reference contract has unexpected method boundary")
    if contract.get("definition_gate_version") != "1.1":
        raise D3M4Error("M3 reference does not use Definition Gate v1.1")
    return M3ReferenceData(trace, source, verification, contract, source_hashes)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with path.open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, compresslevel=9, mtime=0) as gz:
            gz.write(raw)


def safe_reset_output_dir(output_dir: Path, package_root: Path, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    package_root = package_root.resolve()
    if package_root not in output_dir.parents or output_dir == package_root:
        raise D3M4Error("Output directory must be a child of the package root")
    forbidden = {package_root / x for x in ("src", "configs", "tests", "docs", "inputs")}
    if output_dir in {x.resolve() for x in forbidden}:
        raise D3M4Error("Refusing unsafe output directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise D3M4Error(f"Output directory not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
