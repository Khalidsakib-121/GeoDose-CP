from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

EXPECTED_INPUT_HASHES = {
    "stage3a": "9a46e21f4f488447f0829f079c8f19a345e6ac1748bc6907e20917bbb096d406",
    "stage3b_outputs": "a7e7dc14c73a7d6e88ad7ce40ed3d0f506ee862554ce9de25d922c730dced62e",
    "stage3b_source": "25ccf20697b1289019fbd6570171648080511c916af420107501163c67ef475d",
}


class Stage3CError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_inputs(stage3a: Path, stage3b_outputs: Path, stage3b_source: Path) -> Dict[str, Any]:
    observed = {
        "stage3a": sha256_file(stage3a),
        "stage3b_outputs": sha256_file(stage3b_outputs),
        "stage3b_source": sha256_file(stage3b_source),
    }
    mismatches = {
        key: {"expected": EXPECTED_INPUT_HASHES[key], "observed": value}
        for key, value in observed.items()
        if value != EXPECTED_INPUT_HASHES[key]
    }
    if mismatches:
        raise Stage3CError(f"Frozen input hash mismatch: {mismatches}")
    for path in (stage3a, stage3b_outputs, stage3b_source):
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise Stage3CError(f"ZIP integrity failure in {path.name}: {bad}")
    return {"status": "pass", "observed_sha256": observed}


def _read_csv_from_zip(zf: zipfile.ZipFile, name: str, **kwargs: Any) -> pd.DataFrame:
    raw = zf.read(name)
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), **kwargs)


def _read_json_from_zip(zf: zipfile.ZipFile, name: str) -> Dict[str, Any]:
    return json.loads(zf.read(name).decode("utf-8"))


@dataclass
class Stage3AData:
    seeds: pd.DataFrame
    methods: pd.DataFrame
    metrics: pd.DataFrame
    ablations: pd.DataFrame
    stresses: pd.DataFrame
    nuisance_models: pd.DataFrame
    verification: Dict[str, Any]


@dataclass
class Stage3BData:
    units: pd.DataFrame
    targets: pd.DataFrame
    treatment_truth: pd.DataFrame
    target_design_truth: pd.DataFrame
    true_edges: pd.DataFrame
    fitted_edges: pd.DataFrame
    temporal: pd.DataFrame
    s10_paired: pd.DataFrame
    s10_map: pd.DataFrame
    support_truth: pd.DataFrame
    cases: pd.DataFrame
    feature_contract: Dict[str, Any]
    generator_contract: Dict[str, Any]
    counts: Dict[str, Any]
    verification: Dict[str, Any]


def load_stage3a(stage3a_zip: Path) -> Stage3AData:
    prefix = "outputs_stage3a/"
    with zipfile.ZipFile(stage3a_zip) as zf:
        seeds = _read_csv_from_zip(zf, prefix + "seed_registry.csv")
        methods = _read_csv_from_zip(zf, prefix + "method_registry.csv")
        metrics = _read_csv_from_zip(zf, prefix + "metric_registry.csv")
        ablations = _read_csv_from_zip(zf, prefix + "ablation_registry.csv")
        stresses = _read_csv_from_zip(zf, prefix + "additional_stress_test_registry.csv")
        nuisance_models = _read_csv_from_zip(zf, prefix + "nuisance_model_registry.csv")
        verification = _read_json_from_zip(zf, prefix + "STAGE3A_VERIFICATION.json")
    if verification.get("status") != "verified_complete":
        raise Stage3CError("Stage 3A input is not verified_complete")
    expected_methods = {"M1", "M2", "M3", "M4", "M5", "M6"}
    if set(methods["method_id"]) != expected_methods:
        raise Stage3CError("Stage 3A method registry mismatch")
    return Stage3AData(seeds, methods, metrics, ablations, stresses, nuisance_models, verification)


def load_stage3b(stage3b_zip: Path) -> Stage3BData:
    with zipfile.ZipFile(stage3b_zip) as zf:
        units = _read_csv_from_zip(zf, "stage3b_validation_units.csv.gz")
        targets = _read_csv_from_zip(zf, "stage3b_realized_target_draws.csv.gz")
        observational = _read_csv_from_zip(zf, "stage3b_observational_target_draws.csv.gz").copy()
        # A_star remains the realized observational-law treatment.  target_dose is
        # retained only for query identity; all metrics aggregate under the fixed
        # observational-law group below.
        observational["target_dose"] = observational["A_star"]
        observational["target_scope"] = "observational_law"
        observational["bandwidth"] = float("nan")
        observational["draw_status"] = "drawn"
        observational["q_at_A_star"] = float("nan")
        observational["g_at_A_star"] = float("nan")
        observational["log_oracle_treatment_ratio_at_A_star"] = 0.0
        observational["target_supported_by_generator"] = True
        for col in targets.columns:
            if col not in observational.columns:
                observational[col] = float("nan")
        observational = observational[targets.columns]
        targets = pd.concat([targets, observational], ignore_index=True)
        targets["target_dose_group"] = targets["target_dose"].map(
            lambda x: f"dose_{float(x):.6g}" if pd.notna(x) else "missing"
        )
        targets.loc[targets["target_scope"] == "observational_law", "target_dose_group"] = "observational_law"

        treatment_truth = _read_csv_from_zip(zf, "stage3b_treatment_transport_truth.csv.gz")
        target_design_truth = _read_csv_from_zip(zf, "stage3b_target_design_truth.csv.gz")
        true_edges = _read_csv_from_zip(zf, "stage3b_true_graph_edges_by_case.csv.gz")
        fitted_edges = _read_csv_from_zip(zf, "stage3b_fitted_graph_edges.csv.gz")
        temporal = _read_csv_from_zip(zf, "stage3b_temporal_stress.csv.gz")
        s10_paired = _read_csv_from_zip(zf, "stage3b_s10_paired_aggregation.csv.gz")
        s10_map = _read_csv_from_zip(zf, "stage3b_support_180m_map.csv")
        support_truth = _read_csv_from_zip(zf, "stage3b_oracle_support_diagnostics.csv")
        cases = _read_csv_from_zip(zf, "stage3b_case_registry.csv")
        feature_contract = _read_json_from_zip(zf, "stage3b_method_feature_contract.json")
        generator_contract = _read_json_from_zip(zf, "stage3b_generator_contract.json")
        counts = _read_json_from_zip(zf, "stage3b_counts.json")
        verification = _read_json_from_zip(zf, "STAGE3B_VERIFICATION.json")
    if verification.get("status") != "verified_complete":
        raise Stage3CError("Stage 3B input is not verified_complete")
    if int(cases["case_id"].nunique()) != 27:
        raise Stage3CError("Stage 3B case count is not 27")
    return Stage3BData(
        units=units,
        targets=targets,
        treatment_truth=treatment_truth,
        target_design_truth=target_design_truth,
        true_edges=true_edges,
        fitted_edges=fitted_edges,
        temporal=temporal,
        s10_paired=s10_paired,
        s10_map=s10_map,
        support_truth=support_truth,
        cases=cases,
        feature_contract=feature_contract,
        generator_contract=generator_contract,
        counts=counts,
        verification=verification,
    )


def safe_prepare_output_dir(output_dir: Path, package_root: Path, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    package_root = package_root.resolve()
    forbidden = {
        package_root,
        (package_root / "inputs").resolve(),
        (package_root / "src").resolve(),
        (package_root / "configs").resolve(),
        (package_root / "tests").resolve(),
    }
    if output_dir in forbidden or package_root not in output_dir.parents:
        raise Stage3CError("Output directory must be a dedicated child of the package root")
    if output_dir.exists():
        if any(output_dir.iterdir()) and not overwrite:
            raise Stage3CError(f"Output directory is not empty: {output_dir}")
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def write_csv_gz_deterministic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with path.open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as gz:
            gz.write(raw)


def output_manifest(output_dir: Path, exclude: set[str] | None = None) -> Dict[str, Any]:
    exclude = exclude or set()
    records: Dict[str, Any] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in exclude:
            continue
        records[path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return records


def execution_tree_hash(package_root: Path, relative_paths: list[str]) -> tuple[str, Dict[str, str]]:
    file_hashes: Dict[str, str] = {}
    for name in sorted(relative_paths):
        path = package_root / name
        if not path.exists() or not path.is_file():
            raise Stage3CError(f"Execution-affecting file missing: {name}")
        file_hashes[name] = sha256_file(path)
    canonical = "\n".join(f"{name}\t{file_hashes[name]}" for name in sorted(file_hashes)).encode("utf-8")
    return sha256_bytes(canonical), file_hashes
