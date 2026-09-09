from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


class IntegrationError(RuntimeError):
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
        raise IntegrationError(f"Invalid ZIP: {path}") from exc
    if bad is not None:
        raise IntegrationError(f"ZIP integrity failure in {path}: {bad}")


def read_zip_json(path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        try:
            return json.loads(zf.read(member).decode("utf-8"))
        except KeyError as exc:
            raise IntegrationError(f"Missing {member} in {path}") from exc


def read_zip_csv(path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        try:
            raw = zf.read(member)
        except KeyError as exc:
            raise IntegrationError(f"Missing {member} in {path}") from exc
    if member.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw))


def verify_artifact_hashes(paths: dict[str, Path], expected: dict[str, str]) -> dict[str, Any]:
    audit: dict[str, Any] = {"all_upstream_pass": True, "artifacts": {}}
    for key, digest in expected.items():
        p = Path(paths[key])
        rec: dict[str, Any] = {"path": str(p), "expected_sha256": digest, "exists": p.is_file()}
        if not p.is_file():
            rec["pass"] = False
            audit["all_upstream_pass"] = False
        else:
            observed = sha256_file(p)
            rec["observed_sha256"] = observed
            try:
                verify_zip(p)
                rec["zip_integrity"] = "pass"
            except Exception as exc:
                rec["zip_integrity"] = "fail"
                rec["zip_error"] = str(exc)
            rec["pass"] = bool(observed == digest and rec["zip_integrity"] == "pass")
            audit["all_upstream_pass"] = bool(audit["all_upstream_pass"] and rec["pass"])
        audit["artifacts"][key] = rec
    if not audit["all_upstream_pass"]:
        raise IntegrationError("Frozen upstream artifact verification failed")
    return audit


def _source_hash_map(record: dict[str, Any]) -> dict[str, str]:
    files = record.get("files", {})
    if isinstance(files, dict):
        return {str(k): str(v) for k, v in files.items()}
    if isinstance(files, list):
        out = {}
        for x in files:
            out[str(x["path"])] = str(x["sha256"])
        return out
    raise IntegrationError("Unrecognized source-hash inventory format")


def verify_source_tree(root: Path, source_hash_record: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    expected = _source_hash_map(source_hash_record)
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    matched = 0
    for rel, digest in expected.items():
        fp = root / rel
        if not fp.is_file():
            missing.append(rel)
            continue
        obs = sha256_file(fp)
        if obs != digest:
            mismatched.append({"path": rel, "expected": digest, "observed": obs})
        else:
            matched += 1
    result = {
        "root": str(root),
        "expected_file_count": len(expected),
        "matched_file_count": matched,
        "missing": missing,
        "mismatched": mismatched,
        "pass": matched == len(expected) and not missing and not mismatched,
    }
    if not result["pass"]:
        raise IntegrationError(f"Frozen source tree mismatch at {root}")
    return result


def ast_api_inventory(root: Path, source_hash_record: dict[str, Any]) -> dict[str, Any]:
    expected = _source_hash_map(source_hash_record)
    modules = {}
    for rel in sorted(expected):
        if not rel.endswith(".py") or not rel.startswith("src/"):
            continue
        fp = Path(root) / rel
        tree = ast.parse(fp.read_text(encoding="utf-8"), filename=str(fp), feature_version=10)
        funcs = []
        classes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append({"name": node.name, "args": [a.arg for a in node.args.args], "lineno": node.lineno})
            elif isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({"name": node.name, "methods": methods, "lineno": node.lineno})
        modules[rel] = {"functions": funcs, "classes": classes}
    return {"root": str(root), "modules": modules, "source_only_metadata": True}


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
        raise IntegrationError("Output directory must be a child of package root")
    forbidden = {package_root / x for x in ("src", "vendor", "configs", "tests", "docs", "inputs")}
    if output_dir in {x.resolve() for x in forbidden}:
        raise IntegrationError("Unsafe output directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise IntegrationError(f"Output directory not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
