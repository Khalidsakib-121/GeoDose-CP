from __future__ import annotations

"""Stdlib-only Windows path/provenance preflight for GeoDose Stage3D D4.

Runs BEFORE creating/installing the package-local virtual environment so missing/wrong
frozen inputs fail quickly. Scientific computation still performs the full verification again.
"""

import glob
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise PreflightError(f"Expected JSON object: {path}")
    return obj


def read_json_from_zip(path: Path, suffix: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        matches = [n for n in zf.namelist() if n.replace("\\", "/").endswith(suffix)]
        if len(matches) != 1:
            raise PreflightError(f"Expected exactly one {suffix} in {path}; found {len(matches)}")
        obj = json.loads(zf.read(matches[0]).decode("utf-8"))
        if not isinstance(obj, dict):
            raise PreflightError(f"Expected JSON object for {suffix} in {path}")
        return obj


def zip_integrity(path: Path) -> int:
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise PreflightError(f"ZIP CRC failure in {path}: {bad}")
            return len(zf.namelist())
    except zipfile.BadZipFile as exc:
        raise PreflightError(f"Invalid ZIP archive: {path}") from exc


def resolve_file(key: str, configured: dict[str, Any], fallbacks: dict[str, Any], expected: dict[str, str]) -> Path:
    candidates: list[Path] = []
    wrong_hash: list[tuple[str, str]] = []
    for raw in configured.get(key, []):
        p = Path(str(raw))
        candidates.append(p)
        if not p.is_file():
            continue
        observed = sha256_file(p)
        if observed.lower() == str(expected[key]).lower():
            return p
        wrong_hash.append((str(p), observed))

    # Fallbacks are intentionally enabled only for the three D3 output artifacts.
    for pattern in fallbacks.get(key, []):
        for raw in sorted(glob.glob(str(pattern), recursive=True), key=str.lower):
            p = Path(raw)
            if p in candidates or not p.is_file():
                continue
            candidates.append(p)
            observed = sha256_file(p)
            if observed.lower() == str(expected[key]).lower():
                return p
            wrong_hash.append((str(p), observed))

    lines = [f"  - {p}" for p in candidates] or ["  - (none)"]
    if wrong_hash:
        lines.append("Existing candidates with wrong SHA-256 (rejected):")
        lines.extend(f"  - {p}: {h}" for p, h in wrong_hash)
    raise PreflightError(f"Could not locate accepted {key}. Tried:\n" + "\n".join(lines))


def resolve_dir(key: str, configured: dict[str, Any]) -> Path:
    tried: list[str] = []
    for raw in configured.get(key, []):
        p = Path(str(raw))
        tried.append(str(p))
        if p.is_dir():
            return p
    raise PreflightError(f"Could not locate {key}. Tried:\n  - " + "\n  - ".join(tried or ["(none)"]))


def verify_source_tree(root: Path, files: dict[str, str], label: str) -> dict[str, Any]:
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    for rel, expected_sha in sorted(files.items()):
        p = root / Path(rel)
        if not p.is_file():
            missing.append(rel)
            continue
        observed = sha256_file(p)
        if observed.lower() != str(expected_sha).lower():
            mismatched.append({"path": rel, "expected": str(expected_sha), "observed": observed})
    if missing or mismatched:
        raise PreflightError(f"{label} source tree mismatch: missing={missing}; mismatched={mismatched}")
    return {"expected": len(files), "matched": len(files)}


def main() -> int:
    if os.name != "nt":
        # Development QA may import/parse this file on other OSes; operational use is Windows-only.
        print("D4 WINDOWS PREFLIGHT: non-Windows host; operational path resolution skipped.")
        return 0

    cfg = load_json(ROOT / "configs" / "local_paths_windows.json")
    fallback = load_json(ROOT / "configs" / "fallback_globs_windows.json")
    expected = load_json(ROOT / "configs" / "expected_upstream_hashes.json")
    required_file_keys = [
        "stage3a_output_zip", "stage3b_output_zip", "stage3c_output_zip", "stage3d_d1_output_zip",
        "stage3d_d2_output_zip", "stage3d_d3_m3_output_zip", "stage3d_d3_m4_output_zip",
        "stage3d_d3_integration_output_zip",
    ]
    if set(required_file_keys) != set(expected):
        raise PreflightError(
            "Frozen upstream hash registry keys do not exactly match the D4 input contract: "
            f"required={sorted(required_file_keys)} registry={sorted(expected)}"
        )

    resolved: dict[str, Path] = {}
    for key in required_file_keys:
        resolved[key] = resolve_file(key, cfg, fallback, expected)
        members = zip_integrity(resolved[key])
        print(f"  PASS {key}: {resolved[key]} [{members} ZIP members]")

    resolved["stage3c_source_root"] = resolve_dir("stage3c_source_root", cfg)
    resolved["stage3d_d2_source_root"] = resolve_dir("stage3d_d2_source_root", cfg)

    s3c_inv = read_json_from_zip(resolved["stage3c_output_zip"], "stage3c_source_hashes.json")
    d2_inv = read_json_from_zip(resolved["stage3d_d2_output_zip"], "stage3d_d2_source_hashes.json")
    if not isinstance(s3c_inv.get("files"), dict) or not isinstance(d2_inv.get("files"), dict):
        raise PreflightError("Unexpected Stage3C/D2 source hash inventory schema")
    s3c = verify_source_tree(resolved["stage3c_source_root"], s3c_inv["files"], "Stage3C")
    d2 = verify_source_tree(resolved["stage3d_d2_source_root"], d2_inv["files"], "Stage3D D2")
    print(f"  PASS stage3c_source_root: {s3c['matched']}/{s3c['expected']} files")
    print(f"  PASS stage3d_d2_source_root: {d2['matched']}/{d2['expected']} files")

    # Pre-venv check of the vendored accepted M3/M4 engine bytes.
    m4_src = read_json_from_zip(resolved["stage3d_d3_m4_output_zip"], "stage3d_d3_m4_source_hashes.json")
    rows = m4_src.get("files")
    if not isinstance(rows, list):
        raise PreflightError("Unexpected accepted M4 source hash inventory schema")
    m4_map = {str(x["path"]).replace("\\", "/"): str(x["sha256"]) for x in rows}
    vendor = ROOT / "vendor" / "geodose_stage3d_d3_m4"
    vendored = sorted(vendor.glob("*.py"))
    if not vendored:
        raise PreflightError("Vendored accepted M3/M4 source files are missing")
    for p in vendored:
        rel = "src/geodose_stage3d_d3_m4/" + p.name
        expected_sha = m4_map.get(rel)
        if expected_sha is None or sha256_file(p) != expected_sha:
            raise PreflightError(f"Vendored accepted M3/M4 byte mismatch: {rel}")
    print(f"  PASS vendored accepted M3/M4 bytes: {len(vendored)}/{len(vendored)} files")

    # Explicit regression for the dependency that caused the v1.0.3 failure.
    d3 = resolved["stage3d_d3_integration_output_zip"]
    if "GeoDose_Stage3D_D3_M1_M6_Exact_Integration_v1_0_1" not in str(d3):
        # A hash-identical copied path is allowed only via fallback; make that visible rather than silently rejecting it.
        print("  NOTE accepted D3 integration was found at a noncanonical copied path; SHA-256 is exact.")
    if sha256_file(d3) != expected["stage3d_d3_integration_output_zip"]:
        raise PreflightError("Accepted D3 integration SHA-256 regression failed")

    print("D4 WINDOWS INPUT PREFLIGHT PASSED before virtual-environment creation.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"D4 WINDOWS INPUT PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
