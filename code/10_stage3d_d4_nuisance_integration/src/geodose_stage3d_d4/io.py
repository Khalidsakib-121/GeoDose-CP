from __future__ import annotations

import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


class D4Error(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    text = df.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(text.encode("utf-8"))


def read_json_from_zip(path: Path, logical_name: str) -> Any:
    with zipfile.ZipFile(path) as zf:
        matches = [n for n in zf.namelist() if n.replace("\\", "/").endswith(logical_name)]
        if len(matches) != 1:
            raise D4Error(f"Expected one {logical_name} in {path}, found {len(matches)}")
        return json.loads(zf.read(matches[0]).decode("utf-8"))


def read_csv_from_zip(path: Path, logical_name: str, **kwargs: Any) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        matches = [n for n in zf.namelist() if n.replace("\\", "/").endswith(logical_name)]
        if len(matches) != 1:
            raise D4Error(f"Expected one {logical_name} in {path}, found {len(matches)}")
        raw=zf.read(matches[0])
        if logical_name.endswith('.gz'):
            raw=gzip.decompress(raw)
        return pd.read_csv(io.BytesIO(raw), **kwargs)


def verify_zip(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise D4Error(f"Missing input archive: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise D4Error(f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, observed {observed}")
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise D4Error(f"ZIP CRC failure in {path}: {bad}")
        count = len(zf.namelist())
    return {"path": str(path), "sha256": observed, "zip_integrity": True, "member_count": count}


def source_hash_inventory_from_output_zip(path: Path, logical_name: str) -> dict[str, str]:
    obj = read_json_from_zip(path, logical_name)
    files = obj.get("files")
    if not isinstance(files, dict):
        raise D4Error(f"Unexpected source hash inventory in {logical_name}")
    return {str(k).replace("\\", "/"): str(v) for k, v in files.items()}


def verify_source_tree(root: Path, expected_files: dict[str, str]) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise D4Error(f"Missing source root: {root}")
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    for rel, expected in sorted(expected_files.items()):
        fp = root / Path(rel)
        if not fp.is_file():
            missing.append(rel)
            continue
        observed = sha256_file(fp)
        if observed != expected:
            mismatched.append({"path": rel, "expected": expected, "observed": observed})
    return {
        "root": str(root),
        "expected_file_count": len(expected_files),
        "matched_file_count": len(expected_files) - len(missing) - len(mismatched),
        "missing": missing,
        "mismatched": mismatched,
        "pass": not missing and not mismatched,
    }


def ensure_empty_output_dir(path: Path, overwrite: bool) -> None:
    path = Path(path)
    if path.exists():
        files = list(path.iterdir())
        if files and not overwrite:
            raise D4Error(f"Output directory is not empty: {path}. Use --overwrite only for a deliberate rerun.")
        if overwrite:
            for fp in files:
                if fp.is_file() or fp.is_symlink():
                    fp.unlink()
                elif fp.is_dir():
                    import shutil
                    shutil.rmtree(fp)
    path.mkdir(parents=True, exist_ok=True)
