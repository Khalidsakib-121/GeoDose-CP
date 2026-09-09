from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
import yaml


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(payload)


def package_file_hashes(root: str | Path, *, excluded_parts: Iterable[str] = ()) -> dict[str, str]:
    root = Path(root).resolve()
    excluded = set(excluded_parts)
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in excluded for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        result[rel.as_posix()] = sha256_file(path)
    return result


def tree_hash(file_hashes: dict[str, str]) -> str:
    return canonical_json_hash(file_hashes)


def environment_inventory() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "PyYAML": yaml.__version__,
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
