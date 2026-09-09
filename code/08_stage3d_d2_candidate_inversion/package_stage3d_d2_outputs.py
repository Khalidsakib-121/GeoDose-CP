from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "outputs_stage3d_d2"
    target = root / "GeoDose_Stage3D_D2_OUTPUTS.zip"
    required = output / "STAGE3D_D2_VERIFICATION.json"
    if not required.is_file():
        raise RuntimeError("Verification artifact missing; run verifier before packaging")
    if target.exists():
        target.unlink()
    fixed_time = (2026, 8, 6, 0, 0, 0)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output.iterdir()):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.name, date_time=fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"Output ZIP CRC failure: {bad}")
    print(f"Created {target.name}")
    print(f"Output ZIP SHA-256: {sha256(target)}")


if __name__ == "__main__":
    main()
