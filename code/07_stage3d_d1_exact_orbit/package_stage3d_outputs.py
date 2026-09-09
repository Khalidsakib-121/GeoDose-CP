from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs_stage3d_d1"
ARCHIVE = ROOT / "GeoDose_Stage3D_D1_OUTPUTS.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    required = OUTPUT / "STAGE3D_D1_VERIFICATION.json"
    if not required.is_file():
        raise RuntimeError("Run and verify Stage 3D D1 before packaging outputs")
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    fixed_time = (2026, 8, 6, 0, 0, 0)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(OUTPUT.iterdir()):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.name, fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(ARCHIVE) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Output ZIP CRC failure: {bad}")
    print(f"Created {ARCHIVE.name}")
    print(f"Output ZIP SHA-256: {sha256(ARCHIVE)}")


if __name__ == "__main__":
    main()
