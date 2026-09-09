from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs_stage3c_reference"
DESTINATION = ROOT / "GeoDose_Stage3C_REFERENCE_OUTPUTS.zip"
FIXED_TIME = (2026, 8, 6, 0, 0, 0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit("Missing outputs_stage3c_reference")
    files = sorted(path for path in SOURCE.iterdir() if path.is_file())
    if not files:
        raise SystemExit("No Stage 3C output files found")
    if DESTINATION.exists():
        DESTINATION.unlink()
    with zipfile.ZipFile(DESTINATION, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(DESTINATION) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise SystemExit(f"Output ZIP integrity failure: {bad}")
    print(f"Created {DESTINATION.name}")
    print(f"Output ZIP SHA-256: {sha256_file(DESTINATION)}")


if __name__ == "__main__":
    main()
