from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from geodose_stage3d.runner import run_d1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GeoDose-CP Stage 3D D0/D1 exact-orbit algebra gate")
    parser.add_argument("--output", default="outputs_stage3d_d1", help="Package-local output directory")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate the dedicated output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_d1(ROOT, ROOT / args.output, overwrite=args.overwrite)
