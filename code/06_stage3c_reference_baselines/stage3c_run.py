from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geodose_stage3c.runner import run_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3a", type=Path, default=ROOT / "inputs/stage3a/GeoDose_Stage3A_OUTPUTS.zip")
    parser.add_argument("--stage3b", type=Path, default=ROOT / "inputs/stage3b/GeoDose_Stage3B_OUTPUTS.zip")
    parser.add_argument(
        "--stage3b-source",
        type=Path,
        default=ROOT / "inputs/stage3b/GeoDose_Stage3B_Controlled_Generator_v1_2_FREEZE.zip",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs_stage3c_reference")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    counts = run_reference(args.stage3a, args.stage3b, args.stage3b_source, args.output, args.overwrite)
    print("STAGE 3C REFERENCE INFRASTRUCTURE GENERATION COMPLETE")
    print(f"Validation cases: {counts['validation_cases']}")
    print(f"Target queries: {counts['target_queries']:,}")
    print(f"Interval rows: {counts['interval_rows']:,}")
    print("Implemented principal baselines: M1, M2, M5")
    print("Implemented heuristic ablations: H1, H4")
    print("M3 implemented: no — deferred to graph-local residual-law stage")
    print("M4 implemented: no — deferred until final M3 spatial law exists")
    print("M6 implemented: no")
    print("Production experiments run: no")


if __name__ == "__main__":
    main()
