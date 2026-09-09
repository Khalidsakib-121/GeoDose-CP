from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from geodose_stage3d.d2_runner import run_d2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoDose-CP Stage 3D D2 G3 candidate inversion")
    parser.add_argument("--output", default="outputs_stage3d_d2")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    counts = run_d2(root, root / args.output, overwrite=args.overwrite)
    print("STAGE 3D D2 G3 CANDIDATE INVERSION GENERATION COMPLETE")
    print(f"Full inversion fixtures: {counts['full_inversion_fixture_count']}")
    print(f"Candidate trace rows: {counts['candidate_trace_rows']:,}")
    print(f"Prediction-set components: {counts['prediction_component_rows']:,}")
    print(f"Topology validation fixtures: {counts['topology_validation_rows']}")
    print(f"Size-8 stress rows: {counts['size8_stress_rows']:,}")
    print("G3 candidate p-values implemented: yes")
    print("Domain-truncated outer prediction sets generated: yes")
    print("Unbounded prediction sets claimed: no")
    print("Coverage evaluated: no")
    print("M3 implemented: no")
    print("M4 implemented: no")
    print("M6 implemented: no")
    print("Production experiments run: no")


if __name__ == "__main__":
    main()
