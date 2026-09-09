from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geodose_stage3b.core import (  # noqa: E402
    EXPECTED_STAGE3A_SHA256,
    SCRIPT_VERSION,
    Stage3BError,
    build_case_registry,
    build_exact_orbit_fixtures,
    build_generator_contract,
    build_method_feature_contract,
    build_paired_s10_aggregation,
    build_support_180m_map,
    dataframe_hash,
    frame_to_cases,
    generate_case,
    read_stage3a_archive,
    require,
    sha256_file,
    software_inventory,
    utc_now,
    validate_stage3a_contract,
    write_csv,
    write_csv_gz,
    write_json,
)


def output_manifest(
    output_dir: Path,
    input_sha: str,
    implementation_hashes: Dict[str, str],
    case_count: int,
    replication: int,
) -> Dict[str, Any]:
    files: Dict[str, Dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"stage3b_manifest.json", "STAGE3B_VERIFICATION.json"}:
            continue
        files[path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    combined = json.dumps(implementation_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib
    return {
        "script_version": SCRIPT_VERSION,
        "stage": "3B",
        "created_utc": utc_now(),
        "input_stage3a_archive_sha256": input_sha,
        "expected_stage3a_archive_sha256": EXPECTED_STAGE3A_SHA256,
        "implementation_files": implementation_hashes,
        "implementation_fingerprint_sha256": hashlib.sha256(combined).hexdigest(),
        "validation_case_count": case_count,
        "seed_phase": "pilot",
        "replication": int(replication),
        "production_experiments_run": False,
        "methods_implemented": False,
        "coverage_evaluated": False,
        "outputs": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoDose-CP Stage 3B controlled generator")
    parser.add_argument("--input", default="inputs/stage3a/GeoDose_Stage3A_OUTPUTS.zip")
    parser.add_argument("--output", default="outputs_stage3b")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--replication", type=int, default=1, help="Frozen pilot replication 1-20; default 1 is the Stage 3B validation release")
    args = parser.parse_args()

    if args.self_test:
        from tests.test_stage3b import run_tests

        run_tests()
        print("STAGE3B GENERATOR UNIT TESTS PASSED")
        return

    require(1 <= int(args.replication) <= 20, "--replication must be within the frozen pilot range 1-20")
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    require(output_dir.name == "outputs_stage3b", "Stage 3B output directory must be named exactly outputs_stage3b")
    require(output_dir.parent == ROOT, "Stage 3B output directory must be inside the package root")
    if output_dir.exists():
        require(args.overwrite, f"Output exists; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    stage3a = read_stage3a_archive(input_path)
    cases_frame = build_case_registry()
    cases = frame_to_cases(cases_frame)
    require(len(cases) == 27, f"Expected 27 validation cases, found {len(cases)}")
    alignment = validate_stage3a_contract(stage3a, cases_frame)
    write_csv(output_dir / "stage3b_case_registry.csv", cases_frame)
    write_json(output_dir / "stage3b_stage3a_alignment.json", alignment)

    basis_cache: Dict[Tuple[Any, ...], Any] = {}
    unit_parts: List[pd.DataFrame] = []
    hard_parts: List[pd.DataFrame] = []
    treatment_transport_parts: List[pd.DataFrame] = []
    target_design_parts: List[pd.DataFrame] = []
    target_parts: List[pd.DataFrame] = []
    observational_parts: List[pd.DataFrame] = []
    support_parts: List[pd.DataFrame] = []
    temporal_parts: List[pd.DataFrame] = []
    fitted_edge_parts: List[pd.DataFrame] = []
    true_edge_parts: List[pd.DataFrame] = []
    diagnostic_rows: List[Dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        print(f"[{i:02d}/{len(cases):02d}] Generating {case.case_id}: {case.case_name}")
        result = generate_case(case, stage3a, basis_cache, replication=int(args.replication))
        unit_parts.append(result["units"])
        hard_parts.append(result["hard_truth"])
        treatment_transport_parts.append(result["treatment_transport_truth"])
        target_design_parts.append(result["target_design_truth"])
        target_parts.append(result["target_draws"])
        if not result["observational_target_draws"].empty:
            observational_parts.append(result["observational_target_draws"])
        support_parts.append(result["support_diagnostics"])
        if not result["temporal"].empty:
            temporal_parts.append(result["temporal"])
        fitted_edge_parts.append(result["fitted_edges"])
        true_edge_parts.append(result["true_edges"].assign(case_id=case.case_id, scenario_id=case.scenario_id, replication=int(args.replication)))
        diagnostic_rows.append(result["diagnostics"])

    units = pd.concat(unit_parts, ignore_index=True)
    hard_truth = pd.concat(hard_parts, ignore_index=True)
    treatment_transport = pd.concat(treatment_transport_parts, ignore_index=True)
    target_design = pd.concat(target_design_parts, ignore_index=True)
    target_draws = pd.concat(target_parts, ignore_index=True)
    observational = pd.concat(observational_parts, ignore_index=True) if observational_parts else pd.DataFrame()
    support = pd.concat(support_parts, ignore_index=True)
    temporal = pd.concat(temporal_parts, ignore_index=True) if temporal_parts else pd.DataFrame()
    fitted_edges = pd.concat(fitted_edge_parts, ignore_index=True)
    true_edges_by_case = pd.concat(true_edge_parts, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)

    # Stable ordering is part of the reproducibility contract.
    units = units.sort_values(["case_id", "node_index"]).reset_index(drop=True)
    hard_truth = hard_truth.sort_values(["case_id", "unit_id", "target_dose"]).reset_index(drop=True)
    treatment_transport = treatment_transport.sort_values(["case_id", "unit_id", "target_dose"]).reset_index(drop=True)
    target_design = target_design.sort_values(["case_id", "node_index"]).reset_index(drop=True)
    target_draws = target_draws.sort_values(["case_id", "unit_id", "target_dose"]).reset_index(drop=True)
    if not observational.empty:
        observational = observational.sort_values(["case_id", "unit_id"]).reset_index(drop=True)
    support = support.sort_values(["case_id", "target_dose"]).reset_index(drop=True)
    if not temporal.empty:
        temporal = temporal.sort_values(["case_id", "unit_id", "year"]).reset_index(drop=True)
    fitted_edges = fitted_edges.sort_values(["case_id", "source_node", "target_node"]).reset_index(drop=True)
    true_edges_by_case = true_edges_by_case.sort_values(["case_id", "source_node", "target_node"]).reset_index(drop=True)
    diagnostics = diagnostics.sort_values("case_id").reset_index(drop=True)

    write_csv_gz(output_dir / "stage3b_validation_units.csv.gz", units)
    write_csv_gz(output_dir / "stage3b_hard_truth.csv.gz", hard_truth)
    write_csv_gz(output_dir / "stage3b_treatment_transport_truth.csv.gz", treatment_transport)
    write_csv_gz(output_dir / "stage3b_target_design_truth.csv.gz", target_design)
    write_csv_gz(output_dir / "stage3b_realized_target_draws.csv.gz", target_draws)
    if not observational.empty:
        write_csv_gz(output_dir / "stage3b_observational_target_draws.csv.gz", observational)
    write_csv(output_dir / "stage3b_oracle_support_diagnostics.csv", support)
    if not temporal.empty:
        write_csv_gz(output_dir / "stage3b_temporal_stress.csv.gz", temporal)
    write_csv_gz(output_dir / "stage3b_fitted_graph_edges.csv.gz", fitted_edges)
    write_csv_gz(output_dir / "stage3b_true_graph_edges_by_case.csv.gz", true_edges_by_case)
    write_csv(output_dir / "stage3b_scenario_diagnostics.csv", diagnostics)

    # Canonical graph fixtures and support mapping.
    from geodose_stage3b.core import GraphBasis

    basis25 = basis_cache[(25, 25, "primary")]
    basis25_small = basis_cache[(25, 25, "small_calibration")]
    basis25_maup = basis_cache[(25, 25, "maup_aligned")]
    basis13 = basis_cache[(13, 13, "primary")]
    write_csv(output_dir / "stage3b_true_queen_edges_25x25_primary.csv", basis25.edges_queen)
    write_csv(output_dir / "stage3b_rook_edges_25x25_primary.csv", basis25.edges_rook)
    write_csv(output_dir / "stage3b_true_queen_edges_25x25_small_calibration.csv", basis25_small.edges_queen)
    write_csv(output_dir / "stage3b_true_queen_edges_25x25_maup_aligned.csv", basis25_maup.edges_queen)
    write_csv(output_dir / "stage3b_true_queen_edges_13x13_primary.csv", basis13.edges_queen)
    support_map = build_support_180m_map()
    write_csv(output_dir / "stage3b_support_180m_map.csv", support_map)
    paired_s10 = build_paired_s10_aggregation(units, support_map)
    write_csv_gz(output_dir / "stage3b_s10_paired_aggregation.csv.gz", paired_s10)
    write_json(output_dir / "stage3b_exact_orbit_fixtures.json", build_exact_orbit_fixtures(units, target_draws, basis25.edges_queen))

    # Deterministic replay evidence for one complete case.
    replay_case = [case for case in cases if case.case_id == "S1_BASE"][0]
    replay_a = generate_case(replay_case, stage3a, basis_cache, replication=int(args.replication))["units"].sort_values("node_index").reset_index(drop=True)
    replay_b = generate_case(replay_case, stage3a, basis_cache, replication=int(args.replication))["units"].sort_values("node_index").reset_index(drop=True)
    replay_hash_a = dataframe_hash(replay_a)
    replay_hash_b = dataframe_hash(replay_b)
    require(replay_hash_a == replay_hash_b, "Deterministic replay failed")
    write_json(
        output_dir / "stage3b_replay_diagnostics.json",
        {
            "case_id": "S1_BASE",
            "replication": int(args.replication),
            "hash_first": replay_hash_a,
            "hash_second": replay_hash_b,
            "identical": True,
        },
    )

    write_json(output_dir / "stage3b_generator_contract.json", build_generator_contract(len(cases)))
    write_json(output_dir / "stage3b_method_feature_contract.json", build_method_feature_contract())
    write_json(output_dir / "stage3b_environment_inventory.json", software_inventory())
    write_json(
        output_dir / "stage3b_counts.json",
        {
            "case_count": len(cases),
            "replication": int(args.replication),
            "seed_phase": "pilot",
            "unit_rows": len(units),
            "hard_truth_rows": len(hard_truth),
            "treatment_transport_rows": len(treatment_transport),
            "target_design_rows": len(target_design),
            "realized_target_draw_rows": len(target_draws),
            "observational_target_draw_rows": len(observational),
            "support_diagnostic_rows": len(support),
            "temporal_rows": len(temporal),
            "fitted_graph_edge_rows": len(fitted_edges),
            "true_graph_edge_rows": len(true_edges_by_case),
            "s10_paired_aggregation_rows": len(paired_s10),
            "production_experiments_run": False,
        },
    )

    implementation_paths = {
        "stage3b_generate.py": Path(__file__).resolve(),
        "src/geodose_stage3b/core.py": ROOT / "src" / "geodose_stage3b" / "core.py",
        "verify_stage3b.py": ROOT / "verify_stage3b.py",
        "tests/test_stage3b.py": ROOT / "tests" / "test_stage3b.py",
        "requirements_frozen_py310.txt": ROOT / "requirements_frozen_py310.txt",
    }
    implementation_hashes = {name: sha256_file(path) for name, path in implementation_paths.items()}
    manifest = output_manifest(output_dir, stage3a["sha256"], implementation_hashes, len(cases), int(args.replication))
    write_json(output_dir / "stage3b_manifest.json", manifest)

    print("STAGE 3B GENERATION COMPLETE")
    print(f"Validation cases: {len(cases)}")
    print(f"Pilot replication: {int(args.replication)}")
    print(f"Unit rows: {len(units):,}")
    print(f"Realized target-draw rows: {len(target_draws):,}")
    print("Methods implemented: no")
    print("Coverage evaluated: no")


if __name__ == "__main__":
    try:
        main()
    except Stage3BError as exc:
        print(f"STAGE 3B FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
