from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="inputs")
    p.add_argument("--output", default="outputs")
    args = p.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    package_dir = Path(__file__).resolve().parent

    manifest_path = output_dir / "stage2b_manifest.json"
    require(manifest_path.exists(), "Missing stage2b_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    require(manifest.get("script_version") == "1.2.0-local-stac", "Unexpected Stage 2B script version")
    require(manifest.get("stage2_graph_frozen_before_dea") is True, "Graph freeze flag failed")
    require(manifest.get("selection_uses_pv_outcome") is False, "Selection improperly used PV outcomes")
    require(manifest.get("mine_count") == 5, "Expected five mines")
    require(manifest.get("block_count") == 27042, "Expected 27,042 blocks")
    require(manifest.get("block_year_count") == 81126, "Expected 81,126 block-year rows")
    require(manifest.get("selected_mine_count") == 3, "Expected exactly three selected mines")
    require(
        sha256_file(package_dir / "stage2b_local_stac_prescreen.py") == manifest.get("stage2b_script_sha256"),
        "Stage 2B implementation checksum mismatch",
    )

    require(
        sha256_file(input_dir / "mine_blocks_90m.gpkg") == manifest["input_mine_blocks_sha256"],
        "Frozen Stage 2A GeoPackage checksum mismatch",
    )
    require(
        sha256_file(input_dir / "block_attributes.parquet") == manifest["input_block_attributes_sha256"],
        "Frozen Stage 2A block table checksum mismatch",
    )
    require(
        sha256_file(input_dir / "stage2_manifest.json") == manifest["input_stage2_manifest_sha256"],
        "Frozen Stage 2A manifest checksum mismatch",
    )

    for name, info in manifest["outputs"].items():
        path = output_dir / name
        require(path.exists(), f"Missing output {name}")
        require(path.stat().st_size == info["size_bytes"], f"Size mismatch for {name}")
        require(sha256_file(path) == info["sha256"], f"Checksum mismatch for {name}")

    block_year = pd.read_parquet(output_dir / "dea_block_year_prescreen.parquet")
    summary = pd.read_csv(output_dir / "dea_mine_prescreen_summary.csv", encoding="utf-8-sig")
    selected = pd.read_csv(output_dir / "selected_three_mines.csv", encoding="utf-8-sig")
    scenes = pd.read_csv(output_dir / "dea_scene_inventory.csv", encoding="utf-8-sig")
    config = json.loads((output_dir / "stage2b_config.json").read_text(encoding="utf-8"))
    environment = json.loads((output_dir / "dea_environment_inventory.json").read_text(encoding="utf-8"))

    require(config.get("years") == [2023, 2024, 2025], "Unexpected year configuration")
    require(config.get("output_crs") == "EPSG:9473", "Unexpected output CRS")
    require(config.get("resolution_m") == 30.0, "Unexpected DEA resolution")
    require(config.get("resampling") == "nearest", "Resampling rule is not frozen to nearest")
    require(config.get("selection_uses_pv_outcome") is False, "Config permits outcome-driven selection")
    require(environment.get("explicit_time_query_for_both_products") is True, "Explicit FC/WOfS time query not recorded")
    require(environment.get("stac_url") == "https://explorer.dea.ga.gov.au/stac", "Unexpected DEA STAC endpoint")
    require(environment.get("stac_search_method") == "GET", "STAC search is not frozen to GET")
    require(environment.get("stac_search_spatial_filter") == "bbox", "STAC spatial search is not frozen to bbox")
    require(environment.get("aws_unsigned") is True, "Public unsigned S3 access not recorded")
    require("odc-stac" in environment.get("packages", {}), "odc-stac provenance missing")
    require(environment.get("resampling") == "nearest", "Environment inventory resampling mismatch")
    require(set(environment.get("products", [])) == {"ga_ls_fc_3", "ga_ls_wo_3"}, "Unexpected DEA products")

    required = {
        "block_id", "MineID", "MineN", "year", "valid_observation_count",
        "valid_pixel_fraction", "pv_median", "ue_median", "high_ue_fraction_25",
        "water_fraction", "block_year_eligible", "mapped_rehabilitation_fraction",
        "graph_component_id", "graph_component_size", "fold_id",
    }
    require(required.issubset(block_year.columns), f"Missing block-year fields: {required - set(block_year.columns)}")
    require(len(block_year) == 81126, "Incorrect block-year row count")
    require(block_year["block_id"].nunique() == 27042, "Incorrect unique block count")
    require(set(block_year["year"].unique()) == {2023, 2024, 2025}, "Incorrect year set")
    require(not block_year.duplicated(["block_id", "year"]).any(), "Duplicate block-year records")
    require(block_year["MineID"].nunique() == 5, "Incorrect mine count")
    component_ids = block_year["graph_component_id"].astype(str)
    require(component_ids.str.len().gt(0).all(), "Empty graph component IDs")
    require(component_ids.str.contains("_CC", regex=False).all(), "Malformed graph component IDs")

    require(block_year["valid_observation_count"].ge(0).all(), "Negative observation counts")
    require(block_year["valid_pixel_fraction"].between(0, 1, inclusive="both").all(), "Invalid valid_pixel_fraction")
    for col in ["high_ue_fraction_20", "high_ue_fraction_25", "high_ue_fraction_30", "water_fraction", "unclear_fraction"]:
        values = block_year[col].dropna()
        require(values.between(0, 1, inclusive="both").all(), f"Invalid range in {col}")
    require(block_year["mapped_rehabilitation_fraction"].between(0, 1, inclusive="both").all(), "Exposure outside [0,1]")
    require(block_year["pv_median"].dropna().between(0, 100, inclusive="both").all(), "PV outside [0,100]")
    require(block_year["ue_median"].dropna().between(0, 127, inclusive="both").all(), "UE outside [0,127]")

    require(len(summary) == 5 and summary["MineID"].nunique() == 5, "Mine summary incomplete")
    require("mine_prescreen_usable" in summary.columns, "Mine-level non-vacuity status missing")
    usable = bool_series(summary["mine_prescreen_usable"])
    require(usable.notna().all(), "Invalid mine_prescreen_usable values")
    require(int(usable.sum()) >= 3, "Fewer than three mines pass the mine-level non-vacuity gate")

    require(len(selected) == 3 and selected["MineID"].nunique() == 3, "Selected mine table incorrect")
    require(selected["selection_order"].astype(int).tolist() == [1, 2, 3], "Selection order incorrect")
    selected_flags = bool_series(selected["selected_for_real_demonstration"])
    selected_usable = bool_series(selected["mine_prescreen_usable"])
    require(selected_flags.notna().all() and selected_flags.all(), "Selection flag failed")
    require(selected_usable.notna().all() and selected_usable.all(), "An unusable mine was selected")

    scene_required = {
        "MineID", "MineN", "year", "solar_day",
        "fc_solar_day_count_before_alignment", "wo_solar_day_count_before_alignment",
        "aligned_solar_day_count", "fc_solar_days_without_wo", "wo_solar_days_without_fc",
    }
    require(scene_required.issubset(scenes.columns), f"Missing scene inventory fields: {scene_required - set(scenes.columns)}")
    require(set(scenes["year"].unique()) == {2023, 2024, 2025}, "Scene inventory year set incorrect")
    require(not scenes.duplicated(["MineID", "solar_day"]).any(), "Duplicate mine solar-day inventory rows")
    for (mine_id, year), group in scenes.groupby(["MineID", "year"], sort=False):
        aligned = int(group["aligned_solar_day_count"].iloc[0])
        require(len(group) == aligned, f"Aligned scene count mismatch for {mine_id} {year}")
        require(group["aligned_solar_day_count"].nunique() == 1, f"Inconsistent alignment metadata for {mine_id} {year}")
        require(group["fc_solar_day_count_before_alignment"].nunique() == 1, f"Inconsistent FC count for {mine_id} {year}")
        require(group["wo_solar_day_count_before_alignment"].nunique() == 1, f"Inconsistent WOfS count for {mine_id} {year}")
        require(int(group["fc_solar_day_count_before_alignment"].iloc[0]) >= aligned, "FC count smaller than aligned count")
        require(int(group["wo_solar_day_count_before_alignment"].iloc[0]) >= aligned, "WOfS count smaller than aligned count")

    verification = {
        "status": "verified_complete",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": "1.2.0-local-stac",
        "mine_count": 5,
        "block_count": 27042,
        "block_year_count": 81126,
        "selected_mine_count": 3,
        "years": [2023, 2024, 2025],
        "selection_uses_pv_outcome": False,
        "checks": {
            "frozen_stage2a_checksums": "pass",
            "implementation_checksum": "pass",
            "output_checksums": "pass",
            "complete_block_year_cartesian_product": "pass",
            "required_fields": "pass",
            "quality_metric_ranges": "pass",
            "pv_ue_ranges": "pass",
            "explicit_fc_wo_time_queries": "pass",
            "scene_alignment_inventory": "pass",
            "mine_level_non_vacuity": "pass",
            "three_mines_selected": "pass",
            "selection_outcome_blind": "pass",
            "environment_inventory": "pass",
            "public_stac_get_bbox_search": "pass",
            "public_stac_unsigned_s3_access": "pass",
        },
    }
    (output_dir / "STAGE2B_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print("STAGE 2B VERIFIED COMPLETE")
    print("Mines: 5")
    print("Blocks: 27,042")
    print("Block-year rows: 81,126")
    print("Selected mines: 3")


if __name__ == "__main__":
    try:
        main()
    except VerificationError as exc:
        print(f"STAGE 2B VERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
