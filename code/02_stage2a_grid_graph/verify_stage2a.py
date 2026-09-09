
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REQUIRED_FIELDS = [
    "block_id",
    "MineID",
    "MineN",
    "ReportYr",
    "block_area_m2",
    "mine_footprint_area_m2",
    "footprint_coverage",
    "rehabilitation_area_m2",
    "disturbance_area_m2",
    "mapped_rehabilitation_fraction",
    "geometry_status",
    "graph_degree",
    "fold_id",
]


class VerificationError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise VerificationError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=r"D:\G3_STAGE1_MIN_AUDIT")
    p.add_argument("--output", default=r"D:\G3_STAGE2_BLOCKS")
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    manifest_path = output_dir / "stage2_manifest.json"
    require(manifest_path.exists(), f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Source provenance.
    source_map = {
        "L03_cleaned.gpkg": input_dir / "L03_cleaned.gpkg",
        "L04_cleaned.gpkg": input_dir / "L04_cleaned.gpkg",
        "stage1_run_manifest.json": input_dir / "run_manifest.json",
    }
    for name, path in source_map.items():
        expected = manifest["source_files"].get(name)
        if expected is None and name == "stage1_run_manifest.json":
            continue
        require(path.exists(), f"Missing source file {path}")
        require(sha256_file(path) == expected, f"Source checksum mismatch: {name}")

    # Output checksums.
    for name, info in manifest["outputs"].items():
        path = output_dir / name
        require(path.exists(), f"Missing output {name}")
        require(sha256_file(path) == info["sha256"], f"Output checksum mismatch: {name}")

    gpkg = output_dir / "mine_blocks_90m.gpkg"
    layers = set(gpd.list_layers(gpkg)["name"])
    require({"blocks_90m", "mine_footprints"}.issubset(layers), "Required GPKG layers missing")
    blocks = gpd.read_file(gpkg, layer="blocks_90m")
    footprints = gpd.read_file(gpkg, layer="mine_footprints")
    attrs = pd.read_parquet(output_dir / "block_attributes.parquet")
    edges = pd.read_csv(output_dir / "block_graph_edges.csv", encoding="utf-8-sig")
    splits = pd.read_csv(output_dir / "leave_one_mine_out_splits.csv", encoding="utf-8-sig")
    summary = pd.read_csv(output_dir / "stage2_block_summary.csv", encoding="utf-8-sig")

    require(str(blocks.crs).upper() == "EPSG:9473", f"Unexpected CRS: {blocks.crs}")
    require(len(footprints) == 5, f"Expected 5 mine footprints, found {len(footprints)}")
    require(blocks["MineID"].nunique() == 5, "Expected five mines in blocks")
    require(len(blocks) == manifest["block_count"], "Block count differs from manifest")
    require(len(edges) == manifest["edge_count"], "Edge count differs from manifest")
    require(splits["fold_id"].nunique() == 5, "Expected five LOMO folds")
    require(len(summary) == 5, "Expected five mine summary rows")

    missing = [c for c in REQUIRED_FIELDS if c not in blocks.columns]
    require(not missing, f"Missing required block fields: {missing}")
    require(not blocks["block_id"].duplicated().any(), "Duplicate block IDs")
    require(blocks.geometry.is_valid.all(), "Invalid block geometry")
    require((~blocks.geometry.is_empty).all(), "Empty block geometry")
    require(np.allclose(blocks["block_area_m2"], 8100.0), "Block area is not 8,100 m2")
    require((blocks["footprint_coverage"] >= 0.70 - 1e-10).all(), "Coverage threshold violation")
    require(blocks["mapped_rehabilitation_fraction"].between(0, 1, inclusive="both").all(),
            "Mapped rehabilitation fraction outside [0,1]")
    require(set(blocks["block_id"]) == set(attrs["block_id"]),
            "GPKG and Parquet block IDs differ")

    block_ids = set(blocks["block_id"])
    require(set(edges["source_block_id"]).issubset(block_ids), "Unknown source edge endpoint")
    require(set(edges["target_block_id"]).issubset(block_ids), "Unknown target edge endpoint")
    require((edges["source_block_id"] < edges["target_block_id"]).all(),
            "Edges are not canonical undirected pairs")
    require(not edges[["source_block_id", "target_block_id"]].duplicated().any(),
            "Duplicate graph edges")

    degree = pd.concat([edges["source_block_id"], edges["target_block_id"]]).value_counts()
    actual = blocks.set_index("block_id")["graph_degree"].astype(int)
    degree = degree.reindex(actual.index, fill_value=0).astype(int)
    require(degree.equals(actual), "Graph degrees do not match the edge list")

    require(len(splits) == 5 * len(blocks), "LOMO split row count is incorrect")
    mine_by_block = blocks.set_index("block_id")["MineID"].to_dict()
    for fold_id, group in splits.groupby("fold_id"):
        held = group["held_out_MineID"].drop_duplicates()
        require(len(held) == 1, f"{fold_id} has multiple held-out mines")
        held_id = held.iloc[0]
        expected_test = {bid for bid, mid in mine_by_block.items() if mid == held_id}
        actual_test = set(group.loc[group["role"] == "test", "block_id"])
        require(actual_test == expected_test, f"{fold_id} test assignment is incorrect")

    result = {
        "status": "verified_complete",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "mine_count": int(blocks["MineID"].nunique()),
        "block_count": int(len(blocks)),
        "edge_count": int(len(edges)),
        "fold_count": int(splits["fold_id"].nunique()),
        "target_crs": str(blocks.crs),
        "graph_frozen_before_dea": bool(manifest["graph_frozen_before_dea"]),
        "checks": {
            "source_checksums": "pass",
            "output_checksums": "pass",
            "required_fields": "pass",
            "unique_block_ids": "pass",
            "geometry_validity": "pass",
            "coverage_threshold": "pass",
            "exposure_range": "pass",
            "graph_degree_consistency": "pass",
            "lomo_assignment": "pass",
        },
    }
    (output_dir / "STAGE2A_VERIFICATION.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("STAGE 2A VERIFIED COMPLETE")
    print(f"Mines: {result['mine_count']}")
    print(f"Blocks: {result['block_count']}")
    print(f"Queen edges: {result['edge_count']}")
    print(f"LOMO folds: {result['fold_count']}")


if __name__ == "__main__":
    try:
        main()
    except VerificationError as exc:
        print(f"STAGE 2A VERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
