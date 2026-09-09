
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

VERSION = "1.0.0"
TARGET_CRS = "EPSG:9473"
GRID_SIZE_M = 90.0
MIN_FOOTPRINT_COVERAGE = 0.70
EXPECTED_MINE_COUNT = 5

REQUIRED_BLOCK_FIELDS = [
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


class Stage2Error(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def mine_token(mine_id: str) -> str:
    token = re_sub_non_alnum(clean_text(mine_id).strip("{}"))
    return token[:12] if token else hashlib.sha1(clean_text(mine_id).encode("utf-8")).hexdigest()[:12]


def re_sub_non_alnum(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "", value)


def safe_make_valid(geom):
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    if geom is None or geom.is_empty:
        return geom
    # Retain polygonal parts only.
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = []
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            if part.geom_type in ("Polygon", "MultiPolygon") and not part.is_empty:
                parts.append(part)
    return unary_union(parts) if parts else None


def load_dissolved(path: Path, label: str) -> gpd.GeoDataFrame:
    if not path.exists():
        raise Stage2Error(f"Missing input: {path}")
    layers = set(gpd.list_layers(path)["name"])
    if "dissolved_by_mine_year" not in layers:
        raise Stage2Error(f"{path} has no dissolved_by_mine_year layer")
    gdf = gpd.read_file(path, layer="dissolved_by_mine_year")
    needed = {"MineID_norm", "MineN", "ReportYr_num", "geometry"}
    missing = needed - set(gdf.columns)
    if missing:
        raise Stage2Error(f"{label} missing columns: {sorted(missing)}")
    if gdf.crs is None:
        raise Stage2Error(f"{label} has no CRS")
    if str(gdf.crs).upper() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    gdf = gdf.copy()
    gdf["MineID_norm"] = gdf["MineID_norm"].map(clean_text)
    gdf["MineN"] = gdf["MineN"].map(clean_text)
    gdf["ReportYr_num"] = pd.to_numeric(gdf["ReportYr_num"], errors="coerce")
    if gdf["MineID_norm"].eq("").any() or gdf["MineN"].eq("").any() or gdf["ReportYr_num"].isna().any():
        raise Stage2Error(f"{label} contains blank mine identity or report year")
    gdf["geometry"] = gdf.geometry.map(safe_make_valid)
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any() or (~gdf.geometry.is_valid).any():
        raise Stage2Error(f"{label} contains unusable geometry after validation")
    dup = gdf.duplicated(["MineID_norm"], keep=False)
    if dup.any():
        raise Stage2Error(
            f"{label} must contain exactly one dissolved row per selected mine; duplicates: "
            f"{gdf.loc[dup, 'MineID_norm'].tolist()}"
        )
    return gdf


def align_mines(l03: gpd.GeoDataFrame, l04: gpd.GeoDataFrame) -> list[dict]:
    ids03, ids04 = set(l03["MineID_norm"]), set(l04["MineID_norm"])
    if ids03 != ids04:
        raise Stage2Error(
            f"L03/L04 mine sets differ. only_L03={sorted(ids03-ids04)}, only_L04={sorted(ids04-ids03)}"
        )
    if len(ids03) != EXPECTED_MINE_COUNT:
        raise Stage2Error(f"Expected exactly {EXPECTED_MINE_COUNT} mines, found {len(ids03)}")
    out = []
    for mid in sorted(ids03):
        r3 = l03.loc[l03["MineID_norm"] == mid].iloc[0]
        r4 = l04.loc[l04["MineID_norm"] == mid].iloc[0]
        y3, y4 = int(r3["ReportYr_num"]), int(r4["ReportYr_num"])
        if y3 != y4:
            raise Stage2Error(f"Report-year mismatch for {mid}: L03={y3}, L04={y4}")
        n3, n4 = clean_text(r3["MineN"]), clean_text(r4["MineN"])
        if n3 != n4:
            raise Stage2Error(f"Mine-name mismatch for {mid}: L03={n3!r}, L04={n4!r}")
        rehab = safe_make_valid(r3.geometry)
        disturb = safe_make_valid(r4.geometry)
        footprint = safe_make_valid(unary_union([rehab, disturb]))
        if footprint is None or footprint.is_empty or not footprint.is_valid:
            raise Stage2Error(f"Invalid union footprint for {mid}")
        out.append({
            "MineID": mid,
            "MineN": n3,
            "ReportYr": y3,
            "rehab": rehab,
            "disturbance": disturb,
            "footprint": footprint,
        })
    return out


def candidate_cells(footprint, size: float) -> tuple[gpd.GeoDataFrame, int]:
    minx, miny, maxx, maxy = footprint.bounds
    cols = np.arange(math.floor(minx / size), math.ceil(maxx / size), dtype=np.int64)
    rows = np.arange(math.floor(miny / size), math.ceil(maxy / size), dtype=np.int64)
    col_grid, row_grid = np.meshgrid(cols, rows)
    col_values = col_grid.ravel()
    row_values = row_grid.ravel()
    x0 = col_values.astype(float) * size
    y0 = row_values.astype(float) * size
    geom_array = shapely.box(x0, y0, x0 + size, y0 + size)
    gdf = gpd.GeoDataFrame(
        {
            "grid_row": row_values.astype(int),
            "grid_col": col_values.astype(int),
        },
        geometry=geom_array,
        crs=TARGET_CRS,
    )
    return gdf, len(gdf)



def geometry_parts(geom) -> np.ndarray:
    if geom is None or geom.is_empty:
        return np.array([], dtype=object)
    if geom.geom_type == "Polygon":
        return np.array([geom], dtype=object)
    if geom.geom_type == "MultiPolygon":
        return np.array(list(geom.geoms), dtype=object)
    raise Stage2Error(f"Expected polygonal geometry, got {geom.geom_type}")


def intersection_area_by_parts(cell_geometries: np.ndarray, geom) -> np.ndarray:
    """Exact intersection areas using an STRtree over disjoint dissolved polygon parts."""
    parts = geometry_parts(geom)
    if len(parts) == 0:
        return np.zeros(len(cell_geometries), dtype=float)
    tree = shapely.STRtree(parts)
    pairs = tree.query(cell_geometries, predicate="intersects")
    out = np.zeros(len(cell_geometries), dtype=float)
    if pairs.size == 0:
        return out
    cell_idx = pairs[0]
    part_idx = pairs[1]
    intersections = shapely.intersection(cell_geometries[cell_idx], parts[part_idx])
    areas = shapely.area(intersections)
    np.add.at(out, cell_idx, areas)
    return out


def build_blocks_for_mine(mine: dict, size: float, min_coverage: float) -> tuple[gpd.GeoDataFrame, dict]:
    cells, n_candidates = candidate_cells(mine["footprint"], size)
    cell_array = cells.geometry.to_numpy()
    footprint_area_all = intersection_area_by_parts(cell_array, mine["footprint"])
    block_area_all = np.full(len(cells), size * size, dtype=float)
    coverage_all = footprint_area_all / block_area_all
    keep = coverage_all >= (min_coverage - 1e-12)
    cells = cells.loc[keep].copy().reset_index(drop=True)
    footprint_area = footprint_area_all[keep]
    coverage = coverage_all[keep]
    if cells.empty:
        raise Stage2Error(f"No blocks retained for {mine['MineN']}")

    retained_cells = cells.geometry.to_numpy()
    rehab_area = intersection_area_by_parts(retained_cells, mine["rehab"])
    disturbance_area = intersection_area_by_parts(retained_cells, mine["disturbance"])
    block_area = np.full(len(cells), size * size, dtype=float)

    tol = 1e-5
    if np.any(rehab_area - footprint_area > tol) or np.any(disturbance_area - footprint_area > tol):
        raise Stage2Error(f"Intersection area exceeds footprint area for {mine['MineN']}")

    frac = np.divide(
        rehab_area,
        footprint_area,
        out=np.full_like(rehab_area, np.nan, dtype=float),
        where=footprint_area > 0,
    )
    frac = np.clip(frac, 0.0, 1.0)

    token = mine_token(mine["MineID"])
    cells["block_id"] = [
        f"{token}_{mine['ReportYr']}_R{r}_C{c}"
        for r, c in zip(cells["grid_row"], cells["grid_col"])
    ]
    cells["MineID"] = mine["MineID"]
    cells["MineN"] = mine["MineN"]
    cells["ReportYr"] = int(mine["ReportYr"])
    cells["block_area_m2"] = block_area
    cells["mine_footprint_area_m2"] = footprint_area
    cells["footprint_coverage"] = coverage
    cells["rehabilitation_area_m2"] = rehab_area
    cells["disturbance_area_m2"] = disturbance_area
    cells["mapped_rehabilitation_fraction"] = frac
    cells["geometry_status"] = np.where(
        coverage >= 0.999999,
        "retained_full_90m",
        "retained_boundary_ge_70pct",
    )
    cells["is_boundary_block"] = coverage < 0.999999
    cells["centroid_x"] = cells.geometry.centroid.x
    cells["centroid_y"] = cells.geometry.centroid.y

    summary = {
        "MineID": mine["MineID"],
        "MineN": mine["MineN"],
        "ReportYr": int(mine["ReportYr"]),
        "candidate_grid_cells": int(n_candidates),
        "retained_blocks": int(len(cells)),
        "full_blocks": int((coverage >= 0.999999).sum()),
        "boundary_blocks": int((coverage < 0.999999).sum()),
        "mine_footprint_area_ha": float(mine["footprint"].area / 10000.0),
        "retained_footprint_area_ha": float(footprint_area.sum() / 10000.0),
        "represented_footprint_fraction": float(footprint_area.sum() / mine["footprint"].area),
        "rehabilitation_area_in_retained_blocks_ha": float(rehab_area.sum() / 10000.0),
        "disturbance_area_in_retained_blocks_ha": float(disturbance_area.sum() / 10000.0),
        "mapped_fraction_min": float(np.nanmin(frac)),
        "mapped_fraction_q25": float(np.nanquantile(frac, 0.25)),
        "mapped_fraction_median": float(np.nanmedian(frac)),
        "mapped_fraction_q75": float(np.nanquantile(frac, 0.75)),
        "mapped_fraction_max": float(np.nanmax(frac)),
        "n_exact_zero_blocks": int(np.isclose(frac, 0.0, atol=1e-12).sum()),
        "n_exact_one_blocks": int(np.isclose(frac, 1.0, atol=1e-12).sum()),
    }
    return cells, summary


NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
]


def build_queen_graph(blocks: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_rows = []
    degree = {bid: 0 for bid in blocks["block_id"]}
    component = {}
    component_size = {}

    for mine_id, group in blocks.groupby("MineID", sort=True):
        lookup = {
            (int(row.grid_row), int(row.grid_col)): row.block_id
            for row in group.itertuples()
        }
        mine_edges = set()
        adjacency = {bid: set() for bid in group["block_id"]}
        for (r, c), bid in lookup.items():
            for dr, dc in NEIGHBOR_OFFSETS:
                nb = lookup.get((r + dr, c + dc))
                if nb is None or nb == bid:
                    continue
                a, b = sorted((bid, nb))
                mine_edges.add((a, b))
                adjacency[bid].add(nb)
        for a, b in sorted(mine_edges):
            edge_rows.append({
                "MineID": mine_id,
                "source_block_id": a,
                "target_block_id": b,
                "contiguity": "queen_8_neighbor",
            })
            degree[a] += 1
            degree[b] += 1

        # Connected components, deterministic by sorted block ID.
        seen = set()
        comp_num = 0
        for start in sorted(adjacency):
            if start in seen:
                continue
            comp_num += 1
            q = deque([start])
            seen.add(start)
            members = []
            while q:
                cur = q.popleft()
                members.append(cur)
                for nxt in sorted(adjacency[cur]):
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            cid = f"{mine_token(mine_id)}_CC{comp_num:03d}"
            for bid in members:
                component[bid] = cid
                component_size[bid] = len(members)

    edges = pd.DataFrame(edge_rows, columns=[
        "MineID", "source_block_id", "target_block_id", "contiguity"
    ])
    attrs = pd.DataFrame({
        "block_id": blocks["block_id"],
        "graph_degree": blocks["block_id"].map(degree).astype(int),
        "graph_component_id": blocks["block_id"].map(component),
        "graph_component_size": blocks["block_id"].map(component_size).astype(int),
    })
    return edges, attrs


def make_lomo_splits(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mines = (
        blocks[["MineID", "MineN"]]
        .drop_duplicates()
        .sort_values(["MineN", "MineID"])
        .reset_index(drop=True)
    )
    fold_map = {
        row.MineID: f"LOMO_{i+1:02d}"
        for i, row in mines.iterrows()
    }
    block_fold = blocks["MineID"].map(fold_map)

    rows = []
    for row in mines.itertuples():
        fold_id = fold_map[row.MineID]
        for b in blocks[["block_id", "MineID", "MineN"]].itertuples():
            rows.append({
                "fold_id": fold_id,
                "held_out_MineID": row.MineID,
                "held_out_MineN": row.MineN,
                "block_id": b.block_id,
                "block_MineID": b.MineID,
                "block_MineN": b.MineN,
                "role": "test" if b.MineID == row.MineID else "development_pool",
            })
    split_df = pd.DataFrame(rows)
    fold_table = mines.copy()
    fold_table["fold_id"] = fold_table["MineID"].map(fold_map)
    return split_df, pd.DataFrame({
        "block_id": blocks["block_id"],
        "fold_id": block_fold,
    })


def validate_outputs(blocks: gpd.GeoDataFrame, edges: pd.DataFrame, splits: pd.DataFrame):
    missing = [c for c in REQUIRED_BLOCK_FIELDS if c not in blocks.columns]
    if missing:
        raise Stage2Error(f"Required block fields missing: {missing}")
    if blocks["block_id"].duplicated().any():
        raise Stage2Error("Duplicate block_id values")
    if blocks.geometry.is_empty.any() or (~blocks.geometry.is_valid).any():
        raise Stage2Error("Invalid full block geometry")
    if not np.allclose(blocks["block_area_m2"], GRID_SIZE_M**2):
        raise Stage2Error("Block areas are not 8,100 m²")
    if (blocks["footprint_coverage"] < MIN_FOOTPRINT_COVERAGE - 1e-10).any():
        raise Stage2Error("A retained block violates the coverage threshold")
    if not blocks["mapped_rehabilitation_fraction"].between(0, 1, inclusive="both").all():
        raise Stage2Error("Mapped rehabilitation fraction outside [0,1]")
    edge_pairs = edges[["source_block_id", "target_block_id"]].apply(tuple, axis=1)
    if edge_pairs.duplicated().any():
        raise Stage2Error("Duplicate graph edges")
    if (edges["source_block_id"] >= edges["target_block_id"]).any():
        raise Stage2Error("Graph edges are not canonical undirected pairs")
    computed_degree = pd.concat([
        edges["source_block_id"], edges["target_block_id"]
    ]).value_counts()
    expected = blocks.set_index("block_id")["graph_degree"]
    aligned = computed_degree.reindex(expected.index, fill_value=0).astype(int)
    if not aligned.equals(expected.astype(int)):
        raise Stage2Error("Graph degree does not match edge list")
    n_mines = blocks["MineID"].nunique()
    if n_mines != EXPECTED_MINE_COUNT:
        raise Stage2Error(f"Expected {EXPECTED_MINE_COUNT} mines in blocks, found {n_mines}")
    if splits["fold_id"].nunique() != EXPECTED_MINE_COUNT:
        raise Stage2Error("Incorrect number of LOMO folds")
    expected_split_rows = len(blocks) * EXPECTED_MINE_COUNT
    if len(splits) != expected_split_rows:
        raise Stage2Error(f"Expected {expected_split_rows} split rows, got {len(splits)}")
    test_counts = splits.loc[splits["role"] == "test"].groupby("fold_id")["block_id"].count()
    if (test_counts <= 0).any() or len(test_counts) != EXPECTED_MINE_COUNT:
        raise Stage2Error("A LOMO fold has no test blocks")


def write_parquet_or_fail(df: pd.DataFrame, path: Path, allow_csv_fallback: bool):
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        if not allow_csv_fallback:
            raise Stage2Error(
                "Could not write Parquet. Install pyarrow (included in requirements.txt). "
                f"Original error: {exc}"
            ) from exc
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        path.write_text(
            "PARQUET_NOT_CREATED_IN_TEST_ENVIRONMENT; see block_attributes.csv",
            encoding="utf-8",
        )


def self_test() -> None:
    # Four cells in a 2x2 arrangement must form a complete queen graph K4.
    records = []
    for r in (0, 1):
        for c in (0, 1):
            records.append({
                "block_id": f"B{r}{c}",
                "MineID": "M",
                "MineN": "Mine",
                "grid_row": r,
                "grid_col": c,
                "geometry": box(c*90, r*90, (c+1)*90, (r+1)*90),
            })
    gdf = gpd.GeoDataFrame(records, crs=TARGET_CRS)
    edges, attrs = build_queen_graph(gdf)
    assert len(edges) == 6, len(edges)
    assert set(attrs["graph_degree"]) == {3}, attrs
    # Exposure identity.
    cell = box(0, 0, 90, 90)
    rehab = box(0, 0, 45, 90)
    assert abs(rehab.intersection(cell).area / cell.area - 0.5) < 1e-12
    print("Stage 2A self-test passed.")


def run(args) -> None:
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    l03_path = input_dir / "L03_cleaned.gpkg"
    l04_path = input_dir / "L04_cleaned.gpkg"
    stage1_manifest = input_dir / "run_manifest.json"

    if output_dir.exists():
        if not args.overwrite:
            raise Stage2Error(f"Output exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    l03 = load_dissolved(l03_path, "L03")
    l04 = load_dissolved(l04_path, "L04")
    mines = align_mines(l03, l04)

    full_parts, summaries = [], []
    footprints = []
    for mine in mines:
        full, summary = build_blocks_for_mine(
            mine, GRID_SIZE_M, MIN_FOOTPRINT_COVERAGE
        )
        full_parts.append(full)
        summaries.append(summary)
        footprints.append({
            "MineID": mine["MineID"],
            "MineN": mine["MineN"],
            "ReportYr": mine["ReportYr"],
            "geometry": mine["footprint"],
        })

    blocks = gpd.GeoDataFrame(
        pd.concat(full_parts, ignore_index=True), geometry="geometry", crs=TARGET_CRS
    )
    footprint_gdf = gpd.GeoDataFrame(footprints, geometry="geometry", crs=TARGET_CRS)

    edges, graph_attrs = build_queen_graph(blocks)
    blocks = blocks.merge(graph_attrs, on="block_id", how="left", validate="one_to_one")

    splits, block_fold = make_lomo_splits(pd.DataFrame(blocks.drop(columns="geometry")))
    blocks = blocks.merge(block_fold, on="block_id", how="left", validate="one_to_one")

    summary_df = pd.DataFrame(summaries)
    graph_summary = (
        blocks.groupby(["MineID", "MineN"], as_index=False)
        .agg(
            graph_edges=("graph_degree", lambda s: int(s.sum() // 2)),
            graph_degree_min=("graph_degree", "min"),
            graph_degree_mean=("graph_degree", "mean"),
            graph_degree_max=("graph_degree", "max"),
            graph_components=("graph_component_id", "nunique"),
            isolated_blocks=("graph_degree", lambda s: int((s == 0).sum())),
        )
    )
    summary_df = summary_df.merge(graph_summary, on=["MineID", "MineN"], how="left")
    summary_df["support_warning"] = np.where(
        summary_df["represented_footprint_fraction"] < 0.85,
        "LOW_REPRESENTED_FOOTPRINT_FRACTION",
        np.where(
            summary_df["isolated_blocks"] > 0,
            "ISOLATED_BLOCKS_PRESENT",
            "OK",
        )
    )

    validate_outputs(blocks, edges, splits)

    gpkg_path = output_dir / "mine_blocks_90m.gpkg"
    # Full 90m squares are the frozen graph and analytical support.
    # DEA extraction must mask values to the mine footprint using the mine_footprints layer.
    blocks.to_file(gpkg_path, layer="blocks_90m", driver="GPKG", engine="pyogrio")
    footprint_gdf.to_file(gpkg_path, layer="mine_footprints", driver="GPKG", mode="a", engine="pyogrio")

    attribute_cols = [c for c in blocks.columns if c != "geometry"]
    attrs = pd.DataFrame(blocks[attribute_cols]).sort_values("block_id").reset_index(drop=True)
    parquet_path = output_dir / "block_attributes.parquet"
    write_parquet_or_fail(attrs, parquet_path, args.allow_csv_fallback)

    edges = edges.sort_values(["MineID", "source_block_id", "target_block_id"]).reset_index(drop=True)
    edges.to_csv(output_dir / "block_graph_edges.csv", index=False, encoding="utf-8-sig")
    splits = splits.sort_values(["fold_id", "role", "block_id"]).reset_index(drop=True)
    splits.to_csv(output_dir / "leave_one_mine_out_splits.csv", index=False, encoding="utf-8-sig")
    summary_df.sort_values("MineN").to_csv(
        output_dir / "stage2_block_summary.csv", index=False, encoding="utf-8-sig"
    )

    config = {
        "script_version": VERSION,
        "primary_support": "90 m x 90 m globally anchored grid in EPSG:9473",
        "grid_size_m": GRID_SIZE_M,
        "minimum_footprint_coverage": MIN_FOOTPRINT_COVERAGE,
        "exposure_name": "mapped rehabilitation implementation fraction",
        "exposure_formula": "Area(block intersect rehabilitation) / Area(block intersect union(rehabilitation, disturbance))",
        "exposure_interpretation": "noncausal snapshot exposure",
        "graph": "within-mine queen contiguity from eight grid-index neighbours",
        "graph_frozen_before_dea": True,
        "split": "leave-one-mine-out; held-out mine=test, all other mines=development_pool",
        "full_block_layer": "blocks_90m",
        "eo_mask_rule": "mask each full 90 m block to the corresponding mine_footprint during DEA extraction",
        "target_crs": TARGET_CRS,
    }
    (output_dir / "stage2_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    output_files = [
        gpkg_path,
        parquet_path,
        output_dir / "block_graph_edges.csv",
        output_dir / "leave_one_mine_out_splits.csv",
        output_dir / "stage2_block_summary.csv",
        output_dir / "stage2_config.json",
    ]
    manifest = {
        "script_version": VERSION,
        "run_utc": utc_now(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "raw_stage1_inputs_modified": False,
        "target_crs": TARGET_CRS,
        "grid_size_m": GRID_SIZE_M,
        "minimum_footprint_coverage": MIN_FOOTPRINT_COVERAGE,
        "mine_count": int(blocks["MineID"].nunique()),
        "block_count": int(len(blocks)),
        "edge_count": int(len(edges)),
        "fold_count": int(splits["fold_id"].nunique()),
        "graph_frozen_before_dea": True,
        "exposure_name": "mapped rehabilitation implementation fraction",
        "source_files": {
            "L03_cleaned.gpkg": sha256_file(l03_path),
            "L04_cleaned.gpkg": sha256_file(l04_path),
            "stage1_run_manifest.json": sha256_file(stage1_manifest) if stage1_manifest.exists() else None,
        },
        "software": {
            "python": platform.python_version(),
            "geopandas": gpd.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "shapely": shapely.__version__,
        },
        "outputs": {},
        "validation": {
            "required_fields_present": True,
            "unique_block_ids": True,
            "valid_geometries": True,
            "queen_degree_matches_edge_list": True,
            "mapped_fraction_in_0_1": True,
            "five_lomo_folds": True,
        },
    }
    for p in output_files:
        if p.exists():
            manifest["outputs"][p.name] = {
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
    manifest_path = output_dir / "stage2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Add manifest's own file only after writing; self-hash is intentionally omitted.
    print(f"Stage 2A completed: {output_dir}")
    print(f"Mines: {manifest['mine_count']}")
    print(f"Blocks: {manifest['block_count']}")
    print(f"Queen edges: {manifest['edge_count']}")
    print(f"LOMO folds: {manifest['fold_count']}")
    print("Graph frozen before DEA: true")
    print("Raw Stage 1 inputs were read only and were not modified.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build frozen 90 m GeoDose-CP spatial supports, queen graph, and LOMO folds."
    )
    p.add_argument("--input", default=r"D:\G3_STAGE1_MIN_AUDIT")
    p.add_argument("--output", default=r"D:\G3_STAGE2_BLOCKS")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--allow-csv-fallback", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    try:
        main()
    except Stage2Error as exc:
        print(f"STAGE 2A FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
