from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import xarray as xr

SCRIPT_VERSION = "1.2.0-local-stac"


class Stage2BError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    years: tuple[int, ...] = (2023, 2024, 2025)
    output_crs: str = "EPSG:9473"
    resolution_m: float = 30.0
    raster_align_m: float = 0.0
    resampling: str = "nearest"
    min_scene_dry_fraction: float = 0.70
    min_valid_observations: int = 3
    min_valid_pixel_fraction: float = 0.70
    high_ue_threshold: float = 25.0
    high_ue_sensitivity_thresholds: tuple[float, ...] = (20.0, 25.0, 30.0)
    max_high_ue_fraction: float = 0.50
    max_water_fraction: float = 0.50
    min_years_required: int = 3
    block_layer: str = "blocks_90m"
    footprint_layer: str = "mine_footprints"
    fc_product: str = "ga_ls_fc_3"
    wo_product: str = "ga_ls_wo_3"
    stac_url: str = "https://explorer.dea.ga.gov.au/stac"
    stac_search_method: str = "GET"
    stac_search_spatial_filter: str = "bbox"
    stac_search_attempts: int = 5
    stac_retry_seconds: tuple[int, ...] = (5, 15, 30, 60)
    stac_timeout_seconds: int = 120
    access_method: str = "DEA Explorer STAC API GET/bbox search plus unsigned public AWS S3 COG streaming"
    io_threads: int = 4
    measurements: tuple[str, ...] = ("pv", "ue")
    mask_rule: str = "WOfS dry=True equivalent: water bitfield exactly 0"
    annual_composite: str = (
        "per-pixel temporal median across accepted clear-dry solar-day observations, "
        "then block spatial median"
    )
    selection_uses_pv_outcome: bool = False
    min_mine_valid_block_count: int = 100
    min_mine_valid_block_fraction: float = 0.10


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage2BError(message)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GeoDose-CP Stage 2B local DEA STAC pre-screen")
    p.add_argument("--input", default="inputs", help="Directory containing frozen Stage 2A files")
    p.add_argument("--output", default="outputs", help="Output directory")
    p.add_argument("--restart", action="store_true", help="Delete existing checkpoints and start again")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--connection-test", action="store_true", help="Test public DEA STAC and S3 access using a tiny load")
    return p.parse_args()


def load_frozen_inputs(input_dir: Path, cfg: Config):
    gpkg = input_dir / "mine_blocks_90m.gpkg"
    attrs_path = input_dir / "block_attributes.parquet"
    manifest_path = input_dir / "stage2_manifest.json"
    verification_path = input_dir / "STAGE2A_VERIFICATION.json"

    for p in (gpkg, attrs_path, manifest_path, verification_path):
        require(p.exists(), f"Missing frozen Stage 2A input: {p}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    require(verification.get("status") == "verified_complete", "Stage 2A is not verified_complete")
    require(manifest.get("graph_frozen_before_dea") is True, "Stage 2A graph was not frozen before DEA")
    require(manifest.get("mine_count") == 5, "Expected exactly five frozen mines")
    require(manifest.get("block_count") == 27042, "Unexpected frozen block count")
    require(manifest.get("target_crs") == cfg.output_crs, "Unexpected Stage 2A CRS")

    # Verify packaged copies against the frozen manifest.
    expected_outputs = manifest.get("outputs", {})
    for name in ("mine_blocks_90m.gpkg", "block_attributes.parquet"):
        expected = expected_outputs.get(name, {}).get("sha256")
        require(expected is not None, f"Stage 2A manifest lacks checksum for {name}")
        require(sha256_file(input_dir / name) == expected, f"Checksum mismatch for {name}")

    blocks = gpd.read_file(gpkg, layer=cfg.block_layer)
    footprints = gpd.read_file(gpkg, layer=cfg.footprint_layer)
    require(str(blocks.crs).upper() == cfg.output_crs, f"Blocks CRS is {blocks.crs}, not {cfg.output_crs}")
    require(str(footprints.crs).upper() == cfg.output_crs, "Footprint CRS mismatch")
    require(len(blocks) == 27042 and blocks["MineID"].nunique() == 5, "Frozen block table changed")
    require(not blocks["block_id"].duplicated().any(), "Duplicate frozen block_id")
    require(len(footprints) == 5 and footprints["MineID"].nunique() == 5, "Footprints changed")
    require(blocks.geometry.is_valid.all() and (~blocks.geometry.is_empty).all(), "Invalid block geometry")
    require(footprints.geometry.is_valid.all() and (~footprints.geometry.is_empty).all(), "Invalid footprint geometry")

    return blocks, footprints, manifest, verification


def get_geobox(ds):
    # Supports both classic datacube xarray objects and odc-geo accessor objects.
    if hasattr(ds, "geobox"):
        return ds.geobox
    if hasattr(ds, "odc") and getattr(ds.odc, "geobox", None) is not None:
        return ds.odc.geobox
    raise Stage2BError("Could not obtain a GeoBox from the loaded DEA dataset")


def align_fc_wo(fc: xr.Dataset, wo: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset, dict]:
    # Products are derived from the same Landsat source and grouped by solar day.
    # Normalising timestamps to UTC dates makes matching robust to harmless
    # sub-second metadata differences between derived FC and WOfS items.
    fc_count = int(fc.sizes.get("time", 0))
    wo_count = int(wo.sizes.get("time", 0))
    fc_dates = pd.to_datetime(fc.time.values).normalize().to_numpy(dtype="datetime64[ns]")
    wo_dates = pd.to_datetime(wo.time.values).normalize().to_numpy(dtype="datetime64[ns]")
    require(len(np.unique(fc_dates)) == len(fc_dates), "Duplicate FC dates after solar-day grouping")
    require(len(np.unique(wo_dates)) == len(wo_dates), "Duplicate WOfS dates after solar-day grouping")
    fc = fc.assign_coords(time=fc_dates)
    wo = wo.assign_coords(time=wo_dates)
    fc2, wo2 = xr.align(fc, wo, join="inner")
    aligned_count = int(fc2.sizes.get("time", 0))
    require(aligned_count > 0, "No common FC/WOfS solar-day observations")
    diagnostics = {
        "fc_solar_day_count_before_alignment": fc_count,
        "wo_solar_day_count_before_alignment": wo_count,
        "aligned_solar_day_count": aligned_count,
        "fc_solar_days_without_wo": max(fc_count - aligned_count, 0),
        "wo_solar_days_without_fc": max(wo_count - aligned_count, 0),
    }
    return fc2, wo2, diagnostics


def build_mine_year_query(footprint, year: int, cfg: Config) -> dict:
    xmin, ymin, xmax, ymax = footprint.bounds
    pad = cfg.resolution_m
    return {
        "x": (xmin - pad, xmax + pad),
        "y": (ymin - pad, ymax + pad),
        "time": (f"{year}-01-01", f"{year}-12-31"),
        "output_crs": cfg.output_crs,
        "resolution": cfg.resolution_m,
        "anchor": "edge",
        "groupby": "solar_day",
        "resampling": cfg.resampling,
    }


def wofs_fuser(dest: np.ndarray, src: np.ndarray) -> None:
    """Official DEA WOfS bit-field fusion logic for overlapping same-day scenes."""
    empty = (dest & 1).astype(bool)
    both = ~empty & ~((src & 1).astype(bool))
    dest[empty] = src[empty]
    dest[both] |= src[both]


def footprint_to_wgs84(footprint, cfg: Config):
    return gpd.GeoSeries([footprint], crs=cfg.output_crs).to_crs("EPSG:4326").iloc[0]


def search_collection_items(catalog, collection: str, footprint, year: int, cfg: Config):
    """Search DEA STAC using the documented GET/bbox pattern with bounded retries.

    PySTAC Client defaults to POST. In some Windows/proxy and transient DEA API
    conditions, the landing page succeeds but POST /search returns an opaque
    APIError. DEA's public examples use bbox queries; forcing GET also produces
    a reproducible URL and avoids sending a complex polygon body.
    """
    geom4326 = footprint_to_wgs84(footprint, cfg)
    bbox = [float(v) for v in geom4326.bounds]
    last_error = None

    for attempt in range(1, cfg.stac_search_attempts + 1):
        try:
            search = catalog.search(
                method=cfg.stac_search_method,
                collections=[collection],
                bbox=bbox,
                datetime=f"{year}-01-01/{year}-12-31",
                limit=100,
            )
            items = list(search.items())
            require(items, f"No STAC items for {collection}, {year}, bbox={bbox}")
            # Deterministic ordering before grouping/fusing.
            items.sort(key=lambda item: (str(item.datetime or item.properties.get("datetime", "")), item.id))
            return items
        except Stage2BError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= cfg.stac_search_attempts:
                break
            delay = cfg.stac_retry_seconds[min(attempt - 1, len(cfg.stac_retry_seconds) - 1)]
            print(
                f"DEA STAC search attempt {attempt}/{cfg.stac_search_attempts} failed "
                f"for {collection} {year}; retrying in {delay}s. "
                f"Error: {type(exc).__name__}: {exc!r}",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise Stage2BError(
        "DEA STAC search failed after "
        f"{cfg.stac_search_attempts} attempts for collection={collection}, year={year}, "
        f"method={cfg.stac_search_method}, bbox={bbox}. "
        f"Original error: {type(last_error).__name__}: {last_error!r}"
    ) from last_error


def validate_assets(items, required: set[str], collection: str) -> None:
    available = set().union(*(set(item.assets) for item in items))
    require(required.issubset(available), f"{collection} lacks required STAC assets: {required - available}")


def stac_band_config(cfg: Config) -> dict:
    return {
        cfg.fc_product: {
            "assets": {
                "pv": {"data_type": "uint8", "nodata": 255, "unit": "percent"},
                "ue": {"data_type": "uint8", "nodata": 255, "unit": "1"},
            },
            "warnings": "ignore",
        },
        cfg.wo_product: {
            "assets": {
                "water": {"data_type": "uint8", "nodata": 1, "unit": "1"},
            },
            "warnings": "ignore",
        },
    }


def load_mine_year(catalog, footprint, year: int, cfg: Config):
    import odc.stac

    query = build_mine_year_query(footprint, year, cfg)
    fc_items = search_collection_items(catalog, cfg.fc_product, footprint, year, cfg)
    wo_items = search_collection_items(catalog, cfg.wo_product, footprint, year, cfg)
    validate_assets(fc_items, {"pv", "ue"}, cfg.fc_product)
    validate_assets(wo_items, {"water"}, cfg.wo_product)

    common = dict(
        crs=query["output_crs"],
        resolution=query["resolution"],
        anchor=query["anchor"],
        x=query["x"],
        y=query["y"],
        groupby=query["groupby"],
        fail_on_error=True,
        pool=cfg.io_threads,
        stac_cfg=stac_band_config(cfg),
    )
    fc = odc.stac.load(
        fc_items,
        bands=list(cfg.measurements),
        resampling={"pv": cfg.resampling, "ue": cfg.resampling},
        **common,
    )
    require(fc.sizes.get("time", 0) > 0, f"No loaded {cfg.fc_product} observations for {year}")

    wo = odc.stac.load(
        wo_items,
        bands=["water"],
        like=fc,
        groupby=query["groupby"],
        resampling={"water": cfg.resampling},
        fail_on_error=True,
        pool=cfg.io_threads,
        stac_cfg=stac_band_config(cfg),
        fuse_func={"water": "stage2b_local_stac_prescreen.wofs_fuser"},
    )
    require(wo.sizes.get("time", 0) > 0, f"No loaded {cfg.wo_product} observations for {year}")
    fc, wo, diagnostics = align_fc_wo(fc, wo)
    diagnostics.update({
        "fc_stac_item_count": len(fc_items),
        "wo_stac_item_count": len(wo_items),
        "stac_url": cfg.stac_url,
        "stac_search_method": cfg.stac_search_method,
        "stac_search_spatial_filter": cfg.stac_search_spatial_filter,
    })
    return fc, wo, diagnostics

def rasterize_support(mine_blocks: gpd.GeoDataFrame, footprint, ds: xr.Dataset):
    geobox = get_geobox(ds)
    shape = tuple(geobox.shape)
    transform = geobox.transform

    block_codes = np.arange(1, len(mine_blocks) + 1, dtype=np.int32)
    block_raster = rasterio.features.rasterize(
        ((geom, int(code)) for geom, code in zip(mine_blocks.geometry, block_codes)),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="int32",
    )
    footprint_mask = rasterio.features.rasterize(
        [(footprint, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    block_raster[~footprint_mask] = 0

    counts = np.bincount(block_raster.ravel(), minlength=len(mine_blocks) + 1)
    require(np.all(counts[1:] > 0), "At least one retained block contains no within-footprint 30 m pixel centre")
    return block_raster, counts


def safe_nanmedian(arr: np.ndarray, axis=None):
    if arr.size == 0 or not np.isfinite(arr).any():
        if axis is None:
            return np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(arr, axis=axis)


def aggregate_one_block_year(
    block_row,
    code: int,
    block_raster: np.ndarray,
    pv: np.ndarray,
    ue: np.ndarray,
    water: np.ndarray,
    scene_dates: np.ndarray,
    year: int,
    cfg: Config,
) -> dict:
    pix = np.flatnonzero(block_raster.ravel() == code)
    n_pix = int(len(pix))

    pv_b = pv.reshape(pv.shape[0], -1)[:, pix]
    ue_b = ue.reshape(ue.shape[0], -1)[:, pix]
    wo_b = water.reshape(water.shape[0], -1)[:, pix]

    fc_valid = (
        np.isfinite(pv_b)
        & np.isfinite(ue_b)
        & (pv_b >= 0)
        & (pv_b <= 100)
        & (ue_b >= 0)
        & (ue_b <= 127)
    )
    dry = wo_b == 0
    wet = wo_b == 128
    clear = dry | wet
    valid_dry = dry & fc_valid

    per_scene_valid_fraction = valid_dry.sum(axis=1) / n_pix
    accepted_scene = per_scene_valid_fraction >= cfg.min_scene_dry_fraction
    valid_observation_count = int(accepted_scene.sum())

    if valid_observation_count > 0:
        valid_acc = valid_dry[accepted_scene]
        pv_acc = np.where(valid_acc, pv_b[accepted_scene], np.nan)
        ue_acc = np.where(valid_acc, ue_b[accepted_scene], np.nan)
        counts_per_pixel = valid_acc.sum(axis=0)
        composite_pixel_ok = counts_per_pixel >= cfg.min_valid_observations
        valid_pixel_fraction = float(composite_pixel_ok.mean())

        pv_pixel_median = safe_nanmedian(pv_acc, axis=0)
        ue_pixel_median = safe_nanmedian(ue_acc, axis=0)
        pv_pixel_median[~composite_pixel_ok] = np.nan
        ue_pixel_median[~composite_pixel_ok] = np.nan
        pv_median = float(safe_nanmedian(pv_pixel_median))
        ue_median = float(safe_nanmedian(ue_pixel_median))
        median_valid_observations_per_pixel = float(np.median(counts_per_pixel))
        min_valid_observations_per_pixel = int(counts_per_pixel.min())
    else:
        valid_pixel_fraction = 0.0
        pv_pixel_median = np.full(n_pix, np.nan)
        ue_pixel_median = np.full(n_pix, np.nan)
        pv_median = np.nan
        ue_median = np.nan
        median_valid_observations_per_pixel = 0.0
        min_valid_observations_per_pixel = 0

    high_fracs = {}
    valid_ue_pixels = np.isfinite(ue_pixel_median)
    for threshold in cfg.high_ue_sensitivity_thresholds:
        key = f"high_ue_fraction_{int(threshold)}"
        high_fracs[key] = (
            float((ue_pixel_median[valid_ue_pixels] >= threshold).mean())
            if valid_ue_pixels.any()
            else np.nan
        )

    clear_count = int(clear.sum())
    water_fraction = float(wet.sum() / clear_count) if clear_count > 0 else np.nan
    unclear_fraction = float(1.0 - clear_count / clear.size) if clear.size > 0 else np.nan
    pv_out_of_range_fraction = float(
        ((np.isfinite(pv_b)) & ((pv_b < 0) | (pv_b > 100))).sum() / pv_b.size
    )

    high_main = high_fracs[f"high_ue_fraction_{int(cfg.high_ue_threshold)}"]
    passes = bool(
        valid_observation_count >= cfg.min_valid_observations
        and valid_pixel_fraction >= cfg.min_valid_pixel_fraction
        and np.isfinite(pv_median)
        and np.isfinite(ue_median)
        and np.isfinite(water_fraction)
        and water_fraction <= cfg.max_water_fraction
        and np.isfinite(high_main)
        and high_main <= cfg.max_high_ue_fraction
    )

    result = {
        "block_id": block_row.block_id,
        "MineID": block_row.MineID,
        "MineN": block_row.MineN,
        "ReportYr": int(block_row.ReportYr),
        "year": int(year),
        "mapped_rehabilitation_fraction": float(block_row.mapped_rehabilitation_fraction),
        "graph_degree": int(block_row.graph_degree),
        "graph_component_id": str(block_row.graph_component_id),
        "graph_component_size": int(block_row.graph_component_size),
        "fold_id": block_row.fold_id,
        "within_footprint_pixel_count": n_pix,
        "total_solar_day_observations": int(len(scene_dates)),
        "valid_observation_count": valid_observation_count,
        "valid_pixel_fraction": valid_pixel_fraction,
        "median_accepted_scene_valid_fraction": (
            float(np.median(per_scene_valid_fraction[accepted_scene]))
            if accepted_scene.any()
            else 0.0
        ),
        "median_valid_observations_per_pixel": median_valid_observations_per_pixel,
        "min_valid_observations_per_pixel": min_valid_observations_per_pixel,
        "pv_median": pv_median,
        "ue_median": ue_median,
        "water_fraction": water_fraction,
        "unclear_fraction": unclear_fraction,
        "pv_out_of_range_fraction": pv_out_of_range_fraction,
        "block_year_eligible": passes,
    }
    result.update(high_fracs)
    return result


def process_mine_year(catalog, mine_blocks, footprint_row, year: int, cfg: Config):
    fc, wo, alignment = load_mine_year(catalog, footprint_row.geometry, year, cfg)
    block_raster, _ = rasterize_support(mine_blocks, footprint_row.geometry, fc)

    pv = fc["pv"].values.astype("float32", copy=False)
    ue = fc["ue"].values.astype("float32", copy=False)
    water = wo["water"].values.astype("uint8", copy=False)
    scene_dates = pd.to_datetime(fc.time.values).to_numpy()

    require(pv.shape == ue.shape == water.shape, "FC and WOfS array shapes differ")

    rows = []
    for code, row in enumerate(mine_blocks.itertuples(index=False), start=1):
        rows.append(
            aggregate_one_block_year(
                row, code, block_raster, pv, ue, water, scene_dates, year, cfg
            )
        )

    scene_rows = []
    for t in scene_dates:
        scene_rows.append(
            {
                "MineID": footprint_row.MineID,
                "MineN": footprint_row.MineN,
                "year": year,
                "solar_day": pd.Timestamp(t).date().isoformat(),
                **alignment,
            }
        )
    return rows, scene_rows


def mine_summary(block_year: pd.DataFrame, blocks: gpd.GeoDataFrame, footprints: gpd.GeoDataFrame, cfg: Config):
    year_count = len(cfg.years)
    by_block = block_year.groupby(["MineID", "block_id"], sort=False)
    block_status = by_block.agg(
        observed_years=("year", "nunique"),
        eligible_years=("block_year_eligible", "sum"),
        missing_pv_years=("pv_median", lambda s: int(s.isna().sum())),
        persistent_high_ue_years=(
            f"high_ue_fraction_{int(cfg.high_ue_threshold)}",
            lambda s: int((s > cfg.max_high_ue_fraction).sum()),
        ),
        water_dominated_years=("water_fraction", lambda s: int((s > cfg.max_water_fraction).sum())),
    ).reset_index()
    block_status["all_years_eligible"] = (
        (block_status["observed_years"] == year_count)
        & (block_status["eligible_years"] == year_count)
    )
    block_status["persistently_high_ue"] = block_status["persistent_high_ue_years"] >= 2

    merged = block_year.merge(
        block_status[["MineID", "block_id", "all_years_eligible", "persistently_high_ue"]],
        on=["MineID", "block_id"],
        how="left",
        validate="many_to_one",
    )

    rows = []
    fp_centroids = footprints.set_index("MineID").geometry.centroid
    for mine_id, group in merged.groupby("MineID", sort=False):
        block_ids = group["block_id"].unique()
        n_blocks = len(block_ids)
        all_years = group.drop_duplicates("block_id")["all_years_eligible"]
        persistent = group.drop_duplicates("block_id")["persistently_high_ue"]
        mine_blocks = blocks[blocks["MineID"] == mine_id]
        exp = mine_blocks["mapped_rehabilitation_fraction"].to_numpy()
        centroid = fp_centroids.loc[mine_id]

        rows.append(
            {
                "MineID": mine_id,
                "MineN": group["MineN"].iloc[0],
                "total_blocks": n_blocks,
                "valid_block_count_all_years": int(all_years.sum()),
                "valid_block_fraction_all_years": float(all_years.mean()),
                "missing_block_year_rate": float(group["pv_median"].isna().mean()),
                "ineligible_block_year_rate": float((~group["block_year_eligible"]).mean()),
                "median_valid_observation_count": float(group["valid_observation_count"].median()),
                "median_valid_pixel_fraction": float(group["valid_pixel_fraction"].median()),
                "median_ue": float(group["ue_median"].median(skipna=True)),
                "median_high_ue_fraction_25": float(group["high_ue_fraction_25"].median(skipna=True)),
                "persistent_high_ue_block_fraction": float(persistent.mean()),
                "water_dominated_block_year_rate": float((group["water_fraction"] > cfg.max_water_fraction).mean()),
                "median_water_fraction": float(group["water_fraction"].median(skipna=True)),
                "exposure_min": float(np.min(exp)),
                "exposure_p05": float(np.quantile(exp, 0.05)),
                "exposure_median": float(np.median(exp)),
                "exposure_p95": float(np.quantile(exp, 0.95)),
                "exposure_max": float(np.max(exp)),
                "interior_exposure_fraction": float(((exp > 0) & (exp < 1)).mean()),
                "centroid_x": float(centroid.x),
                "centroid_y": float(centroid.y),
            }
        )

    summary = pd.DataFrame(rows)
    # Quality score uses no PV outcome value. It uses availability, missingness and quality only.
    summary["valid_count_rank"] = summary["valid_block_count_all_years"].rank(pct=True, method="average")
    summary["missingness_rank"] = (1 - summary["missing_block_year_rate"]).rank(pct=True, method="average")
    summary["ue_rank"] = (-summary["median_ue"]).rank(pct=True, method="average")
    summary["quality_score"] = (
        0.55 * summary["valid_count_rank"]
        + 0.25 * summary["missingness_rank"]
        + 0.20 * summary["ue_rank"]
    )
    return merged, summary


def minmax(values: pd.Series) -> pd.Series:
    vmin, vmax = float(values.min()), float(values.max())
    if math.isclose(vmin, vmax):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - vmin) / (vmax - vmin)


def select_three_mines(summary: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Greedy selection: quality is dominant; geographic and exposure diversity are tie-breakers.
    # A non-vacuity gate prevents the workflow from forcing three unusable mines.
    work = summary.copy().reset_index(drop=True)
    work["mine_prescreen_usable"] = (
        (work["valid_block_count_all_years"] >= cfg.min_mine_valid_block_count)
        & (work["valid_block_fraction_all_years"] >= cfg.min_mine_valid_block_fraction)
        & np.isfinite(work["median_ue"])
    )
    usable = work.index[work["mine_prescreen_usable"]]
    require(
        len(usable) >= 3,
        "Fewer than three mines pass the prespecified Stage 2B mine-level non-vacuity gate",
    )
    selected: list[int] = []

    first = int(work.loc[usable, "quality_score"].idxmax())
    selected.append(first)

    while len(selected) < 3:
        candidates = usable.difference(selected)
        chosen = work.loc[selected]
        scores = []
        for idx in candidates:
            row = work.loc[idx]
            geo_dist = np.sqrt(
                (chosen["centroid_x"].to_numpy() - row["centroid_x"]) ** 2
                + (chosen["centroid_y"].to_numpy() - row["centroid_y"]) ** 2
            ).min()
            exp_dist = np.abs(chosen["exposure_median"].to_numpy() - row["exposure_median"]).min()
            scores.append((idx, row["quality_score"], geo_dist, exp_dist))
        temp = pd.DataFrame(scores, columns=["idx", "quality", "geo", "exposure"]).set_index("idx")
        temp["geo_norm"] = minmax(temp["geo"])
        temp["exposure_norm"] = minmax(temp["exposure"])
        # Do not let diversity rescue a very poor-quality mine.
        quality_floor = float(temp["quality"].max() - 0.20)
        eligible = temp[temp["quality"] >= quality_floor].copy()
        eligible["selection_score"] = (
            0.75 * eligible["quality"]
            + 0.15 * eligible["geo_norm"]
            + 0.10 * eligible["exposure_norm"]
        )
        selected.append(int(eligible["selection_score"].idxmax()))

    work["selected_for_real_demonstration"] = False
    work.loc[selected, "selected_for_real_demonstration"] = True
    work["selection_order"] = np.nan
    for order, idx in enumerate(selected, start=1):
        work.loc[idx, "selection_order"] = order
    work["selection_basis"] = (
        "DEA availability/quality first; geographic and mapped-exposure diversity as tie-breakers; "
        "PV outcome values excluded"
    )
    selected_table = work[work["selected_for_real_demonstration"]].sort_values("selection_order")
    return work, selected_table


def config_hash(cfg: Config) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_paths(checkpoint_dir: Path, mine_id: str, year: int):
    token = mine_id.strip("{}").replace("-", "")[:16]
    stem = f"{token}_{year}"
    return (
        checkpoint_dir / f"{stem}_block_year.parquet",
        checkpoint_dir / f"{stem}_scenes.csv",
        checkpoint_dir / f"{stem}_checkpoint.json",
    )


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def load_checkpoint(
    checkpoint_dir: Path, mine_id: str, year: int, expected_rows: int, cfg: Config, source_hash: str
):
    block_path, scene_path, meta_path = checkpoint_paths(checkpoint_dir, mine_id, year)
    if not (block_path.exists() and scene_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    require(meta.get("config_hash") == config_hash(cfg), f"Checkpoint config mismatch for {mine_id} {year}")
    require(meta.get("source_hash") == source_hash, f"Checkpoint source mismatch for {mine_id} {year}")
    require(meta.get("expected_rows") == expected_rows, f"Checkpoint row expectation mismatch for {mine_id} {year}")
    require(meta.get("block_sha256") == sha256_file(block_path), f"Checkpoint block checksum failed for {mine_id} {year}")
    require(meta.get("scene_sha256") == sha256_file(scene_path), f"Checkpoint scene checksum failed for {mine_id} {year}")
    block_df = pd.read_parquet(block_path)
    scene_df = pd.read_csv(scene_path, encoding="utf-8-sig")
    require(len(block_df) == expected_rows, f"Incomplete checkpoint for {mine_id} {year}")
    require(block_df["year"].eq(year).all(), f"Checkpoint year mismatch for {mine_id} {year}")
    return block_df.to_dict("records"), scene_df.to_dict("records")


def save_checkpoint(
    checkpoint_dir: Path, mine_id: str, year: int, rows: list[dict], scenes: list[dict], cfg: Config, source_hash: str
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    block_path, scene_path, meta_path = checkpoint_paths(checkpoint_dir, mine_id, year)
    atomic_write_parquet(pd.DataFrame(rows), block_path)
    atomic_write_csv(pd.DataFrame(scenes), scene_path)
    meta = {
        "mine_id": mine_id,
        "year": year,
        "expected_rows": len(rows),
        "config_hash": config_hash(cfg),
        "source_hash": source_hash,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "block_sha256": sha256_file(block_path),
        "scene_sha256": sha256_file(scene_path),
    }
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(meta_path)


def write_environment_inventory(catalog, output_dir: Path, cfg: Config) -> Path:
    import platform
    import importlib.metadata as metadata

    # Confirm both public collections are visible through the unauthenticated STAC API.
    collection_ids = {c.id for c in catalog.get_collections()}
    require(cfg.fc_product in collection_ids, f"DEA STAC lacks collection {cfg.fc_product}")
    require(cfg.wo_product in collection_ids, f"DEA STAC lacks collection {cfg.wo_product}")

    def package_version(name: str):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    inventory = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "access_method": cfg.access_method,
        "stac_url": cfg.stac_url,
        "stac_search_method": cfg.stac_search_method,
        "stac_search_spatial_filter": cfg.stac_search_spatial_filter,
        "stac_search_attempts": cfg.stac_search_attempts,
        "stac_timeout_seconds": cfg.stac_timeout_seconds,
        "aws_unsigned": True,
        "packages": {
            "odc-stac": package_version("odc-stac"),
            "odc-geo": package_version("odc-geo"),
            "pystac-client": package_version("pystac-client"),
            "boto3": package_version("boto3"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "geopandas": package_version("geopandas"),
            "xarray": package_version("xarray"),
            "rasterio": package_version("rasterio"),
            "pyarrow": package_version("pyarrow"),
        },
        "products": [cfg.fc_product, cfg.wo_product],
        "measurements": {cfg.fc_product: list(cfg.measurements), cfg.wo_product: ["water"]},
        "explicit_time_query_for_both_products": True,
        "resampling": cfg.resampling,
        "output_crs": cfg.output_crs,
        "resolution_m": cfg.resolution_m,
        "wofs_fuser": "official DEA bit-field fusion logic",
    }
    path = output_dir / "dea_environment_inventory.json"
    path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return path


def configure_public_dea_access(cfg: Config):
    import odc.stac
    import pystac_client

    odc.stac.configure_rio(
        cloud_defaults=True,
        aws={"aws_unsigned": True, "region_name": "ap-southeast-2"},
    )
    return pystac_client.Client.open(cfg.stac_url, timeout=(30, cfg.stac_timeout_seconds))


def connection_test(input_dir: Path, cfg: Config) -> None:
    blocks, footprints, _, _ = load_frozen_inputs(input_dir, cfg)
    catalog = configure_public_dea_access(cfg)
    fp = footprints.sort_values("MineN").iloc[0]
    geom = fp.geometry
    center = geom.centroid
    tiny = center.buffer(90.0).envelope.intersection(geom)
    require(not tiny.is_empty, "Could not construct connection-test area")
    fc, wo, diagnostics = load_mine_year(catalog, tiny, 2025, cfg)
    require(fc.sizes.get("time", 0) > 0 and wo.sizes.get("time", 0) > 0, "Tiny STAC load returned no data")
    require({"pv", "ue"}.issubset(fc.data_vars), "FC test load lacks pv/ue")
    require("water" in wo.data_vars, "WOfS test load lacks water")
    print("PUBLIC DEA STAC CONNECTION TEST PASSED")
    print(f"STAC: {cfg.stac_url}")
    print(f"Mine: {fp.MineN}")
    print(f"Aligned solar days in 2025: {diagnostics['aligned_solar_day_count']}")

def write_outputs(
    output_dir: Path,
    block_year: pd.DataFrame,
    scene_inventory: pd.DataFrame,
    mine_summary_df: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: Config,
    input_dir: Path,
    stage2_manifest: dict,
    runtime_seconds: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    final_names = [
        "stage2b_config.json", "dea_block_year_prescreen.parquet",
        "dea_block_year_prescreen.csv.gz", "dea_scene_inventory.csv",
        "dea_mine_prescreen_summary.csv", "selected_three_mines.csv",
        "Stage2B_Selection_Decision.md", "dea_environment_inventory.json",
        "stage2b_manifest.json", "STAGE2B_VERIFICATION.json",
    ]
    for name in final_names:
        path = output_dir / name
        if path.exists():
            path.unlink()

    config_path = output_dir / "stage2b_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    block_year_path = output_dir / "dea_block_year_prescreen.parquet"
    block_year.to_parquet(block_year_path, index=False)
    block_year.to_csv(output_dir / "dea_block_year_prescreen.csv.gz", index=False, compression="gzip")
    scene_inventory.to_csv(output_dir / "dea_scene_inventory.csv", index=False, encoding="utf-8-sig")
    mine_summary_df.to_csv(output_dir / "dea_mine_prescreen_summary.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_dir / "selected_three_mines.csv", index=False, encoding="utf-8-sig")

    decision_lines = [
        "# GeoDose-CP Stage 2B DEA Pre-screen Decision",
        "",
        "## Selected mines",
        "",
    ]
    for row in selected.itertuples(index=False):
        decision_lines.append(
            f"{int(row.selection_order)}. **{row.MineN}** — "
            f"{row.valid_block_count_all_years:,} blocks eligible in all three years; "
            f"missing block-year rate {row.missing_block_year_rate:.2%}; "
            f"median UE {row.median_ue:.2f}."
        )
    decision_lines.extend(
        [
            "",
            "Selection used DEA availability, missingness, UE/water quality, geography, and the frozen mapped-exposure distribution.",
            "PV outcome values were not used to choose mines.",
            "",
            "## Interpretation",
            "",
            "The selected mines are for a product-aware EO deployment demonstration, not observational proof of a causal rehabilitation effect.",
        ]
    )
    (output_dir / "Stage2B_Selection_Decision.md").write_text("\n".join(decision_lines), encoding="utf-8")

    output_files = [p for p in output_dir.iterdir() if p.is_file() and p.name != "stage2b_manifest.json"]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input_stage2_manifest_sha256": sha256_file(input_dir / "stage2_manifest.json"),
        "input_mine_blocks_sha256": sha256_file(input_dir / "mine_blocks_90m.gpkg"),
        "input_block_attributes_sha256": sha256_file(input_dir / "block_attributes.parquet"),
        "stage2_graph_frozen_before_dea": bool(stage2_manifest["graph_frozen_before_dea"]),
        "years": list(cfg.years),
        "mine_count": int(block_year["MineID"].nunique()),
        "block_count": int(block_year["block_id"].nunique()),
        "block_year_count": int(len(block_year)),
        "selected_mine_count": int(len(selected)),
        "selection_uses_pv_outcome": False,
        "runtime_seconds": runtime_seconds,
        "stage2b_script_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
            for p in sorted(output_files)
        },
    }
    (output_dir / "stage2b_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def self_test() -> None:
    cfg = Config()
    # One synthetic block, four 30m pixels, four dates.
    block = type(
        "Block",
        (),
        dict(
            block_id="B1",
            MineID="M1",
            MineN="Mine",
            ReportYr=2026,
            mapped_rehabilitation_fraction=0.5,
            graph_degree=4,
            graph_component_id="M1_CC001",
            graph_component_size=10,
            fold_id="F1",
        ),
    )()
    block_raster = np.array([[1, 1], [1, 1]], dtype=np.int32)
    pv = np.array(
        [
            [[10, 20], [30, 40]],
            [[20, 30], [40, 50]],
            [[30, 40], [50, 60]],
            [[40, 50], [60, 70]],
        ],
        dtype=float,
    )
    ue = np.full_like(pv, 10.0)
    water = np.zeros_like(pv, dtype=np.uint8)
    dates = pd.date_range("2023-01-01", periods=4).to_numpy()
    result = aggregate_one_block_year(block, 1, block_raster, pv, ue, water, dates, 2023, cfg)
    require(result["graph_component_id"] == "M1_CC001", "Self-test component ID preservation failed")
    require(result["valid_observation_count"] == 4, "Self-test valid scene count failed")
    require(math.isclose(result["valid_pixel_fraction"], 1.0), "Self-test pixel coverage failed")
    require(math.isclose(result["pv_median"], 40.0), f"Self-test median failed: {result['pv_median']}")
    require(result["block_year_eligible"] is True, "Self-test eligibility failed")
    fake_footprint = type("Footprint", (), {"bounds": (0.0, 0.0, 90.0, 90.0)})()
    q = build_mine_year_query(fake_footprint, 2024, cfg)
    require(q["time"] == ("2024-01-01", "2024-12-31"), "Self-test explicit time query failed")
    require(q["resampling"] == "nearest", "Self-test resampling rule failed")
    require(q["anchor"] == "edge", "Self-test pixel anchor failed")
    d = np.array([[1, 0], [128, 1]], dtype=np.uint8)
    s = np.array([[0, 128], [0, 128]], dtype=np.uint8)
    wofs_fuser(d, s)
    require(d.tolist() == [[0, 128], [128, 128]], "Self-test WOfS fuser failed")
    print("SELF-TEST PASSED")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    cfg = Config()

    if args.connection_test:
        connection_test(input_dir, cfg)
        return

    if output_dir.exists() and args.restart:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    blocks, footprints, stage2_manifest, _ = load_frozen_inputs(input_dir, cfg)
    source_hash = sha256_file(input_dir / "mine_blocks_90m.gpkg")

    catalog = configure_public_dea_access(cfg)
    write_environment_inventory(catalog, output_dir, cfg)
    all_rows = []
    all_scenes = []
    start = time.time()

    for fp in footprints.sort_values("MineN").itertuples(index=False):
        mine_blocks = blocks[blocks["MineID"] == fp.MineID].sort_values("block_id").copy()
        print(f"\nMine: {fp.MineN} | blocks={len(mine_blocks):,}")
        for year in cfg.years:
            cached = load_checkpoint(
                checkpoint_dir, fp.MineID, year, len(mine_blocks), cfg, source_hash
            )
            if cached is not None:
                rows, scenes = cached
                print(f"  {year}: resumed verified checkpoint ({len(rows):,} rows)")
            else:
                print(f"  Loading and processing {year} ...", flush=True)
                rows, scenes = process_mine_year(catalog, mine_blocks, fp, year, cfg)
                save_checkpoint(
                    checkpoint_dir, fp.MineID, year, rows, scenes, cfg, source_hash
                )
                print(f"  {year}: checkpoint saved; {len(scenes)} solar-day observations; {len(rows):,} block-year rows")
            all_rows.extend(rows)
            all_scenes.extend(scenes)

    block_year = pd.DataFrame(all_rows)
    scene_inventory = pd.DataFrame(all_scenes).drop_duplicates().sort_values(["MineN", "year", "solar_day"])
    require(len(block_year) == len(blocks) * len(cfg.years), "Incomplete block-year table")
    require(not block_year.duplicated(["block_id", "year"]).any(), "Duplicate block-year rows")

    block_year, summary = mine_summary(block_year, blocks, footprints, cfg)
    summary, selected = select_three_mines(summary, cfg)
    runtime = time.time() - start
    write_outputs(output_dir, block_year, scene_inventory, summary, selected, cfg, input_dir, stage2_manifest, runtime)

    print("\nSTAGE 2B EXTRACTION COMPLETE")
    print(f"Block-year rows: {len(block_year):,}")
    print("Selected mines:")
    for row in selected.sort_values("selection_order").itertuples(index=False):
        print(f"  {int(row.selection_order)}. {row.MineN}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Stage2BError as exc:
        print(f"STAGE 2B FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
