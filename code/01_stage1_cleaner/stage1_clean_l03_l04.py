#!/usr/bin/env python3
"""Minimum, auditable cleaner for NSW L03 Rehabilitation and L04 Disturbance.

Reads the verified v3 export without changing it, filters to five selected mines,
repairs polygon geometry, flags temporal anomalies and duplicates, reprojects to
EPSG:9473, and dissolves by MineID/layer/ReportYr so overlapping polygons are
not double counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

VERSION = "1.1.0"
TARGET_CRS = "EPSG:9473"
LAYER_SPECS = {
    "L03": {"folder": "L03_rehabilitation", "event_years": ["YrLndEs", "YrVegEs"]},
    "L04": {"folder": "L04_disturbance", "event_years": ["DistYr"]},
}
CORE_FIELDS = ["OBJECTID", "MineID", "MineN", "ReportYr"]


class AuditError(RuntimeError):
    pass


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_mine_id(value: object) -> str:
    text = clean_text(value).upper()
    return text


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def find_vector_source(layer_dir: Path) -> Path:
    preferred = layer_dir / "features_geojson4326.geojson"
    if preferred.exists():
        return preferred
    # A user may have converted the source to a GeoPackage manually; permit it.
    gpkg = layer_dir / "features.gpkg"
    if gpkg.exists():
        return gpkg
    raise AuditError(
        f"No readable vector file found in {layer_dir}. Expected "
        "features_geojson4326.geojson from exporter v3."
    )


def read_attributes(root: Path, code: str) -> pd.DataFrame:
    spec = LAYER_SPECS[code]
    path = root / spec["folder"] / "attributes.csv"
    if not path.exists():
        raise AuditError(f"Missing {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing = [c for c in CORE_FIELDS if c not in df.columns]
    if missing:
        raise AuditError(f"{path} is missing fields: {missing}")
    for c in ["MineID", "MineN", "ReportYr", "OBJECTID"]:
        df[c] = df[c].map(clean_text)
    df["MineID_norm"] = df["MineID"].map(normalize_mine_id)
    df["ReportYr_num"] = pd.to_numeric(df["ReportYr"], errors="coerce")
    return df


def prepare_selection(root: Path, selection_path: Path) -> None:
    l03 = read_attributes(root, "L03")
    l04 = read_attributes(root, "L04")
    # Hard-core fields only for ranking; geometry is checked later.
    l03 = l03[(l03.MineID_norm != "") & (l03.MineN != "") & l03.ReportYr_num.notna()].copy()
    l04 = l04[(l04.MineID_norm != "") & (l04.MineN != "") & l04.ReportYr_num.notna()].copy()

    def summarize(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return (
            df.groupby("MineID_norm", as_index=False)
            .agg(
                **{
                    f"{prefix}_features": ("OBJECTID", "size"),
                    f"{prefix}_report_years": ("ReportYr_num", "nunique"),
                    f"{prefix}_mine_name": ("MineN", lambda s: s.value_counts().index[0]),
                }
            )
        )

    summary = summarize(l03, "l03").merge(summarize(l04, "l04"), on="MineID_norm", how="inner")
    years03 = l03.groupby("MineID_norm")["ReportYr_num"].apply(lambda s: set(s.dropna().astype(int)))
    years04 = l04.groupby("MineID_norm")["ReportYr_num"].apply(lambda s: set(s.dropna().astype(int)))
    def common_year_set(mine_id: str) -> list[int]:
        return sorted(years03.get(mine_id, set()) & years04.get(mine_id, set()))

    def consecutive_pairs(years: list[int]) -> list[str]:
        values = sorted(set(int(y) for y in years))
        value_set = set(values)
        return [f"{y}-{y+1}" for y in values if y + 1 in value_set]

    summary["_common_year_list"] = summary.MineID_norm.map(common_year_set)
    summary["common_report_years"] = summary["_common_year_list"].map(
        lambda ys: ";".join(map(str, ys))
    )
    summary["n_common_report_years"] = summary["_common_year_list"].map(len)
    summary["consecutive_common_pairs"] = summary["_common_year_list"].map(
        lambda ys: ";".join(consecutive_pairs(ys))
    )
    summary["n_consecutive_common_pairs"] = summary["consecutive_common_pairs"].map(
        lambda x: 0 if not x else len(x.split(";"))
    )
    summary["has_consecutive_common_pair"] = summary["n_consecutive_common_pairs"] > 0
    summary["MineN"] = summary["l03_mine_name"].where(summary["l03_mine_name"] != "", summary["l04_mine_name"])
    # Rank without using outcomes. Longitudinal feasibility comes first; polygon evidence breaks ties.
    summary["balanced_feature_count"] = summary[["l03_features", "l04_features"]].min(axis=1)
    summary = summary.sort_values(
        [
            "has_consecutive_common_pair",
            "n_consecutive_common_pairs",
            "n_common_report_years",
            "balanced_feature_count",
            "l03_features",
            "l04_features",
        ],
        ascending=False,
    )
    summary.insert(0, "include", 0)
    cols = [
        "include", "MineID_norm", "MineN", "l03_features", "l04_features",
        "l03_report_years", "l04_report_years", "n_common_report_years",
        "common_report_years", "has_consecutive_common_pair",
        "n_consecutive_common_pairs", "consecutive_common_pairs",
        "balanced_feature_count",
    ]
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    summary[cols].to_csv(selection_path, index=False, encoding="utf-8-sig")
    print(f"Created {selection_path}")
    print("Open it, set include=1 for exactly five mines, save, then run cleaning.")


def read_selected_ids(path: Path) -> list[str]:
    if not path.exists():
        raise AuditError(f"Selection file does not exist: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    if "include" not in df.columns or "MineID_norm" not in df.columns:
        raise AuditError("Selection CSV must contain include and MineID_norm columns")
    include = pd.to_numeric(df["include"], errors="coerce").fillna(0).astype(int)
    ids = [normalize_mine_id(v) for v in df.loc[include == 1, "MineID_norm"] if normalize_mine_id(v)]
    ids = list(dict.fromkeys(ids))
    if len(ids) != 5:
        raise AuditError(f"Exactly five unique mines must have include=1; found {len(ids)}")
    return ids


def polygonal_only(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = []
        for part in geom.geoms:
            p = polygonal_only(part)
            if p is None or p.is_empty:
                continue
            if isinstance(p, Polygon):
                polys.append(p)
            elif isinstance(p, MultiPolygon):
                polys.extend(list(p.geoms))
        if not polys:
            return None
        merged = unary_union(polys)
        return merged if isinstance(merged, (Polygon, MultiPolygon)) else polygonal_only(merged)
    return None


def parse_chart_date(series: pd.Series) -> pd.Series:
    text = series.map(clean_text)
    numeric = pd.to_numeric(text, errors="coerce")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    mask_num = numeric.notna()
    if mask_num.any():
        out.loc[mask_num] = pd.to_datetime(numeric.loc[mask_num], unit="ms", errors="coerce", utc=True)
    mask_other = (~mask_num) & (text != "")
    if mask_other.any():
        out.loc[mask_other] = pd.to_datetime(text.loc[mask_other], errors="coerce", utc=True)
    return out


def parse_submit_date(series: pd.Series) -> pd.Series:
    text = series.map(clean_text)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    fourteen = text.str.fullmatch(r"\d{14}")
    if fourteen.any():
        parsed = pd.to_datetime(text.loc[fourteen], format="%Y%m%d%H%M%S", errors="coerce", utc=True)
        out.loc[fourteen] = parsed
    other = (text != "") & (~fourteen)
    if other.any():
        out.loc[other] = pd.to_datetime(text.loc[other], errors="coerce", utc=True)
    return out


def canonical_geometry_hash(geom) -> str:
    if geom is None or geom.is_empty:
        return ""
    normalized = shapely.normalize(geom)
    return shapely.to_wkb(normalized, hex=True, output_dimension=2)


def load_selected_layer(root: Path, code: str, selected_ids: Sequence[str]) -> tuple[gpd.GeoDataFrame, Path]:
    layer_dir = root / LAYER_SPECS[code]["folder"]
    source = find_vector_source(layer_dir)
    gdf = gpd.read_file(source, engine="pyogrio")
    missing = [c for c in CORE_FIELDS if c not in gdf.columns]
    if missing:
        raise AuditError(f"{source} is missing fields: {missing}")
    for c in gdf.columns:
        if c != gdf.geometry.name and gdf[c].dtype == object:
            gdf[c] = gdf[c].map(clean_text)
    gdf["MineID_norm"] = gdf["MineID"].map(normalize_mine_id)
    gdf = gdf[gdf["MineID_norm"].isin(selected_ids)].copy()
    if gdf.empty:
        raise AuditError(f"No selected records found in {source}")
    if gdf.crs is None:
        raise AuditError(f"{source} has no CRS; do not guess it")

    # Independent alignment check: selected CSV and GeoJSON must contain the same
    # OBJECTID multiset. This catches accidental partial/cross-version inputs.
    attrs = read_attributes(root, code)
    attrs = attrs[attrs["MineID_norm"].isin(selected_ids)].copy()
    csv_ids = attrs["OBJECTID"].map(clean_text).tolist()
    geo_ids = gdf["OBJECTID"].map(clean_text).tolist()
    if sorted(csv_ids) != sorted(geo_ids):
        csv_set, geo_set = set(csv_ids), set(geo_ids)
        missing_geo = sorted(csv_set - geo_set)[:20]
        extra_geo = sorted(geo_set - csv_set)[:20]
        raise AuditError(
            f"{code} CSV/GeoJSON OBJECTID mismatch for selected mines. "
            f"CSV rows={len(csv_ids)}, GeoJSON rows={len(geo_ids)}, "
            f"missing_in_geojson={missing_geo}, extra_in_geojson={extra_geo}"
        )

    gdf["source_layer"] = code
    return gdf, source


def add_temporal_flags(gdf: gpd.GeoDataFrame, code: str) -> gpd.GeoDataFrame:
    now = pd.Timestamp.now(tz="UTC")
    current_year = now.year
    report = pd.to_numeric(gdf["ReportYr"].map(clean_text), errors="coerce")
    gdf["ReportYr_num"] = report.astype("Int64")
    gdf["flag_report_year_invalid"] = report.isna() | (report < 1900) | (report > current_year)

    event_invalid = pd.Series(False, index=gdf.index)
    event_future = pd.Series(False, index=gdf.index)
    event_summary = []
    for field in LAYER_SPECS[code]["event_years"]:
        if field not in gdf.columns:
            continue
        raw = gdf[field].map(clean_text)
        year = pd.to_numeric(raw, errors="coerce")
        # Zero/blank is treated as missing, not a fabricated year.
        provided = (raw != "") & (year != 0)
        invalid = provided & (year.isna() | (year < 1900) | (year > 2100))
        future = provided & year.notna() & report.notna() & (year > report)
        gdf[f"flag_{field}_invalid"] = invalid
        gdf[f"flag_{field}_future_vs_report"] = future
        event_invalid |= invalid
        event_future |= future
        event_summary.append(field)
    gdf["flag_event_year_invalid"] = event_invalid
    gdf["flag_event_year_future"] = event_future

    if "ChartDt" in gdf.columns:
        chart = parse_chart_date(gdf["ChartDt"])
        supplied = gdf["ChartDt"].map(clean_text) != ""
        gdf["ChartDt_parsed"] = chart.astype(str)
        gdf["flag_chart_date_invalid"] = supplied & (chart.isna() | (chart.dt.year <= 1900) | (chart > now))
    else:
        gdf["flag_chart_date_invalid"] = False

    if "SubmitDate" in gdf.columns:
        submit = parse_submit_date(gdf["SubmitDate"])
        supplied = gdf["SubmitDate"].map(clean_text) != ""
        gdf["SubmitDate_parsed"] = submit.astype(str)
        gdf["flag_submit_date_invalid"] = supplied & (submit.isna() | (submit.dt.year <= 1900) | (submit > now))
    else:
        gdf["flag_submit_date_invalid"] = False

    gdf["event_date_usable"] = ~(gdf["flag_event_year_invalid"] | gdf["flag_event_year_future"])
    return gdf


def clean_layer(root: Path, code: str, selected_ids: Sequence[str]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, Path]:
    gdf, source = load_selected_layer(root, code, selected_ids)
    # Raw-source records are never edited; all modifications occur in memory and new output files.
    for field in ["MineID", "MineN", "ReportYr", "OBJECTID"]:
        gdf[field] = gdf[field].map(clean_text)

    gdf["flag_blank_MineID"] = gdf["MineID"] == ""
    gdf["flag_blank_MineN"] = gdf["MineN"] == ""
    gdf["flag_blank_ReportYr"] = gdf["ReportYr"] == ""
    gdf["flag_placeholder_core"] = gdf[["flag_blank_MineID", "flag_blank_MineN", "flag_blank_ReportYr"]].any(axis=1)
    gdf["OBJECTID_num"] = pd.to_numeric(gdf["OBJECTID"], errors="coerce").astype("Int64")
    gdf["flag_objectid_invalid"] = gdf["OBJECTID_num"].isna()
    gdf["flag_objectid_duplicate"] = gdf["OBJECTID_num"].duplicated(keep=False) & gdf["OBJECTID_num"].notna()
    gdf = add_temporal_flags(gdf, code)

    gdf["flag_geometry_missing_original"] = gdf.geometry.isna() | gdf.geometry.is_empty
    gdf["flag_geometry_invalid_original"] = ~gdf.geometry.is_valid.fillna(False)

    # Reproject coordinates before area QA. EPSG:9473 is equal-area in metres.
    gdf = gdf.to_crs(TARGET_CRS)
    gdf["area_original_m2"] = gdf.geometry.area.where(~gdf["flag_geometry_missing_original"], pd.NA)

    repaired = gdf.geometry.make_valid()
    repaired = repaired.map(polygonal_only)
    gdf = gdf.set_geometry(repaired)
    gdf["flag_geometry_empty_after_repair"] = gdf.geometry.isna() | gdf.geometry.is_empty
    gdf["flag_geometry_invalid_after_repair"] = ~gdf.geometry.is_valid.fillna(False)
    gdf["area_repaired_m2"] = gdf.geometry.area.where(~gdf["flag_geometry_empty_after_repair"], pd.NA)
    denom = pd.to_numeric(gdf["area_original_m2"], errors="coerce").abs().replace(0, pd.NA)
    gdf["repair_area_change_fraction"] = (
        (pd.to_numeric(gdf["area_repaired_m2"], errors="coerce") - pd.to_numeric(gdf["area_original_m2"], errors="coerce")).abs() / denom
    )
    gdf["flag_repair_area_change_gt_1pct"] = gdf["repair_area_change_fraction"] > 0.01
    gdf["flag_repair_area_change_gt_5pct"] = gdf["repair_area_change_fraction"] > 0.05

    hashes = gdf.geometry.map(canonical_geometry_hash)
    gdf["geometry_hash_sha256"] = hashes.map(lambda x: hashlib.sha256(x.encode("ascii")).hexdigest() if x else "")
    valid_hash = gdf["geometry_hash_sha256"] != ""
    gdf["flag_exact_geometry_duplicate"] = valid_hash & gdf["geometry_hash_sha256"].duplicated(keep=False)
    gdf["geometry_duplicate_group"] = ""
    if gdf["flag_exact_geometry_duplicate"].any():
        codes, _ = pd.factorize(gdf.loc[gdf["flag_exact_geometry_duplicate"], "geometry_hash_sha256"], sort=True)
        gdf.loc[gdf["flag_exact_geometry_duplicate"], "geometry_duplicate_group"] = [f"DUP_{i+1:04d}" for i in codes]

    # Hard cleaning exclusions are deliberately narrow. Bad event-year attributes are
    # flagged for later treatment construction but do not erase valid snapshot geometry.
    gdf["hard_exclusion"] = (
        gdf["flag_placeholder_core"]
        | gdf["flag_report_year_invalid"]
        | gdf["flag_objectid_invalid"]
        | gdf["flag_objectid_duplicate"]
        | gdf["flag_geometry_missing_original"]
        | gdf["flag_geometry_empty_after_repair"]
        | gdf["flag_geometry_invalid_after_repair"]
    )
    gdf["analysis_geometry_eligible"] = ~gdf["hard_exclusion"]

    eligible = gdf[gdf["analysis_geometry_eligible"]].copy()
    if eligible.empty:
        raise AuditError(f"No eligible {code} geometry remains after cleaning")

    # GeoPandas dissolve unions features within each mine/layer/report-year and
    # therefore prevents overlapping records from being double counted.
    eligible["feature_count"] = 1
    eligible["feature_area_m2"] = pd.to_numeric(eligible["area_repaired_m2"], errors="coerce")
    group_cols = ["MineID_norm", "MineN", "source_layer", "ReportYr_num"]
    dissolved = eligible.dissolve(
        by=group_cols,
        aggfunc={"feature_count": "sum", "feature_area_m2": "sum"},
        as_index=False,
        method="unary",
    )
    dissolved["dissolved_area_m2"] = dissolved.geometry.area
    dissolved["overlap_removed_m2"] = (dissolved["feature_area_m2"] - dissolved["dissolved_area_m2"]).clip(lower=0)
    dissolved["overlap_fraction_of_sum"] = dissolved["overlap_removed_m2"] / dissolved["feature_area_m2"].replace(0, pd.NA)
    dissolved["dissolved_area_ha"] = dissolved["dissolved_area_m2"] / 10000.0

    qa_columns = [
        "source_layer", "OBJECTID", "OBJECTID_num", "MineID", "MineID_norm", "MineN", "ReportYr", "ReportYr_num",
        "flag_placeholder_core", "flag_report_year_invalid", "flag_objectid_invalid", "flag_objectid_duplicate",
        "flag_event_year_invalid", "flag_event_year_future", "flag_chart_date_invalid", "flag_submit_date_invalid",
        "event_date_usable", "flag_geometry_missing_original", "flag_geometry_invalid_original",
        "flag_geometry_empty_after_repair", "flag_geometry_invalid_after_repair",
        "area_original_m2", "area_repaired_m2", "repair_area_change_fraction",
        "flag_repair_area_change_gt_1pct", "flag_repair_area_change_gt_5pct",
        "flag_exact_geometry_duplicate", "geometry_duplicate_group", "geometry_hash_sha256",
        "hard_exclusion", "analysis_geometry_eligible",
    ]
    qa = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))[qa_columns]
    return gdf, dissolved, qa, source


def write_gpkg(path: Path, records: gpd.GeoDataFrame, dissolved: gpd.GeoDataFrame) -> None:
    if path.exists():
        path.unlink()
    records.to_file(path, layer="records_flagged", driver="GPKG", engine="pyogrio")
    dissolved.to_file(path, layer="dissolved_by_mine_year", driver="GPKG", engine="pyogrio", mode="a")


def run_clean(root: Path, out: Path, selection: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise AuditError(f"Output directory is not empty: {out}. Use --overwrite to replace generated outputs.")
    out.mkdir(parents=True, exist_ok=True)
    selected = read_selected_ids(selection)
    manifest = {
        "script_version": VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root.resolve()),
        "output_root": str(out.resolve()),
        "target_crs": TARGET_CRS,
        "selected_mine_ids": selected,
        "raw_inputs_modified": False,
        "layers": {},
    }
    all_qa = []
    all_groups = []
    for code in ["L03", "L04"]:
        records, dissolved, qa, source = clean_layer(root, code, selected)
        gpkg = out / f"{code}_cleaned.gpkg"
        write_gpkg(gpkg, records, dissolved)
        qa.to_csv(out / f"{code}_record_qa.csv", index=False, encoding="utf-8-sig")
        group_table = pd.DataFrame(dissolved.drop(columns=dissolved.geometry.name))
        group_table.to_csv(out / f"{code}_group_area_overlap_qa.csv", index=False, encoding="utf-8-sig")
        all_qa.append(qa)
        all_groups.append(group_table)
        manifest["layers"][code] = {
            "input_vector": str(source),
            "input_sha256": sha256_file(source),
            "selected_records": int(len(records)),
            "hard_exclusions": int(records["hard_exclusion"].sum()),
            "eligible_records": int(records["analysis_geometry_eligible"].sum()),
            "dissolved_groups": int(len(dissolved)),
            "output_gpkg": gpkg.name,
        }
    pd.concat(all_qa, ignore_index=True).to_csv(out / "all_record_qa.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_groups, ignore_index=True).to_csv(out / "all_group_area_overlap_qa.csv", index=False, encoding="utf-8-sig")
    manifest["selection_sha256"] = sha256_file(selection)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = []
    for code, item in manifest["layers"].items():
        summary.append({"layer": code, **{k: v for k, v in item.items() if k not in {"input_vector", "input_sha256", "output_gpkg"}}})
    pd.DataFrame(summary).to_csv(out / "cleaning_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Completed. Outputs written to {out}")
    print("Raw export files were read only and were not modified.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help=r"Verified export root, e.g. D:\G3_NSW_FRESH")
    p.add_argument("--output", type=Path, required=True, help=r"New output folder, e.g. D:\G3_STAGE1_MIN_AUDIT")
    p.add_argument("--selection", type=Path, help="CSV created by --prepare-selection")
    p.add_argument("--prepare-selection", action="store_true", help="Create candidate_mines.csv and stop")
    p.add_argument("--overwrite", action="store_true", help="Replace generated output files")
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    out = args.output.expanduser().resolve()
    try:
        if args.prepare_selection:
            prepare_selection(root, out / "selected_mines.csv")
            return 0
        if args.selection is None:
            raise AuditError("--selection is required unless --prepare-selection is used")
        run_clean(root, out, args.selection.expanduser().resolve(), args.overwrite)
        return 0
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
