# GeoDose-CP Stage 1 Minimum Cleaner

This package performs only the minimum defensible L03/L04 cleaning required before the treatment-feasibility audit. Version 1.1 prioritizes longitudinal candidates and verifies CSV/GeoJSON OBJECTID alignment.

## Inputs expected

The verified v3 export must remain unchanged at:

`D:\G3_NSW_FRESH`

The script reads:

- `L03_rehabilitation\attributes.csv`
- `L03_rehabilitation\features_geojson4326.geojson`
- `L04_disturbance\attributes.csv`
- `L04_disturbance\features_geojson4326.geojson`

It does not modify any source file.

## Run

1. Extract this package.
2. Double-click `01_PREPARE_SELECTION.bat`.
3. Open `D:\G3_STAGE1_MIN_AUDIT\selected_mines.csv`.
4. First prioritize rows with `has_consecutive_common_pair=TRUE`. Then set `include=1` for exactly five mines and leave the rest as `0`. If fewer than five have an adjacent common-year pair, fill the remaining slots using the highest `balanced_feature_count` rows; those additional mines are for the spatial fallback audit, not longitudinal treatment construction.
5. Double-click `02_RUN_CLEANING.bat`.

## What the script does

- Verifies that the selected `attributes.csv` and `features_geojson4326.geojson` contain exactly the same `OBJECTID` multiset.
- Removes records from analysis when `MineID`, `MineN`, or `ReportYr` is blank.
- Flags invalid/future event-year, chart-date, and submission-date attributes.
- Preserves valid snapshot geometry even when an event-year attribute is bad; such rows are marked `event_date_usable=False` for later treatment construction.
- Repairs invalid geometry with `make_valid`.
- keeps only polygonal components after repair.
- removes empty/unusable geometry from analysis.
- identifies duplicate `OBJECTID` values.
- identifies exact duplicate geometries using normalized WKB hashes.
- reprojects to `EPSG:9473` (GDA2020 / Australian Albers).
- dissolves by `MineID`, layer, and `ReportYr`.
- quantifies overlap removed by the dissolve.
- writes complete QA ledgers and a SHA-256 run manifest.

## Outputs

`D:\G3_STAGE1_MIN_AUDIT`

- `selected_mines.csv`
- `L03_cleaned.gpkg`
  - `records_flagged`
  - `dissolved_by_mine_year`
- `L04_cleaned.gpkg`
  - `records_flagged`
  - `dissolved_by_mine_year`
- `L03_record_qa.csv`
- `L04_record_qa.csv`
- `L03_group_area_overlap_qa.csv`
- `L04_group_area_overlap_qa.csv`
- `all_record_qa.csv`
- `all_group_area_overlap_qa.csv`
- `cleaning_summary.csv`
- `run_manifest.json`

## Interpretation

The dissolved layers are the cleaned mine-report-year snapshot geometries. They are not yet the treatment dose.

Do not use a polygon as an annual treatment event when `event_date_usable=False`. Do not calculate area directly from the raw EPSG:4326 GeoJSON. Use the EPSG:9473 outputs.

## Safety rules

- Never overwrite `D:\G3_NSW_FRESH`.
- Do not manually edit source polygons.
- Keep flagged records in the QA ledger.
- Use dissolved area, not the sum of overlapping polygon areas.
- A future or malformed event-year flag is handled during treatment construction; it is not silently corrected.
