# GeoDose-CP Stage 2A Block Builder

This package creates the frozen spatial support required before any DEA outcome is added.

## Input

It reads only:

- `D:\G3_STAGE1_MIN_AUDIT\L03_cleaned.gpkg`
- `D:\G3_STAGE1_MIN_AUDIT\L04_cleaned.gpkg`
- `D:\G3_STAGE1_MIN_AUDIT\run_manifest.json`

The Stage 1 files are never modified.

## Run

1. Extract this package outside the Stage 1 folder.
2. Double-click `RUN_STAGE2A.bat`.
3. Wait for `STAGE 2A VERIFIED COMPLETE`.

The output is recreated at:

`D:\G3_STAGE2_BLOCKS`

## Frozen construction

- CRS: EPSG:9473.
- Primary support: globally anchored 90 m × 90 m cells.
- Retention: at least 70% of each cell lies inside the union of cleaned L03 and L04.
- Exposure name: **mapped rehabilitation implementation fraction**.
- Exposure:
  `Area(block ∩ rehabilitation) / Area(block ∩ union(rehabilitation, disturbance))`.
- Interpretation: noncausal snapshot exposure.
- Graph: within-mine eight-neighbour queen contiguity.
- Splits: five leave-one-mine-out folds.
- The graph is frozen before DEA extraction.

## Required outputs

- `mine_blocks_90m.gpkg`
  - `blocks_90m`
  - `mine_footprints`
- `block_attributes.parquet`
- `block_graph_edges.csv`
- `leave_one_mine_out_splits.csv`
- `stage2_block_summary.csv`
- `stage2_manifest.json`
- `STAGE2A_VERIFICATION.json`

The package also writes `stage2_config.json`.

## Important interpretation

The output is not an annual rehabilitation treatment. It is a frozen spatial snapshot support and noncausal mapped exposure.

For later DEA extraction, use the full 90 m block as the frozen graph support but mask observations to the corresponding `mine_footprints` geometry. The fields `mine_footprint_area_m2` and `footprint_coverage` record the usable within-footprint area.

## Expected result for the current five-mine Stage 1 files

The validated reference run produced:

- 5 mines
- 27,042 retained blocks
- 102,893 undirected queen edges
- 5 leave-one-mine-out folds

Small disconnected mine-footprint islands may create a few degree-zero nodes. They are preserved and reported rather than silently deleted.
