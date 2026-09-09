# GeoDose-CP Stage 2B — Local DEA STAC solution

This package performs the same Stage 2B pre-screen without a DEA Sandbox account.
It searches the public DEA Explorer STAC API and streams public Cloud-Optimised GeoTIFF data from the `dea-public-data` AWS S3 bucket using unsigned access.

## Requirements

- Windows 10/11, 64-bit
- Python 3.11, 64-bit
- Stable internet connection
- Recommended: at least 16 GB RAM and 10 GB free disk space
- PyCharm is optional; the included BAT file is simpler

No AWS account, AWS key, DEA Sandbox account, or local Open Data Cube database is required.

## Frozen inputs

The verified Stage 2A inputs are already included under `inputs/`. Do not replace or edit them.

## First run

1. Extract the ZIP to a short local path, e.g. `D:\GeoDose_Stage2B_Local_STAC_v1_0`.
2. Install 64-bit Python 3.11 if `py -3.11 --version` does not work.
3. Double-click `TEST_PUBLIC_DEA_ACCESS.bat`.
4. Only after it reports `PUBLIC DEA STAC CONNECTION TEST PASSED`, double-click `RUN_STAGE2B_LOCAL_STAC.bat`.

The first dependency installation can take several minutes. The full extraction may take hours depending on internet speed and DEA/AWS response times.

## PyCharm run

Open the extracted folder as a project and use a Python 3.11 virtual environment.
Install `requirements.txt`, then create a run configuration for:

- Script: `stage2b_local_stac_prescreen.py`
- Parameters: `--input inputs --output outputs`
- Working directory: the extracted package folder

After the extraction completes, run `verify_stage2b_local_stac.py` with:

`--input inputs --output outputs`

## Resume

Each mine-year is saved as a checksum-verified checkpoint. Rerun the same command after interruption; completed jobs are reused.
Do not use `--restart` unless you deliberately want to delete all progress.

## Data and methods

- STAC API: `https://explorer.dea.ga.gov.au/stac`
- Products: `ga_ls_fc_3` (`pv`, `ue`) and `ga_ls_wo_3` (`water`)
- Years: 2023, 2024, 2025
- Output grid: EPSG:9473, 30 m, edge anchored
- Grouping: solar day
- Resampling: nearest neighbour
- WOfS overlapping-scene fusion: official DEA bit-field fusion logic
- Analysis support: frozen 90 m blocks masked to mine footprints
- Mine selection remains outcome-blind; PV values are not used to choose mines

## Successful completion

The verifier must print:

`STAGE 2B VERIFIED COMPLETE`

and `outputs/STAGE2B_VERIFICATION.json` must contain `"status": "verified_complete"`. The BAT file then creates `GeoDose_Stage2B_OUTPUTS_LOCAL_STAC.zip`.

## Network notes

The workflow streams many small byte ranges from public AWS COG files. Corporate firewalls, antivirus HTTPS inspection, VPNs, or unstable internet can slow or block access. Run the connection test first. Checkpoints make ordinary interruptions recoverable.

## Scientific scope

This is the same product-aware EO availability and quality pre-screen as the Sandbox design. It does not transform the mapped rehabilitation implementation fraction into an annual causal treatment.


## v1.2 STAC search hardening

The DEA `/search` request is explicitly forced to `GET` with a WGS84 bounding box, matching DEA's public STAC examples. The client no longer relies on PySTAC Client's default `POST` search with an `intersects` body. Five bounded retry attempts and clearer failure diagnostics are included for transient DEA API errors.


## v1.2 schema correction

`graph_component_id` is a frozen identifier such as `85D8BA5D8512_CC001`; it is preserved as text in the block-year output. This is a schema correction only.
