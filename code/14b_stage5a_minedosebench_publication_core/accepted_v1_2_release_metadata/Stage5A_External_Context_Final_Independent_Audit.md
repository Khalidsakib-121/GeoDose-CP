# GeoDose-CP Stage5A External Context — Independent Audit

**Audited artifact:** `GeoDose_Stage5A_EXTERNAL_CONTEXT_FINAL_OUTPUTS.zip`  
**Independent audit date:** 2026-08-10  
**Audited ZIP SHA-256:** `06c5255351f7a28b708a8348c3c02ee9ff2a3f31388ec20c38076ce440e69ed7`

## Decision

**PASS — ACCEPT EXTERNAL-CONTEXT PRODUCTION AS FROZEN INPUT TO A REVISED STAGE5A MINE DOSE BENCH FREEZE.**

This acceptance applies to the external-context product only. It does not authorize the unchanged MineDoseBench v1.1.0 freeze, which still expects 23,710 context rows. The next stage must prospectively consume the frozen 23,618-block context-complete substrate and preserve the 92 support exclusions.

## Independent integrity audit

- Final ZIP contains 26 members, including a 25-file immutable output manifest.
- All 25 manifest-listed files independently recomputed to the declared SHA-256 and byte size.
- Output-manifest aggregate independently recomputed exactly as `df953f9b07bfe4ab53f56f38273bf97f00d63afbac4942ea406d8737048dedcd`.
- Embedded source snapshot SHA-256 agrees with the production report: `93468aad45ca145fd3a80ee1758685f8877a46a79d6156a4d77e51198e4e3e52`.
- All 35 source-snapshot files independently match their snapshot-manifest SHA-256 and byte size.
- All 35 snapshot source hashes reconcile exactly with `SOURCE_PROVENANCE_MANIFEST.json`.
- Handoff copies in `READY_FOR_STAGE5A_REVISED_FREEZE/` are byte-identical to their top-level authorities.

## Frozen substrate audit

Original Stage5A PREP required context for 23,710 blocks. The final output partitions these exactly into:

- **23,618 retained context-complete blocks**
- **92 prospectively excluded support-gap blocks**
- retained/excluded sets are unique and disjoint
- union equals the exact original 23,710 PREP-required block IDs
- mine IDs and mine names match the original PREP request exactly
- support threshold remains exactly **0.99**

Retention fraction = **0.9961197807 (99.611978%)**.  
Exclusion fraction = **0.0038802193 (0.388022%)**.

### Attrition by mine

| Mine | Original | Retained | Excluded | Retention |
|---|---:|---:|---:|---:|
| Hunter Valley Operations | 7,873 | 7,873 | 0 | 100.000% |
| Mt Arthur Coal | 6,404 | 6,404 | 0 | 100.000% |
| Mount Thorley Warkworth Complex | 4,273 | 4,273 | 0 | 100.000% |
| Bulga Complex | 3,433 | 3,430 | 3 | 99.9126% |
| Liddell Coal | 1,727 | 1,638 | 89 | 94.8466% |
| **Total** | **23,710** | **23,618** | **92** | **99.6120%** |

The failure pattern matches the accepted support census exactly. No excluded block is recovered by imputation or threshold relaxation.

## Context-table audit

`mine_context_covariates.csv` contains exactly **23,618 rows × 14 columns**: one `block_id` plus the 13 frozen numeric covariates. All block IDs are unique. There are **no NaN, Inf, or missing values**.

All 13 covariates are inside the frozen physical ranges. Observed ranges are:

- rainfall 2023: 421.500–562.400 mm
- rainfall 2024: 448.300–734.200 mm
- short-crop ET 2023: 1370.205–1435.300 mm
- short-crop ET 2024: 1271.660–1314.000 mm
- SOC 0–15 cm: 0.9862–2.8111 %
- pH(CaCl2) 0–15 cm: 4.3161–6.2501
- clay 0–15 cm: 14.8200–46.9225 %
- bulk density 0–15 cm: 1.2380–1.4281 g/cm3
- AWC 0–100 cm: 96.8647–139.2967 mm
- terrain elevation: -28.5850–351.4508 m
- terrain slope: 0.3994–21.1395 degrees
- terrain aspect: 0.0074–359.9971 degrees
- terrain roughness: 0.2576–9.5078 m (3x3 population SD definition)

## Formula audit

The four 0–15 cm properties were independently recomputed from the depth-component audit using `(5*x_0_5 + 10*x_5_15)/15`.

Maximum independent absolute recomputation residuals were numerical-roundoff scale only:

- SOC: 1.78e-15
- clay: 2.84e-14
- bulk density: 8.88e-16
- pH: 2.66e-15

AWC 0–100 cm was independently recomputed as the sum of volumetric-percent layer values converted to water depth over 50, 100, 150, 300, and 400 mm layer thicknesses. Maximum absolute residual: **5.68e-14 mm**.

## Support reconciliation audit

For the 23,618 retained blocks:

- all 18 governing support fractions are >= 0.99
- minimum retained support fraction = **0.993802177913525** (bulk-density layers)
- final production valid-area fractions match the independently completed support census exactly for all 18 governing layers in the direct independent comparison
- SILO climate support and GA terrain-common support are complete for all retained blocks

## Raster/source snapshot audit

Embedded raster metadata is internally coherent with the frozen design:

- SILO AOI annual rasters: EPSG:4326, 0.05-degree grid
- TERN SLGA crops: EPSG:4326, 3 arc-second (~90 m) grid
- GA raw DEM AOI: EPSG:4326, approximately 1 arc-second grid
- derived terrain: EPSG:9473, 30 m grid

No credential value was found in the user-facing textual outputs. The provenance records authentication type only, not the API key.

## Provenance note: TERN BDW unit metadata discrepancy

The embedded TERN point-of-truth JSON for the two BDW assets reports `Units: Percent`, while the authoritative TERN collection-level Release-2 metadata defines Bulk Density - Whole Earth in **g/cm3**. The production `unit_contract` uses g/cm3, and the extracted values (~1.24–1.43) are consistent with that collection-level unit. This is therefore treated as a **source-metadata documentation inconsistency, not a numerical transformation error**. For the paper and downstream Stage5A documentation, cite the TERN collection-level bulk-density authority (Malone, 2023; DOI `10.25919/gxyn-pd07`) and explicitly record this provenance discrepancy.

TERN Release-2 AWC is officially Available Volumetric Water Capacity in percent, supporting the frozen percent-to-mm depth integration. Collection authority: Searle, Somarathna & Malone (2023), DOI `10.25919/4jwj-na34`.

## Reproducibility preservation note

The embedded 26 MB source snapshot contains the exact AOI source/derived rasters used by final production, but it does not include the four original ~418–419 MB SILO annual NetCDF source files. Their hashes are recorded in provenance. Preserve the full existing `CONTEXT_SOURCE_CACHE` separately for maximum auditability and regeneration of the SILO annual AOI sums.

## Final supervisor recommendation

1. **Freeze the 23,618-block external-context product.**
2. **Do not recover or impute the 92 support-gap blocks.**
3. **Do not alter the 0.99 support threshold.**
4. Preserve this final ZIP, the accepted support-census ZIP, the production source package, PowerShell logs, and the full source cache.
5. Next create a prospectively revised Stage5A MineDoseBench freeze package that explicitly accepts the 23,618-block substrate and carries the 92-block exclusion registry forward.
6. Do not run the unchanged MineDoseBench v1.1.0 FREEZE.

**Audit outcome: PASS / ACCEPT FOR REVISED STAGE5A FREEZE INPUT.**
