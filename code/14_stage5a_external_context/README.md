SUPERSEDES v1.0.0: v1.0.1 corrects the base-manifest aggregate recomputation contract; scientific design is unchanged.

# GeoDose-CP Stage5A External Context Production v1.0.1 — FREEZE CANDIDATE

This package operationalizes the **prospective substrate decision** made after the independently verified External Context Support Census v1.0.1. It does not search for better support, change the 0.99 rule, impute unavailable soil context, or examine MineDoseBench outcomes/method performance.

## Frozen authorities
- External Context Acquisition v1.0.1 immutable source aggregate: `7a99a75d...b181`.
- Accepted Support Census output ZIP SHA-256: `2facc838...5152e`.
- Accepted Support Census output aggregate: `053e09d0...8177`.
- Original required substrate: 23,710 blocks.
- Frozen context-complete substrate: 23,618 blocks.
- Frozen support exclusions: 92 blocks.
- Frozen source support gate: valid polygon area fraction >=0.99 on all 18 governing layers.

## Source-drift protection
Production performs **no live source selection**. It requires the exact SILO annual AOI rasters, TERN Release-2 AOI crops/metadata, and GA raw/derived terrain rasters whose SHA-256 identities were already frozen in the successful support census. Missing/mismatched bytes cause a hard failure rather than silent redownload.

## Final context
The final table has exactly one row for each of the 23,618 frozen retained block IDs and exactly 13 numeric context covariates. Soil depth harmonization, AWC integration, terrain derivation, polygon area weighting, physical range gates, source hashes and valid-area reconciliation are all audited.

## Important downstream boundary
The unchanged MineDoseBench Generator v1.1.0 still requires all 23,710 original context rows. Therefore this production package intentionally stops with status `FINAL_CONTEXT_PRODUCTION_PASSED_PENDING_STAGE5A_REVISED_FREEZE`. A new prospective MineDoseBench freeze package must accept the 23,618 frozen substrate before any benchmark outcomes or M2-M6 performance are examined.
