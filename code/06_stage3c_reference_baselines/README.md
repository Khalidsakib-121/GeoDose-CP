# GeoDose-CP Stage 3C Reference Infrastructure v1.1 FREEZE

This package is the corrected Gate C1 implementation for the shared nuisance engine, weighted-conformal infrastructure, and reference baselines.

## Scientific boundary

Implemented principal baselines:

- M1 standard split conformal.
- M2 dose-only weighted conformal.
- M5 conservative graph-safe weighted conformal, with an explicit sampling-eligibility flag.

Implemented preregistered heuristic ablations:

- H1 geographic Euclidean-distance heuristic (Stage 3A ABL7).
- H4 naive product of M2 treatment weights and H1 heuristic weights.

Not claimed in this package:

- M3 spatial-only generalized conformal. The frozen M3 requires a graph-local residual-law generalized-conformal construction and is deferred to Stage 3D.
- Final M4. It is deferred until the final M3 spatial-law weights exist.
- M6, G2/G3, N2/N3, MineDoseBench, production coverage, or pilot-derived thresholds.

This boundary is deliberate: the prior v1.0 incorrectly called a geographic-distance heuristic “M3.”

## Corrected alignment features

- Exact Stage 3A and Stage 3B hashes.
- Stage 3A frozen nuisance seeds and labelled model substreams.
- Four factorial nuisance variants: oracle/estimated treatment crossed with true/fitted graph.
- Content-addressed fit IDs based on actual training data, hyperparameters, package version, and seed.
- Temporal ST2 fixture audit with unit-level role preservation and no row-wise split.
- Paired S10 90 m to 180 m aggregation audit.
- Observational-law metrics grouped as one target category rather than one category per realized draw.
- Theorem-eligibility, false-support, computational-failure, Moran’s I, and semivariogram reporting.
- Separate treatment transport and target-design transport.
- Independent calibration-weight audit used by the verifier.
- Full package-tree provenance, exact environment verification, safe overwrite, deterministic compression; the independent release audit additionally confirms byte-identical scientific outputs across clean generations (runtime/environment bookkeeping excluded).

## Run

Double-click `RUN_STAGE3C.bat`.

The runner requires Python 3.10 and creates/recreates only the package-local `.venv` when exact versions do not match `requirements_py310.txt`.

Accept only when the final terminal output includes:

```text
STAGE 3C REFERENCE INFRASTRUCTURE VERIFIED COMPLETE
Verification checks: 47/47
Implemented principal baselines: M1, M2, M5
Implemented heuristic ablations: H1, H4
M3 implemented: no
M4 implemented: no
M6 implemented: no
Production experiments run: no
```

Upload `GeoDose_Stage3C_REFERENCE_OUTPUTS.zip` for independent output audit before freezing Gate C1.
