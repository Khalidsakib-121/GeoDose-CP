# GeoDose Stage 3D D4 — M6 New-Data + Nuisance Integration v1.1.0

**v1.1.0 is the exact-source-locked D4 rebuild.** It was finalized only after a fresh complete D2 source package was checked 41/41 against the accepted D2 source inventory. The adapter now locks the exact D2 v1.2 signatures, frozen tolerances, and tie-randomization contract instead of inferring constructor/API semantics. The accepted D3 exact-integration dependency remains `v1_0_1` (SHA-256 `f20320ec72ae115709bc809d72a7c8f37f75f12a335c271a895c521900d1a51d`).

## Purpose

This is a **pre-pilot software/scientific integration gate**. It makes the already accepted exact GeoDose-CP G1→G2→G3 implementation callable on **new, small exact Stage 3B validation cases**, and crosses treatment-law and graph-law nuisance modes without rewriting the accepted exact law.

It does **not** run the preregistered 20-replication pilot, does not implement N2/N3, and does not create publication-performance evidence.

## Frozen scientific identities

- **M3:** standardized-residual graph law only; treatment A remains fixed; no q_h/g; no target-design ratio.
- **M4:** exactly one normalization of `d_j * s_j(y)` where `d_j=q_h/g` and `s_j(y)` is the accepted M3 target-source residual marginal.
- **M6:** accepted D2 source-level CandidateEvaluator using the exact G1→G2→G3 implementation. D4 does not reproduce that algebra in new code.
- **H1/H4:** remain heuristic ablations only.
- **Target-design ratio:** excluded from D4 M3/M4/M6; it belongs to the scalable N2/N3 branch.

## Nuisance variants

D4 exercises:

- `OT_TG`: oracle Stage 3B mixed treatment law + true Stage 3B graph.
- `ET_TG`: frozen Stage 3C MixedPropensityModel fitted on nuisance-training units only + true graph.
- `OT_FG`: oracle treatment law + accepted Stage 3B fitted graph.
- `ET_FG`: estimated treatment law + fitted graph.

For the fitted-graph diagnostic, graph degrees are recomputed from the fitted edge set. The registered oracle spatial rho and residual scale are retained deliberately, so D4 isolates **graph-structure misspecification**. This is not a fully estimated spatial-law claim.

The oracle Stage 3B outcome conditional mean/scale are retained throughout D4 to isolate treatment/graph wiring. Estimated-outcome exact-coverage claims are explicitly prohibited here.

## New exact cases

The fixed smoke grid is `candidate_y ∈ {-1, 0, 1}` and is software-diagnostic only.

- `S2_SPATIAL`, target dose 0.50
- `S3_SHIFT`, target dose 0.90
- `S4_RHO040`, target dose 0.90
- `S6_ROOK`, target dose 0.75
- `S6_OMIT50`, target dose 0.75, with a deterministic six-node block that is connected under the accepted fitted graph
- `S7_EXCHANGEABLE`, target dose 0.50
- `ST1_MIXED_ATOMS`, target dose 0
- `ST1_MIXED_ATOMS`, target dose 1

For Stage 3B `iid_continuous` residual cases at `rho=0`, the accepted D2 Gaussian-GMRF specialization is used with `rho=0` as the computationally equivalent iid-Gaussian representation. This representation change is recorded in the M6 preparation audit.

## Mandatory fail-closed sequence

Before any new case is evaluated, the package:

1. verifies the eight accepted upstream output ZIP hashes;
2. verifies the complete local frozen Stage 3C source tree;
3. verifies the complete local frozen D2 source tree;
4. verifies the vendored M3/M4 source bytes against the accepted M4 source inventory;
5. imports `geodose_stage3d` only from the verified D2 source root;
6. replays accepted D2 `G6_GAUSS_INTERIOR` candidate evaluations at `y=-8,0,8` through the **source-level CandidateEvaluator**;
7. additionally replays the accepted duplicate-payload fixture `G6_GAUSS_DUPLICATE` at `y=0` and confirms the accepted 360-state quotient identity;
8. refuses to evaluate new data if those replays differ from accepted D2 by more than `2e-12` or change acceptance;
9. verifies two expected fail-closed paths before the main smoke matrix: an unaudited endpoint (`ST1_INTERIOR_ONLY`) and a deliberately disconnected exact block.

Only after these gates pass are the supported new Stage 3B cases evaluated.

### Exact D2 API lock

`configs/d2_api_lock.json` freezes every accepted D2 callable used by D4, including the keyword-only precision threshold for `prepare_orbit`, the required `non_gaussian_min_abs_residual` constructor argument, and the accepted D2 tie-randomization contract. Accepted replay checks conservative and randomized p-values, tie uniform, acceptance decisions, and quotient-state counts. See `docs/D4_Exact_D2_API_Research.md`.

### Frozen target slot and measurement-error boundary

Stage3B contains many units with role `test_target`. D4 exact size-6 inference freezes the target slot specifically to **node 320**; it never selects a target by role alone. Before exact computation, D4 also verifies that observed and true outcomes coincide for every registered D4 case/dose. This deliberately excludes S9-style measurement-error cases from the D4 exact-wiring gate.

## Important reduction rule

D4 does **not** force numerical claims such as “M6 equals M2 whenever rho=0” or “M6 equals M3 whenever the scenario label says no target shift.” Those equalities require the sufficient factorization and score-compatibility conditions already formalized and tested in D3. Heterogeneous slot covariates/outcome transforms can prevent naive equality even under a simple scenario label.

The accepted D3 reduction status is preserved:

- R1: PASS
- R2: PASS
- R3: **DEFERRED_STAGE3E**
- R4: PASS

## Run on Windows

Recommended extraction path:

`D:\GeoDose_Stage3D_D4_M6_NewData_Nuisance_Integration_v1_1_0`

Then double-click:

`RUN_STAGE3D_D4.bat`

**v1.1.0 launcher hardening:** if the BAT is accidentally opened directly from the Windows ZIP/compressed-folder view, the launcher detects the temporary `AppData\Local\Temp` / `.zip.` path, copies the complete package to `D:\GeoDose_Stage3D_D4_M6_NewData_Nuisance_Integration_v1_1_0`, and relaunches from there before creating `.venv`. If Windows has not materialized the complete package, it stops with an explicit `Extract All` instruction rather than attempting a scientific run. An incomplete local `.venv` from an interrupted run is also rebuilt automatically.

The package creates its own Python 3.10 virtual environment and verifies the pinned Stage 3C-compatible runtime.

If a frozen path differs on your laptop, edit `configs\local_paths_windows.json`. Do not move or modify a frozen upstream project merely to satisfy this package.

## Expected successful output

`GeoDose_Stage3D_D4_M6_NEWDATA_NUISANCE_OUTPUTS.zip`

Return that ZIP for independent review. **Do not start Stage 3E or the 20-replication pilot until the Windows-generated D4 output has been reviewed.**

## Claim boundary

Passing D4 means only that the exact source-level M6 engine has been connected successfully to new small exact cases and that the treatment/graph nuisance plumbing behaves consistently with the frozen M3/M4/M6 definitions.

It does not establish repeated coverage, scalable coverage, estimated-nuisance exact validity, pilot-scale readiness, MineDoseBench performance, NSW performance, or publication superiority.



## v1.0.2 D2 API compatibility fix retained

The byte-verified accepted D2 v1.2 `prepare_orbit` function requires the keyword-only argument `minimum_precision_eigenvalue_required`. D4 retains the v1.0.2 correction and binds that argument from the accepted D2 frozen tolerance registry key `minimum_precision_eigenvalue`; it does not hard-code a newly chosen scientific threshold. The accepted value is 1.0e-10. The adapter validates the exact source signature and fails closed if the contract differs. No scientific method, case, tolerance value, nuisance rule, or claim boundary changed.


## v1.0.3 launcher robustness correction

A Windows Explorer compressed-folder run can execute the BAT from an `AppData\Local\Temp\...zip.<suffix>` path. Earlier launchers tried to create `.venv` there, which can make `ensurepip` fail before D4 starts. v1.0.3 introduced detection of that condition before touching Python; v1.1.0 retains and updates the destination for the current release, self-relocating to the fixed D:\ path above, verifies that the copied package is complete, and relaunches. This is an execution-environment correction only; no scientific code or frozen method contract changed.

## v1.1.0 consolidated preflight hardening

Before accepted-D2 replay, the adapter now performs a fail-closed signature preflight over every D2 callable it will use. This does not alter D2; it prevents serial discovery of API mismatches. `acceptance(result, mode)` must execute successfully and return a boolean; there is no silent fallback. All D4 runtime/provenance version fields are aligned to `1.1.0-stage3d-d4`.

**Fail-fast input preflight:** before creating `.venv` or installing packages, `preflight_windows.py` uses only Python 3.10 standard-library modules to resolve all frozen inputs, validate their eight accepted SHA-256 hashes and ZIP CRCs, verify the Stage 3C (18/18) and D2 (41/41) local source trees, and verify the vendored accepted M3/M4 bytes. The full D4 runner repeats these checks before scientific computation.
