# GeoDose-CP Stage 3D D3 — M1–M6 Exact Integration v1.0.1

This is a **pre-pilot exact-reference integration gate**, not a production experiment.

It preserves the accepted method identities:

- M1, M2, M5: exact frozen Stage 3C implementation bytes.
- M3: accepted D3 residual-only graph/residual-law comparator.
- M4: accepted D3 one-normalization `d_j * s_j(y)` naive comparator.
- M6: accepted D1/D2 G1->G2->G3 exact reference exposed through a read-only adapter over D2-evaluated candidates.
- H1/H4 remain heuristic ablations and are never promoted to M3/M4.

## Why M6 is output-backed in this gate

D2 is frozen and must not be reimplemented merely for integration. This package therefore exposes the already accepted D2 exact candidate evaluations through the common schema and separately verifies the user's local D2 source tree byte-for-byte against the source-hash inventory embedded in accepted D2 output. It also exports an AST API inventory of that verified D2 source for the next **source-level new-data M6 adapter** stage.

Therefore this package does **not** claim that M6 is ready to run new pilot replications yet. That is the next gate after this exact-reference integration passes.

## What this gate runs

1. Exact SHA-256 + ZIP integrity verification for Stage 3A/B/C, D1, D2, accepted M3 and accepted M4 outputs.
2. Full source-tree verification for local frozen Stage 3C and D2 source roots.
3. Exact-byte audit of vendored Stage 3C and accepted M3/M4 engine modules.
4. Replay of all 540 frozen Stage 3C calibration-audit groups, including M1/M2/M5 and H1/H4, without changing their identities.
5. Regression of accepted M3 and M4 structural gates.
6. Regression of accepted D2 p-value/inclusion/tie/subset rules.
7. A common six-method candidate trace on the complete accepted D2 `G6_GAUSS_INTERIOR / S4_RHO060` candidate grid (>=2000 candidates).
8. R1, R2, and R4 in-scope reduction checks. **R3 is explicitly deferred to Stage 3E because it is an N2 criterion reduction.**
9. Nuisance-variant readiness matrix; no estimated-M6 validity claim.
10. D2 source API inventory for the next source-level new-data adapter.

## Windows use

Extract to:

`D:\GeoDose_Stage3D_D3_M1_M6_Exact_Integration_v1_0_1`

Then run:

`RUN_STAGE3D_D3_INTEGRATION.bat`

A successful run creates:

`GeoDose_Stage3D_D3_M1_M6_EXACT_INTEGRATION_OUTPUTS.zip`

Send that ZIP back for independent review. Do **not** start the 20-rep pilot yet.


## v1.0.1 Windows path-resolution fix

The M3 and M4 source packages themselves contain one top-level project directory. If Windows Explorer extracts them into an outer folder of the same name, the generated accepted output ZIPs are normally located one directory deeper, e.g.

`D:\GeoDose_Stage3D_D3_M3_Oracle_v1_0\GeoDose_Stage3D_D3_M3_Oracle_v1_0\GeoDose_Stage3D_D3_M3_ORACLE_OUTPUTS.zip`

and similarly for M4. v1.0.1 checks both the double-nested and single-nested layouts. This is a path-discovery correction only; the scientific integration code is unchanged.
