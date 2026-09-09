# GeoDose Stage 3F — Preregistered S1–S10 20-Replication Pilot + Threshold Freeze

This package is the **pilot/tuning gate**, not a production experiment and not publication-performance evidence.

It reuses the frozen Stage3B generator, accepted Stage3C baselines, accepted D2 G3 engine, accepted D4 M3/M4 integration source, and accepted Stage3E N2/N3 scalable certification source. It does not modify any accepted upstream stage.

## Scientific purpose

1. Run the frozen 20-replication pilot seeds for S1–S10.
2. Evaluate M1–M6 on the registered pilot case matrix.
3. Keep `ORACLE_STRUCTURAL_OT_TG` as the theorem-certified scalable M6 route.
4. Treat RF/XGB and estimated/misspecified propensity routes as diagnostics only.
5. Select operational thresholds only from the three candidate grids frozen in accepted Stage3C:
   - minimum ESS: 3, 5, 10, 15, 20
   - maximum normalized weight: 0.10, 0.15, 0.20, 0.30, 0.50
   - minimum graph-safe count: 5, 10, 15, 20
6. Use controlled pilot coverage/refusal only. Width and NSW are excluded from threshold selection.
7. Freeze the selected thresholds before any production outcome is generated.

Threshold selection is deterministic and lexicographic over the 100 accepted Stage3C tuples. The candidate grids and controlled-pilot coverage/refusal principle were frozen upstream; the exact Stage3F lexicographic software implementation is frozen here before production outcomes and is not overstated as an earlier detailed preregistration. Casewise undercoverage is penalized maximally when fewer than 10 safe-case pilot targets are returned, preventing an all-refuse/tiny-selective-set solution from winning. No post-hoc hard retention cutoff is introduced.
Pilot-scale readiness additionally requires every registered safe case to contribute at least those 10 returned targets under the selected tuple. This is an evidence-adequacy gate, not a fourth tunable threshold. The independent production run remains the first confirmation of frozen-threshold coverage.

## N3 non-vacuity

No additional coverage-lower-bound threshold grid is invented in Stage3F. N3 non-vacuity is a fixed structural rule: the relevant certified/operational lower bound must be finite and strictly greater than zero. This is not tuned using pilot outcomes.

## Frozen target population

A target is drawn uniformly from structurally theorem-eligible final-test slots. Eligibility is defined before treatment/outcome inspection: the target must admit a deterministic connected size-6 block containing one target slot and five calibration slots under the frozen true graph.

Registered fitted-graph diagnostics, especially S6, may disconnect that same frozen structural block. Stage3F does **not** retarget or change the block. Graph-dependent M3/M4/M6 then fail closed with `R05_GRAPH_COMPONENT_TOO_SMALL`; this is a scientific refusal, not a computational failure.

## Width evidence

For M3/M4/M6, the pilot uses a coarse finite-domain candidate grid only as a runtime/topology diagnostic. It is explicitly **not** a coverage-preserving inversion and is not used for threshold selection or matched-coverage efficiency claims. Full publication-quality efficiency is deferred to the production/ablation stage.

## Pilot reporting boundary

The pilot computes the frozen validity, support, refusal, calibration, dose/local coverage, nuisance, runtime and measurement-sensitivity diagnostics needed for threshold freeze. Full matched-coverage width/WIS and prediction-set topology for M3/M4/M6 are intentionally deferred to the production/ablation stage; Stage3F uses only the explicitly labelled diagnostic width subset and never uses it to select thresholds. Moran's I/semivariogram diagnostics are also deferred to production because they are diagnostic-only in the Stage3A registry.

## Run

Use `RUN_STAGE3F.bat` on Windows. The launcher first checks frozen hashes/source trees before creating the Python environment.

A successful run creates:

`GeoDose_Stage3F_PILOT_S1_S10_20REP_THRESHOLD_FREEZE_OUTPUTS.zip`

Send that ZIP for independent review. Do **not** start production until the Stage3F output is accepted.
