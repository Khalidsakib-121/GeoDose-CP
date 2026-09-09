# GeoDose-CP Stage5B MineDoseBench Production Evaluation v1.0.0

This package is the first method-performance stage after the independently accepted/frozen Stage5A MineDoseBench v1.2.0 benchmark.

## One command
On Windows, extract under a short path such as `D:\G5B`, then run:

`RUN_STAGE5B_PRODUCTION.bat`

The run is resumable at one atomic case x replication checkpoint. There are 540 frozen tasks. Re-running the same BAT reuses completed checkpoints.

## Frozen scientific scope
- 27 registered MineDoseBench cases x 20 replications.
- 5 pre-outcome targets per mine/replication/case.
- Primary methods M2–M6 only.
- RF primary; XGBoost confirmation only in frozen S1/S4/S5/S8 scope; oracle nuisance track diagnostic only.
- m=64 scalable sparse/localized graph route; m=32 sensitivity diagnostic.
- Exact full-graph G2/G3 structural audit on frozen size-6 targets.
- Candidate domain [-1,1].
- Stage3F thresholds unchanged: ESS>=3, max weight<=0.50, graph-safe>=20, applicable M6 N3 lower-bound diagnostic finite and >0.
- Refusals reported separately from selective coverage.

## Expected raw row counts
- Primary query rows: 187,500
- Exact audit rows: 16,200
- Full-inversion rows: 150
- Endpoint-audit rows: 4,000
- Hard-dose prediction rows: 227,500
- Nuisance-fit audit rows: 1,300
- Spatial diagnostics: 7,800
- LOMO rows: 1,875
- Runtime rows: 540

The runner refuses if any count differs.

## Result bundle
Only after the terminal verifier passes, the runner creates:

`STAGE5B_PRODUCTION_OUTPUTS\GeoDose_Stage5B_MineDoseBench_PRODUCTION_RESULTS.zip`

Stop after that and perform an independent audit before the NSW real-demonstration stage.

## v1.0.2 software correction
The scientific Stage5B v1.0.0 contract is unchanged. This patch fixes only the S8_SMALL_CAL exact-size6 structural-audit calibration-frame bug. Existing unaffected v1.0.0 atomic checkpoints are intentionally reusable. No threshold, target, seed, predictor, nuisance hyperparameter, or Stage5A byte is changed.
