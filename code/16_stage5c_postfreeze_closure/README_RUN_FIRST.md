# GeoDose-CP Stage5C Post-Freeze Computational Closure v1.0.0

## Purpose

This is the final computational closure before manuscript revision.

It does **not** train, tune, or change GeoDose-CP. It verifies and freezes four evidence blocks:

1. registered Stage5B MineDoseBench primary results;
2. verified exact-sparse deep audit;
3. estimated-g(A|U) sensitivity;
4. E1/E2/E3 external-baseline comparison.

Stage6A NSW remains separately frozen and is deliberately not reprocessed here.

## Run

1. Extract the ZIP to a short Windows path, for example `D:\G5C`.
2. Double-click `RUN_STAGE5C_CLOSURE.bat`.
3. Keep `INPUTS`, `config`, and `_STAGE5C_WORK` unchanged.
4. If the run is interrupted, run the BAT again. Verified completed closure stages are reused.
5. Final outputs appear under `STAGE5C_CLOSURE_OUTPUTS`.

## Final deliverable

`STAGE5C_CLOSURE_OUTPUTS\GeoDose_Stage5C_PostFreeze_Closure_RESULTS.zip`

The final results package contains manuscript-ready tables, figures, evidence classification, claim boundaries, a closure report, verification JSON, and an output manifest.

## Scientific hierarchy

- Stage5B remains the governing registered primary benchmark.
- Exact-sparse, estimated-g, and external-baseline analyses are post-freeze secondary evidence.
- No post-freeze analysis replaces or retrospectively modifies the registered primary benchmark.
- Width comparisons remain restricted to their registered/identity-matched subsets and must not be generalized as universal efficiency evidence.
