# GeoDose Stage4 — Full Controlled Production + F06 + Ablation/Efficiency

This is the production gate after accepted Stage3F. It is intentionally interruption-safe and may run for hours on a Windows CPU.

## Frozen inputs

The launcher hash-validates accepted Stage3A, Stage3B, D4, Stage3E and Stage3F output ZIPs plus the exact locked Stage3B/3C/D2/D4/3E source files before environment creation. The accepted Stage3F thresholds are read from the accepted output ZIP and cannot be recomputed or tuned here.

## Work performed

- prospectively frozen F06 17x17 / 25x25 / 35x35 at fixed 90 m queen support;
- structural F06 QA before production evaluation;
- all 3000 Stage3A `production_provisional` main rows exactly once;
- 600 F06 extension case-replications;
- 590 registered stress extensions (100 ST2 structural-only; 490 method-evaluable);
- M1–M6 with accepted Stage3F/Stage3E/D2/D4 machinery;
- RF empirical track everywhere in main production and XGB confirmation only for S1/S4/S5/S8;
- H1/H4 heuristic ablations;
- ABL0–ABL9 evidence with theorem/diagnostic boundaries recorded;
- independent production coverage confirmation using thresholds frozen on Stage3F;
- local/dose coverage and fifth-percentile local coverage;
- WIS/width where mathematically represented by the accepted baseline methods;
- diagnostic Moran's I and semivariogram, never used as conformal deficits;
- fixed-domain M6-v-M5 matched-coverage efficiency audit for 50 pre-fixed S4 queries;
- deterministic manifests and independent output verification.

## Resumability

One gzip-JSON checkpoint is atomically written after every case-replication and bound to the Stage4 source aggregate and production-contract hash. The normal launcher does **not** use `--overwrite`. If a session stops, rerun `RUN_STAGE4.bat` and it resumes completed checkpoints.

## Scientific boundaries

Full-real-line M6 width is not claimed. Estimated/misspecified nuisance diagnostics are not finite-sample theorem-certified. The target-design-shift stress is a certificate/transport diagnostic rather than an invented shifted-target performance theorem. ST2 is structural-only; ST3 is a causal-identification negative control. No NSW tuning is used.

A successful run creates `GeoDose_Stage4_PRODUCTION_F06_ABLATION_EFFICIENCY_OUTPUTS.zip`. Send it for independent review and do not start MineDoseBench until accepted.
