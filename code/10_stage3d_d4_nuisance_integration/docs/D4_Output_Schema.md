# Stage 3D D4 output schema

The Windows runner creates `outputs_stage3d_d4/` and, only after independent verification, packages the verified files into `GeoDose_Stage3D_D4_M6_NEWDATA_NUISANCE_OUTPUTS.zip`.

## Provenance and source replay

- `stage3d_d4_input_audit.json` — eight accepted upstream archive hashes, Stage 3C source-tree verification, D2 source-tree verification, and vendored M3/M4 byte audit.
- `stage3d_d4_d2_source_replay.csv` — source-level replay of accepted D2 interior candidates plus the accepted duplicate-payload candidate.
- `stage3d_d4_d2_runtime_api.json` — exact runtime signatures/types observed from the accepted local D2 source API.
- `stage3d_d4_source_hashes.json` — D4 source inventory used for the run.
- `stage3d_d4_environment_inventory.json` — frozen execution environment.
- `stage3d_d4_manifest.json` — hashes and sizes of all manifest-tracked output files.
- `STAGE3D_D4_VERIFICATION.json` — independent D4 verifier result.

## New-data/nuisance wiring

- `stage3d_d4_candidate_trace.csv.gz` — candidate-level M3/M4/M6 p-values and inclusion decisions for the fixed smoke grid.
- `stage3d_d4_m3_m4_source_weight_audit.csv.gz` — M3 source marginals, dose log-ratios, M4 masses, and source residuals.
- `stage3d_d4_estimated_treatment_fit_audit.csv` — Stage 3C estimated-treatment fits restricted to nuisance-training units.
- `stage3d_d4_outcome_alignment_audit.csv` — verifies the frozen target node 320 and requires observed-versus-true factual/target outcomes to coincide for the D4 exact-wiring cases; measurement-error cases are excluded from this gate.
- `stage3d_d4_graph_variant_audit.csv` — TG/FG edge counts, connectivity, and block-degree diagnostics.
- `stage3d_d4_m6_prepare_audit.csv` — source-level D2 preparation metadata, including endpoint/interior base configuration and iid-as-rho0-GMRF representation notes.
- `stage3d_d4_m4_recompute_audit.csv` — independent one-normalization reconstruction of M4 from `d_j s_j(y)`.
- `stage3d_d4_m3_treatment_invariance.csv` — M3 OT-vs-ET invariance checks at fixed graph/candidate.
- `stage3d_d4_nuisance_delta_diagnostics.csv` — diagnostic p-value changes due to ET and FG for M4/M6. These are not publication performance results.

## Fail-closed and regression checks

- `stage3d_d4_expected_refusal_audit.csv` — expected refusal smoke tests for an unaudited endpoint and a disconnected exact block.
- `stage3d_d4_refusal_log.csv` — unexpected refusals encountered during supported D4 smoke cases; expected to be empty when the gate passes.
- `stage3d_d4_upstream_reduction_regression.json` — preserves accepted D3 R1/R2/R4 and R3 deferral plus accepted M4 structural nonfactorization.
- `stage3d_d4_case_summary.csv` — compact candidate p-value range/inclusion summary by case, dose, variant, and method.
- `stage3d_d4_claim_boundary.json` — machine-readable statement of what D4 does and does not establish.
- `stage3d_d4_counts.json` — row/count reconciliation.
- `stage3d_d4_runtime.json` — runtime metadata.

## Claim boundary

These outputs are software/scientific gate diagnostics. They are not repeated-coverage evidence, pilot results, production results, scalable N2/N3 certificates, MineDoseBench evidence, NSW results, or publication-performance claims.
