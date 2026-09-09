# Recovery and resumability

The production runner writes extraction checkpoints under the existing acquisition cache:

`CONTEXT_SOURCE_CACHE/_final_context_production_v1_0_1/`

Each scalar checkpoint is keyed by exact source-raster SHA-256, retained block-ID SHA-256, governing-layer ID and the unchanged 0.99 threshold. Re-running the BAT reuses matching checkpoints. A source-byte or block-registry mismatch invalidates that checkpoint and fails/recomputes only the relevant production unit; it does not delete the upstream cache.

Do not delete `CONTEXT_SOURCE_CACHE` until the final Stage5A archive is frozen and backed up.
