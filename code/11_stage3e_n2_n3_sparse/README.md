# GeoDose Stage 3E — N2/N3 Scalable Certification v1.0

This is the pre-pilot scalable-certification gate after accepted D4 v1.1.0.

It implements the frozen N2 Gaussian omitted-information identity, generic target-frame weighting, target-shift amplification, neighborhood monotonicity, R3, an exact sparse special case, the bounded-feature target-design-ratio certificate formula, S3 centered-GMRF score confidence set, separate GMRF-C realized-boundary certificate, N3 separated coverage accounting, treatment-support diagnostics, accepted-D2 exact-vs-sparse G3 checks, a new-data sparse candidate engine, and one full finite-domain sparse prediction-set inversion using the accepted D2 inversion machinery.

It does **not** rewrite G1/G2/G3. It does not run the 20-rep pilot or production experiments. It also does not invent nuisance guarantees that are absent from the frozen mathematics: the Stage3B Gaussian nonidentity target-ratio fixture is explicitly theorem-ineligible for the bounded-feature Hoeffding certificate, the accepted Stage3C ET model is not relabelled as T1-certified, and the S3/GMRF-C certificate is not silently fused into the nonlinear Vecchia estimation error without a frozen bridge.

## Important implementation details

- N2 uses an outcome-blind deterministic maximin ordering and nearest-predecessor neighborhoods `m={0,4,8,16,32,64}`.
- The primary pilot neighborhood is structurally frozen at `m=64` before pilot outcomes; ESS/max-weight/coverage-lower-bound operational thresholds remain deferred to the 20-rep pilot.
- No dense 625×625 precision inverse is formed. Exact full-predecessor conditional variances use sparse reverse elimination; only selected covariance columns are solved for local neighborhoods.
- The exact mixed-treatment transport `q_h/g` is kept separate from the N2/N3 target-design ratio.
- The full scalable-set smoke test preserves D2's finite-domain claim: the returned set is a coverage-preserving outer numerical approximation on the frozen `[-8,8]` domain, not a full-real-line set.
- Stage3E uses the accepted D4 source to prepare new six-slot queries and the accepted D2 G1→G2→G3 engine for candidate p-values and inversion.
- Stage3B `S8_SEVERE_TAIL` stores its valid dose-0.98 target under `target_scope=scenario_primary_extra`. Accepted D2's generic fixture builder filters to `registered_grid`; Stage3E therefore validates the exact frozen D4 target row first, clones the target table locally, aliases only that one cloned row to `registered_grid` for D2 builder compatibility, and records this explicitly in `stage3e_newdata_sparse_prepare_audit.csv`. No upstream Stage3B value is changed.

## Windows

Extract to `D:\GeoDose_Stage3E_N2_N3_Scalable_Certification_v1_0`, open the inner project folder, and run `RUN_STAGE3E.bat`.

The launcher performs provenance checks before creating the Python environment. It verifies the exact accepted Stage3A, Stage3B, D2 and D4 output hashes, the D2 source tree 41/41, and the D4 source tree 39/39 before computation.

On success it creates:

`GeoDose_Stage3E_N2_N3_SCALABLE_CERTIFICATION_OUTPUTS.zip`

Send that ZIP back for independent review. Do not start the 20-rep pilot until the Windows Stage3E output is accepted.
