# CHANGELOG

## v1.1.0 — Exact D2 source-locked D4 rebuild

This is a clean consolidation release after inspection of the complete accepted D2 source tree.

- Verified the supplied D2 source against the accepted D2 source inventory: 41/41 files match; tree hash `8348a72833ab43e622a4ff6083fa057b828c4c2b5c1e49cec22ec43d075a5ecc`.
- Replaced inferred D2 constructor/API mapping with an exact fail-closed API lock for every callable D4 uses.
- Binds the accepted keyword-only `prepare_orbit(..., minimum_precision_eigenvalue_required=...)` argument from the frozen D2 tolerance `1e-10`.
- Binds the accepted `CandidateEvaluator` argument `non_gaussian_min_abs_residual` from the frozen D2 tolerance `1e-14`.
- Reproduces the accepted D2 tie-randomization scheme exactly: S4 replication-1 base seed, derived label `D2_G3_TIE_UNIFORM`, and fixture ID; one uniform is fixed across candidates.
- Accepted source replay now validates conservative p, randomized p, tie uniform, both acceptance decisions, and quotient-state counts.
- Adds `configs/d2_api_lock.json` and `docs/D4_Exact_D2_API_Research.md`.
- Adds an explicit frozen target-slot selector: Stage3B has many `test_target` units, while D4 exact size-6 inference uses node 320. Role-only target selection is prohibited and tested.
- Adds an observed-vs-true outcome alignment gate for all registered D4 cases, excluding measurement-error cases from this exact wiring gate.
- Preserves the accepted D3 exact-integration dependency `v1_0_1` with SHA-256 validated fallback discovery.
- Retains fail-fast Windows input preflight, ZIP/temp self-relocation, partial-venv rebuild, heterogeneous historical verifier-schema support, signed-zero fix, and M4 one-normalization regression.
- Full pre-release logic run against the complete verified D2 source passed, followed by independent D4 verification `64/64`.
- No GeoDose G1/G2/G3 algebra, M3/M4/M6 scientific definition, registered case, candidate grid, accepted D2 tolerance value, threshold, or claim boundary was changed.

## v1.0.5 — Preflight/provenance hardening

- Added stdlib-only pre-venv path/hash/source preflight.
- Added broad D2 API signature preflight and aligned release provenance.
- This version still inferred one required D2 constructor argument and is superseded by v1.1.0.

## v1.0.4 — D3 integration path correction

- Corrected the accepted D3 exact-integration dependency to `v1_0_1`.
- Added SHA-256-validated fallback discovery.

## v1.0.3 — Windows ZIP/temp launcher hardening

- Added self-relocation from Windows Explorer compressed-folder temporary paths.
- Added partial/incompatible virtual-environment rebuild.

## v1.0.2 — D2 `prepare_orbit` binding correction

- Added the accepted keyword-only precision-eigenvalue argument from the frozen D2 tolerance registry.

## v1.0.1 — Historical verifier-schema compatibility

- Added fail-closed handling of the accepted Stage3B/Stage3C/D2-D3 verification JSON variants.

## v1.0.0 — Initial D4 integration gate

- Added source-level M6 new-data adapter concept, nuisance wiring, exact smoke cases, M3 invariance, M4 recomputation, graph diagnostics, and expected refusal checks.
