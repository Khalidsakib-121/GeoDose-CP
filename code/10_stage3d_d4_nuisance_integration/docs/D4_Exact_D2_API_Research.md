# D4 exact-D2 API research and binding record

## Why this record exists

Earlier D4 adapter drafts inferred parts of the frozen D2 callable interface from output artifacts and incomplete source visibility. That was not sufficiently reliable for a source-level research gate. D4 v1.1.0 was rebuilt only after the complete D2 source package was supplied and checked against the accepted D2 source inventory.

## Exact source provenance

- Supplied D2 source ZIP SHA-256: `ce6075cf18c9dfab4e057472c3afa08deb1cada3186172d52286a57516dd41a5`
- Accepted D2 output ZIP SHA-256: `fd2af9c4fb4f6563a3b2e0a56559425b832a1dde330624964470c29ea983ddf1`
- Accepted D2 script version: `1.2.0-stage3d-d2-freeze`
- Accepted D2 source inventory: 41 files
- Supplied source versus accepted source inventory: 41/41 byte hashes matched
- Accepted D2 source tree hash: `8348a72833ab43e622a4ff6083fa057b828c4c2b5c1e49cec22ec43d075a5ecc`

D4 does not modify or vendor the D2 exact law. It imports the user's verified local D2 source tree after the 41/41 verification succeeds.

## Exact callable signatures frozen by D4

D4 v1.1.0 validates these signatures before accepted replay or new-case evaluation:

- `stage3b_data(package_root: 'str | Path') -> 'Stage3BData'`
- `load_d1_fixture_config(package_root: 'str | Path', d1_evaluation_id: 'str') -> 'dict[str, Any]'`
- `build_d2_fixture(data: 'Stage3BData', package_root: 'str | Path', d1_evaluation_id: 'str')`
- `frozen_orbit_seed(package_root: 'str | Path', scenario_id: 'str' = 'S4', replication: 'int' = 1) -> 'int'`
- `derive_seed(base_seed: 'int', *labels: 'str') -> 'int'`
- `prepare_orbit(fixture: 'ExactFixture', *, minimum_precision_eigenvalue_required: 'float') -> 'PreparedOrbit'`
- `CandidateEvaluator(prepared: 'PreparedOrbit', *, alpha: 'float', tie_uniform: 'float', score_abs_tolerance: 'float', score_rel_tolerance: 'float', non_gaussian_min_abs_residual: 'float') -> 'None'`
- `CandidateEvaluator.evaluate(self, candidate_y: 'float', *, keep_state: 'bool' = False) -> 'CandidateResult'`
- `acceptance(result: 'CandidateResult', mode: 'str') -> 'bool'`

The signature lock is stored in `configs/d2_api_lock.json`. Any drift fails closed.

## Frozen D2 values used by the adapter

All values are read from the byte-verified D2 configuration rather than invented by D4:

- primary alpha: `0.10`
- minimum precision eigenvalue: `1e-10`
- score absolute tie tolerance: `1e-12`
- score relative tie tolerance: `1e-12`
- non-Gaussian minimum absolute residual: `1e-14`
- accepted-candidate replay tolerance: `2e-12`

## Exact D2 randomization contract

The accepted D2 runner uses one fixture-specific uniform held fixed across candidate responses. It derives that uniform by:

1. obtaining the S4 replication-1 orbit seed;
2. deriving a sub-seed using `("D2_G3_TIE_UNIFORM", fixture_id)`;
3. drawing one `numpy.random.default_rng(derived_seed).random()` value.

D4 reproduces that contract exactly. It does not use the new case's scenario ID to choose the base orbit seed.

The accepted replay gate checks not only the primary conservative p-value, but also the randomized p-value, tie uniform, conservative/randomized acceptance decisions, and distinct quotient-state count. This makes an incorrect tie seed detectable.

## Accepted replay obtained during pre-release end-to-end test

Using the complete verified D2 source, D4 v1.1.0 reproduced:

- `G6_GAUSS_INTERIOR`, `y=-8`: conservative p error `2.22e-16`; randomized p error `0`; tie-uniform error `5.55e-17`; 720 states.
- `G6_GAUSS_INTERIOR`, `y=0`: conservative p error `0`; randomized p error `5.55e-17`; tie-uniform error `5.55e-17`; 720 states.
- `G6_GAUSS_INTERIOR`, `y=8`: conservative p error approximately `8.43e-81`; randomized p error approximately `4.22e-81`; tie-uniform error `5.55e-17`; 720 states.
- `G6_GAUSS_DUPLICATE`, `y=0`: conservative/randomized p errors `0`; tie-uniform error `1.11e-16`; 360 states.

These are implementation-regression checks, not publication-performance evidence.

## Target-slot identity and outcome alignment

Stage3B contains 125 units with role `test_target` per validation case. D4 exact size-6 inference does **not** select a target by role alone. It freezes the target slot to node `320`, which is the target node already used by the accepted size-6 exact fixtures.

D4 v1.1.0 has an explicit target-selector helper and regression tests. The outcome-alignment audit records the selected target unit/node and requires all eight registered case-dose entries to use node 320.

D4 intentionally excludes Stage3B measurement-error cases such as S9. For every D4 case, it verifies before exact computation that factual `Y_observed_at_A` equals `Y_true_at_A`, and that target `Y_observed_at_A_star` equals `Y_true_at_A_star`, within `1e-14`. The pre-release full run found zero difference for all eight registered case-dose entries.

## Claim boundary

This research closes the source-API integration uncertainty for D4. It does not turn D4 into a scalable method and does not establish repeated-coverage validity under estimated nuisances. D4 remains a small-exact-block integration gate. Stage3E/N2-N3 remains the next scientific stage before the preregistered pilot.
