# Changelog

## v1.0.0-stage3f-pilot-freeze
- Implements the preregistered S1–S10 20-replication pilot and operational threshold freeze.
- Searches only the three accepted Stage3C threshold candidate grids; no unregistered certificate-threshold grid is added.
- Uses a deterministic coverage/refusal lexicographic selector with a Stage3F-frozen 10-return casewise validity objective and maximal insufficient-return penalty; no post-hoc retention cutoff and no width/NSW tuning.
- Blocks pilot-scale readiness unless every registered safe case contributes at least the same 10 returned targets used by the casewise validity objective.
- Implements deterministic uniform selection over structurally theorem-eligible final-test slots.
- Enforces induced connectivity of every structural size-6 exact block.
- Handles fitted-graph disconnection as a fail-closed scientific refusal without retargeting or crashing.
- Reuses accepted D2/D4/Stage3E engines and locks their exact source bytes/APIs.
- Keeps oracle structural M6 theorem-certified and empirical nuisance routes diagnostic only.
- Adds provenance preflight, deterministic outputs, independent verifier, and Windows-safe launchers.
- Adds explicit Stage3A reporting-contract fields and metric-scope mapping; matched-coverage efficiency and spatial diagnostics remain correctly deferred to production.
- Clarifies that candidate grids/coverage-refusal principle are upstream-frozen while the detailed Stage3F lexicographic implementation is frozen before production rather than overclaimed as an earlier detailed preregistration.
