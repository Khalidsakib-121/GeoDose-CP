# GeoDose-CP Stage 3D D2 v1.2 FREEZE

This package is the Stage 3D D2 **reference G3 candidate-inversion gate**.
It inherits the accepted D0/D1 G1/G2 algebra engine and implements:

- candidate-specific quotient orbits after target-response replacement;
- exact G2 orbit-law recomputation at each candidate;
- conservative G3 p-values;
- randomized p-values using only certified structural ties;
- full inclusion of numerically ambiguous near-tie mass;
- deterministic candidate-grid inversion;
- coverage-preserving outer numerical boundary brackets;
- finite-domain, domain-truncated prediction-set components;
- correct endpoint topology and component metadata;
- analytic end-to-end topology tests for connected, disconnected, empty,
  isolated-point, and discontinuous-boundary sets;
- independent offset-grid false-negative and boundary-excess audits;
- size-8 40,320-state p-value stress testing.

## Important claim boundary

The controlled latent outcome contains Gaussian residuals and therefore has
unbounded mathematical support. D2 does **not** infer unbounded prediction sets
from finite tail evaluations. It returns a clearly labelled, coverage-preserving
outer numerical approximation to

`C_alpha ∩ [-8, 8]`

on a fixed domain registered before any outcome is inspected. No full-real-line
or unbounded-set claim is made.

This gate does not evaluate repeated-replication coverage and does not implement
M3, M4, M6, N2/N3, MineDoseBench, or production inference.

## Run

Double-click `RUN_STAGE3D_D2.bat` on Windows. The runner creates or verifies a
package-local Python 3.10 environment with the exact versions in
`requirements_py310.txt`, runs unit tests, generates D2 outputs, runs the
independent verifier, and creates `GeoDose_Stage3D_D2_OUTPUTS.zip`.

Accept only a terminal ending that reports all verification checks passed,
five full-inversion fixtures, five topology-validation fixtures, domain-
truncated outer sets, no unbounded claim, no coverage evaluation, and no
M3/M4/M6/production run.
