# GeoDose-CP Stage 3D D0/D1 Exact-Orbit Algebra v1.1

## Scope

This package implements only the mathematical execution freeze (D0) and the
exact finite-orbit algebra engine (D1) for the GeoDose-CP G1/G2 branch.

Implemented:

- frozen machine-readable theorem, nuisance, positivity, numerical, and refusal contracts;
- fixed spatial slots and movable numerical `(A,Y)` payloads;
- candidate replacement before quotient-orbit construction;
- candidate-specific distinct-orbit enumeration, including target/calibration collisions;
- exact mixed endpoint/interior treatment likelihood;
- target-slot localized intervention tilt and `q/g × g = q` audit;
- positivity over the complete intervention support and over every realized orbit payload;
- mandatory inverse outcome Jacobian direction;
- graph-boundary extraction;
- proper Gaussian GMRF block-boundary residual factor;
- one explicit non-Gaussian componentwise transformed-GMRF specialization;
- independent loop-based scalar and vectorized exact engines;
- full-joint target-law versus graph-local G2 probability audit;
- size-6, size-8, endpoint, duplicate, candidate-collision, boundary, and non-Gaussian fixtures.

Not implemented:

- G3 candidate p-values or candidate inversion;
- prediction sets or coverage;
- final M3, M4, or M6;
- N2/N3 scalable certification;
- estimated-nuisance exact-coverage claims;
- MineDoseBench or production inference.

This boundary is intentional. D1 must pass independently before D2 begins.

## Critical v1.1 correction

Under G1, payload equality is the exact numerical pair `(A,Y)` after replacing
the augmented target response by the candidate value. Source origin is metadata
only. Therefore, when the candidate pair equals a calibration pair, the
quotient orbit and multiplicities are recomputed and collapse accordingly.

## Nuisance claim boundary

D1 uses accepted Stage 3B generator truth for the outcome mean, mixed treatment
law, graph, precision parameters, scale, and boundary residuals. This is an
`oracle_generator_truth_algebra_only` gate. It does not claim exact coverage
with estimated nuisance quantities.

## Exact inputs

The package contains and verifies the accepted Stage 3A, Stage 3B, and Stage 3C
artifacts plus the canonical mathematical documents. Any checksum change causes
closed failure.

## Run

1. Extract to a short Windows path.
2. Open the inner directory containing `RUN_STAGE3D_D1.bat`.
3. Double-click `RUN_STAGE3D_D1.bat`.
4. Upload `GeoDose_Stage3D_D1_OUTPUTS.zip` after the run passes.

The first run requires internet access to create the package-local Python 3.10
virtual environment. The runner enforces the exact pinned NumPy, pandas, SciPy,
and PyYAML versions on every run.

## Expected completion

```text
STAGE 3D D0/D1 EXACT-ORBIT ALGEBRA VERIFIED COMPLETE
Verification checks: 69/69
Evaluations: 7
Orbit state rows: 44,280
G3 implemented: no
Prediction sets generated: no
Coverage evaluated: no
Production experiments run: no
```
