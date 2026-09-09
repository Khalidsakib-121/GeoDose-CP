# GeoDose-CP Stage 3B v1.2 FREEZE — Controlled Generator and Realized Target Engine

## Acceptance boundary

Stage 3B v1.2 FREEZE implements and validates the controlled known-ground-truth generator required by frozen Stage 3A. It stops before any conformal method is implemented.

It generates:

- known mixed treatment laws with genuine atoms at 0 and 1 plus a continuous interior;
- nonlinear potential-outcome truth with coherent unit residuals;
- realized localized intervention draws `A_star` and `Y_i(A_star)`;
- treatment/intervention transport truth `q_h(A|a0)/g(A|X)`;
- a separate N2/N3 target-design density-ratio truth object;
- isolated random, treatment-correlated, substrate-specific, and combined EO measurement-error fixtures;
- graph, temporal, support, hidden-confounding, and exact-orbit fixtures.

It does **not** implement M1–M6, candidate-response inversion, coverage certificates, empirical coverage, interval width, or production experiments.

## Frozen input

The package embeds the independently accepted Stage 3A archive:

`inputs/stage3a/GeoDose_Stage3A_OUTPUTS.zip`

Required SHA-256:

`9a46e21f4f488447f0829f079c8f19a345e6ac1748bc6907e20917bbb096d406`

## Why v1.2 supersedes v1.1

The final consolidated audit of v1.1 identified four generator-contract gaps:

1. `q_h/g` was described as a target-design ratio even though the frozen mathematics treats treatment transport and N2/N3 target-design transport as distinct objects.
2. S9 combined substrate bias with every treatment-error fixture instead of isolating the registered random, treatment-correlated, and substrate-specific components.
3. Exact-orbit fixtures used the factual observational target payload rather than a realized localized target payload with an explicit G3 candidate-response replacement rule.
4. S1 silently replaced the frozen primary bandwidth 0.10 with the 0.20 sensitivity bandwidth.

v1.2 corrects all four in one release.

## Validation matrix

The release generates 27 deterministic validation cases covering:

- S1–S10 and ST1–ST3;
- all five S4 residual-dependence levels;
- a non-Gaussian transformed-GMRF S4 case;
- `S4_RHO060_DESIGN_SHIFT`, a known non-identity N2/N3 design-ratio case;
- S6 rook and 50% edge-omission graphs;
- S8 severe weight concentration and small calibration information;
- five S9 fixtures: random, treatment lambda 0.5, treatment lambda 1.0, substrate bias, and combined error;
- S10 90 m and anchored 180 m support-level reruns;
- genuine mixed endpoint atoms and an interior-only refusal reference;
- repeated unit-year dependence;
- a hidden-spatial-confounding C2 negative control;
- exact-orbit fixtures of sizes 6 and 8.

## Separate transport objects

### Treatment/intervention transport

`stage3b_treatment_transport_truth.csv.gz` stores the exact mixed-measure ratio

`q_h(A | a0) / g(A | X)`.

### Target-design transport

`stage3b_target_design_truth.csv.gz` stores the N2/N3 design ratio. Ordinary cases use the identity ratio. The dedicated design-shift case uses

- design object: `D = X_nonlinear_1`;
- observational law: `N(0,1)`;
- target law: `N(delta,1)`, `delta = 0.75`;
- exact ratio: `r(D) = exp(delta*D - delta^2/2)`.

Raw coordinates are excluded from this ratio object so the common-support model is explicit. Nuisance-training rows are the observational ratio-training sample; support-audit rows are the independent target ratio-training sample. Calibration and final-test rows are evaluation samples.

## Exact-orbit fixture contract

Each fixture has one target slot and calibration slots otherwise. Calibration payloads use factual observed `(A,Y)`. The target payload uses a realized localized `A_star` and the simulation truth response. Stage 3D must replace the target response by candidate `y` for every G3 inversion state. The stored truth response is an oracle coverage-test reference only.

## Main outputs

- `stage3b_validation_units.csv.gz`
- `stage3b_hard_truth.csv.gz`
- `stage3b_treatment_transport_truth.csv.gz`
- `stage3b_target_design_truth.csv.gz`
- `stage3b_realized_target_draws.csv.gz`
- `stage3b_observational_target_draws.csv.gz`
- `stage3b_oracle_support_diagnostics.csv`
- `stage3b_temporal_stress.csv.gz`
- graph and MAUP fixtures
- `stage3b_exact_orbit_fixtures.json`
- `stage3b_manifest.json`
- `STAGE3B_VERIFICATION.json`

## Run on Windows

Double-click:

`RUN_STAGE3B.bat`

The runner requires:

- Python 3.10
- NumPy 2.2.6
- pandas 2.3.3

It reuses `D:\GeoDose_Stage2B_Local_STAC_v1_0\.venv\Scripts\python.exe` when available. No internet or DEA access is required.

## Required result

```text
STAGE3B GENERATOR UNIT TESTS PASSED
STAGE 3B GENERATION COMPLETE
Validation cases: 27
Pilot replication: 1
Unit rows: 16,419
Realized target-draw rows: 23,498
Methods implemented: no
Coverage evaluated: no

STAGE 3B VERIFIED COMPLETE
Validation cases: 27
Unit rows: 16,419
Realized target-draw rows: 23,498
Methods implemented: no
Coverage evaluated: no
Production experiments run: no
```

The runner creates `GeoDose_Stage3B_OUTPUTS.zip`.

## Freeze rule

Freeze Stage 3B after the user's exact Python 3.10 run produces `STAGE 3B VERIFIED COMPLETE` and the resulting output archive passes independent hash/manifest audit.

Reopen Stage 3B only if its frozen generator truth is internally contradictory or a frozen upstream input/specification changes. Later implementation work in Stage 3C–3E is not by itself a reason to revise Stage 3B.
