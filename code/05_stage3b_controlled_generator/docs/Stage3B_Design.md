# GeoDose-CP Stage 3B v1.2 FREEZE Scientific Design

Stage 3B is a generator-validation gate. It converts the frozen Stage 3A architecture into deterministic known-ground-truth data and stops before conformal methods are implemented.

## Primary target

The primary target is the realized localized stochastic potential outcome `Y_i(A_star)`. Interior interventions use a normalized Gaussian kernel truncated to `(0,1)`. Audited synthetic endpoint interventions use exact atoms. The same unit residual is shared across a unit's latent potential-outcome curve.

## Information separation

The controlled graph has three components:

1. independent nuisance-training component;
2. independent support-audit component;
3. calibration-target inference component.

Calibration and target slots remain spatially linked. This preserves the dependence problem while preventing nuisance/support information from being generated within the final inference graph.

## Two distinct transport objects

The treatment/intervention ratio is `q_h(A|a0)/g(A|X)`. It belongs to mixed continuous-treatment transport.

The scalable N2/N3 design ratio is `r(D)=d mu_star/d mu_obs`. It is stored separately. The dedicated design-shift case uses `D=X_nonlinear_1`, an observational `N(0,1)` law, and a target `N(delta,1)` law with known positive ratio `exp(delta*D-delta^2/2)`. Coordinates are not part of `D`.

## Reduction designs

- S1: IID covariates, IID residuals, weak treatment shift, primary bandwidth 0.10.
- S2: IID covariates, spatial residual dependence, observational target law.
- S3: IID covariates, IID residuals, treatment/intervention shift.
- S7: IID covariates, IID residuals, observational target law, linear outcome.

## Measurement-error design

S9 isolates:

- heteroscedastic random error;
- random plus treatment-correlated error at lambda 0.5;
- random plus treatment-correlated error at lambda 1.0;
- random plus substrate-specific bias;
- a combined treatment/substrate case.

The same unit-level innovation is shared across the observed potential-outcome curve. Latent and observed targets remain separate.

## Exact-orbit fixtures

Every size-6/size-8 fixture contains one target slot and calibration slots otherwise. Fixed geometry/roles do not move. Calibration payloads are factual observations. The target payload contains a realized localized `A_star` and oracle truth response. Stage 3D must replace the target response with candidate `y` before every G3 orbit calculation. A duplicate calibration-payload fixture verifies quotient-orbit handling.

## Acceptance boundary

Stage 3B validates generator truth only. M1–M5 belong to Stage 3C, exact G2/G3 to Stage 3D, and scalable N2/N3 certification to Stage 3E. No coverage or efficiency claim is permitted from Stage 3B alone.
