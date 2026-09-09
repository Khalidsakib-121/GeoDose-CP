# Stage3F Mathematical and Algorithmic Audit

## 1. Frozen scope

Stage3F is a controlled 20-replication pilot/tuning gate. It does not alter G1/G2/G3, M3/M4/M6 definitions, Stage3E N2/N3 mathematics, the Stage3B DGP, or accepted upstream results.

## 2. Target population

The Stage3A controlled target population is uniform over theorem-eligible final-test slots under the frozen split. Stage3F operationalizes eligibility without A, Y, support diagnostics, or method output.

For each candidate test target t, Stage3F deterministically grows a size-6 block:

B = {t, c1, ..., c5}

where every newly selected calibration node is adjacent to the already selected set under the frozen true graph. Therefore the induced subgraph on B is connected by construction. A final independent induced-connectivity check fails closed.

A separate deterministic random stream derived from the frozen target-draw seed selects uniformly from the eligible target set. Factor variants in a scenario share the same uniform rank to preserve common-random-number comparability.

## 3. Scores and nuisance roles

The standardized residual architecture is preserved. The oracle structural route uses latent Y_true and the frozen oracle conditional mean. Shared RF/XGB predictor routes use observed EO-product outcomes and are empirical diagnostics, not finite-sample theorem-certified routes.

Treatment transport and target-design transport remain distinct. The current S1–S10 pilot requires identity target-design transport because Stage3E does not certify the nonidentity estimated target-design ratio branch.

## 4. M1–M6

- M1: standard split conformal baseline.
- M2: dose-only weighted conformal using q_h/g.
- M3: accepted spatial-only residual-orbit conformal; no treatment ratio and no target-design ratio.
- M4: accepted naive product with exactly one normalization of d_j s_j; no target-design ratio.
- M5: accepted graph-safe/block baseline.
- M6: accepted D2 G1/G2/G3 candidate evaluator patched with accepted Stage3E m=64 sparse precision.

No private replacement of the accepted exact orbit or candidate evaluator is introduced.

## 5. N2/N3 pilot certificate

For the registered identity target-design pilot cases, Stage3E supplies the sparse approximation KL Δ_N2. Stage3F uses

δ_sparse = sqrt(max(Δ_N2,0)/2)

and the oracle structural lower bound

LB = max(0, 1 - α - δ_sparse)

with α=0.1. Estimated nuisance/misspecification terms are not silently set to zero for empirical tracks; those tracks are explicitly non-certified.

## 6. Operational threshold freeze

Only the accepted Stage3C candidate grids are searched:

- ESS_min ∈ {3,5,10,15,20}
- max normalized weight ∈ {0.10,0.15,0.20,0.30,0.50}
- graph-safe count_min ∈ {5,10,15,20}

No fourth pilot-tuned certificate threshold is added. N3 non-vacuity is a fixed structural prerequisite: finite LB > 0.

Selection uses only oracle structural M6 controlled-pilot outcomes plus pre-outcome support/certificate diagnostics. NSW, RF/XGB outcomes, width, and production outcomes are excluded.

The Stage3F-frozen lexicographic criterion searches all 100 tuples in the three accepted Stage3C grids. The grids and the controlled-pilot coverage/refusal principle are upstream-frozen; this document does not claim that every detailed lexicographic tie-break was preregistered before Stage3F software development. It ranks them by: (1) minimum worst safe-case **penalized undercoverage shortfall**, (2) minimum safe pooled undercoverage shortfall, (3) maximum number of safe-case coverage groups with at least 10 returned pilot targets, (4) minimum return rate in expected-low-information S5/S8 cases, (5) maximum safe-case retention, and then deterministic stricter-threshold tie-breaks. A safe case with fewer than 10 returned targets receives a fixed maximal casewise validity penalty of 1.0. There is **no post-hoc hard retention cutoff** and width is absent from selection.

Pilot-scale readiness requires all registered safe cases to satisfy the same Stage3F-frozen minimum of 10 returned targets used by the casewise coverage objective. This prevents freezing a tuple whose apparent coverage is based on an inadequately small selective subset; it does not introduce a new tunable threshold.

Because the same 20-rep pilot chooses the thresholds, post-freeze binomial coverage checks are descriptive only. The first independent evaluation is the production run.

## 7. Graph misspecification

The target population and exact block are frozen under the true structural graph. For an empirical S6 fitted-graph diagnostic, the fitted graph may disconnect the frozen block. Retargeting would alter the target population and contaminate comparison, so Stage3F instead returns a scientific refusal for graph-dependent M3/M4/M6:

R05_GRAPH_COMPONENT_TOO_SMALL

M1/M2 remain evaluable; M5 follows its graph-safe baseline logic. The event is recorded in the graph audit and is not marked as a computational failure.

## 8. Claim boundaries

Stage3F may freeze operational thresholds and establish readiness for independent production review. It does not provide publication-performance evidence, does not certify estimated nuisance routes, does not establish matched-coverage efficiency from the diagnostic width grid, and does not tune on NSW.
Stage3F explicitly maps the Stage3A metric/reporting registry: full matched-coverage width/WIS and M6 component topology remain production/ablation deliverables, and Moran's I/semivariogram remain diagnostic-only production outputs. Query rows nevertheless carry requested dose, bandwidth, support/positivity, branch, block/boundary, nuisance-certificate, deficit, numerical and refusal fields needed to audit the pilot.
