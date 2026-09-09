# Stage4 frozen mathematical and algorithmic audit

## Purpose

Stage4 is the first untouched production evaluation after Stage3F threshold freeze. It does not change M1–M6, G1/G2/G3, N2/N3, or the operational thresholds. Its job is to (i) freeze/QA F06 before production outcomes, (ii) consume every frozen Stage3A production seed row exactly once in the 3000-row main experiment, (iii) run registered stress/ablation evidence, and (iv) create publication-facing controlled evidence without using NSW for tuning.

## Frozen operational gate

The accepted Stage3F values are immutable: minimum ESS 3, maximum normalized weight 0.50, minimum graph-safe count 20, and M6 N3 non-vacuity requires a finite strictly positive coverage lower bound. Stage4 contains no threshold-selection routine.

## F06 prospective freeze

F06 is a **fixed-resolution increasing-domain sample-size sensitivity**, not a resolution/MAUP experiment:

- small: 17x17 = 289, 90 m queen, 1056 full geographic queen edges, 958 role-cut DGP edges;
- primary: 25x25 = 625, 90 m queen, 2352 full geographic queen edges, 2206 role-cut DGP edges;
- large: 35x35 = 1225, 90 m queen, 4692 full geographic queen edges, 4486 role-cut DGP edges.

The role cut is part of the accepted Stage3B information-split DGP. Only n/domain extent changes. Treatment law, outcome law, residual law/rho, q/g, target-design definition, target-population rule, intervention kernel/endpoint atoms, measurement error, and Stage3F thresholds remain fixed. S10 180 m is not reused as F06-small.

For S4_RHO060 the F06 extensions use master Stage3A S4 production replications 301–400, matching its main-production window. S1_BASE and S5_POOR_OVERLAP use 1–100. F06 small/large have distinct derived common-random-group substreams, so no unsupported strict nested-CRN claim is made.

## Production allocation

The 3000 `production_provisional` Stage3A rows are partitioned prospectively among the frozen case matrix with no reuse/omission. S4 gets 100 rows at each rho 0, .2, .4, .6, .8. S6, S8, S9, and S10 are partitioned across their registered factor levels before any production result is read.

## Spatial/scalable branch

M6 candidate evaluation reuses the accepted D2 CandidateEvaluator and accepted Stage3E m=64 sparse precision/boundary machinery. It does not implement a new G1/G2/G3 engine. q_h/g and target-design transport remain separate. Nonidentity design-shift evidence is explicitly diagnostic because the exact local performance sample is not a shifted target-population sample and no new estimated-ratio finite-sample theorem is invented.

## Fitted graph misspecification

The target/block is selected under the frozen true graph before outcome/support inspection. S6 fitted-graph misspecification is not allowed to retarget. If the registered working graph disconnects the frozen block, graph-dependent methods fail closed. ABL9 evaluates the same empirical RF response model with the true graph only as an oracle diagnostic.

## Spatial diagnostics

Moran's I and the empirical semivariogram are diagnostic-only. With an undirected edge list storing each edge once,

`I = n/E * sum_(i,j in edges) [(e_i-e_bar)(e_j-e_bar)] / sum_i (e_i-e_bar)^2`.

The edge semivariogram is `mean[0.5 (e_i-e_j)^2]`. Neither is substituted for an N2/N3 deficit.

## Efficiency boundary

The accepted G3 numerical inversion is a coverage-preserving outer numerical set only on the frozen candidate domain [-8,8]. Stage4 therefore evaluates M6-v-M5 width on a pre-fixed 50-query S4 subset and intersects M5 with the same domain. `full_real_line_claim=false` always. If M6 touches a domain boundary, no full-line efficiency conclusion is allowed. Coverage matching uses all independent production replications and an absolute tolerance of 0.03; widths use only the fixed subset.

## Ablations

ABL0 full M6; ABL1 M3; ABL2 target-design certificate ratio diagnostic; ABL3 M2; ABL4 naive product M4; ABL5 records the algebraic coincidence with M2 on the identity-design branch; ABL6 raw M6 before frozen operational refusal; ABL7 H1/H4 geographic heuristics (never theorem eligible); ABL8 estimated/misspecified treatment nuisance diagnostic (not finite-sample theorem certified); ABL9 S6 fitted versus oracle spatial graph for the same RF response model.

## Claim boundaries

- Causal identification and conformal validity remain separate.
- ST3 is a hidden-C2 causal-identification negative control; calibration cannot repair identification failure.
- ST2 is a structural temporal diagnostic; no new rowwise temporal conformal theorem is claimed.
- Estimated treatment/outcome nuisance tracks remain empirical diagnostics.
- Production outcomes are never used to retune Stage3F thresholds.
- NSW is absent from tuning/production configuration.
