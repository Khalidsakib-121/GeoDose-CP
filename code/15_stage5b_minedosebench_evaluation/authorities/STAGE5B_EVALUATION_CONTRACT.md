# Prospective Stage5B MineDoseBench evaluation contract — v1.1

This contract is frozen with Stage5A before any MineDoseBench method performance is inspected.

## Primary methods
The definitive proposal specifies a **reduced MineDoseBench method set**. The primary benchmark comparison is therefore exactly:
- M2 dose-only weighted conformal;
- M3 graph/residual spatial-only conformal;
- M4 naive dose×spatial marginal product;
- M5 conservative graph-safe/block fallback;
- M6 full GeoDose-CP.

M1 remains part of controlled simulation and the real NSW six-method demonstration but is not a primary MineDoseBench comparator. H1/H4 remain heuristic controlled-simulation ablations.

## Population and repetitions
Each of 27 cases has 20 repetitions. For each case×replication×mine, **five** distinct test-target blocks are frozen before outcomes. Uniform-target cases sample the test frame uniformly. `MDB_S4_RHO060_DESIGN_SHIFT` samples the test frame without replacement proportional to its frozen pretreatment `r(U)` target-design ratio. No A, Y, support diagnostic or method result enters target selection.

Refusals are abstentions and are reported separately from selective coverage. Monte Carlo uncertainty is clustered by replication/mine. Five targets per mine/replication are used to support mine, dose, spatial-stratum and fifth-percentile local summaries without claiming exact location-conditional coverage.

The primary inference route is scalable N2/N3 with `m=64`. A separate exact-size6 registry supplies one local G2/G3 audit target per case×replication×mine when available; unavailability is recorded, never replaced by a cross-mine orbit.

## Frozen support gates
No benchmark retuning: ESS>=3; max normalized weight<=.50; graph-safe count>=20; applicable N3 lower bound finite and >0.

## Candidate domain and efficiency
Green fractional-cover change is represented as a proportion, so the numerical candidate domain is the physical interval `[-1,1]`. No benchmark output is clipped into this range.

Full inversion is prospectively limited to S4 rho={0,.2,.4,.6,.8}, reps 1 and 20, and primary target rank 1 in each mine: 50 full-inversion queries per candidate-set method. Width is interpreted only when selective coverage differs by <=.03; unmatched groups remain unmatched.

## Endpoint audit
For `MDB_ST1_MIXED_ATOMS`, run exact localized endpoint queries at doses 0 and 1 in addition to the interior reference. For `MDB_ST1_INTERIOR_ONLY`, issue endpoint 0 and 1 negative-control queries and require refusal because those atoms are not audited. Never jitter or clip endpoints.

## Required outputs
Marginal/selective coverage and MC uncertainty; absolute calibration error; mine/dose/local and fifth-percentile summaries; refusal rate/codes; false-support; ESS/max-weight/graph-safe counts; N2/N3 certificate terms; finite-domain width and WIS; hard-dose RMSE; theorem eligibility; runtime/failure; topology/hull inflation; Moran's I/semivariogram as diagnostics only.

S9 reports observed-product and latent truth separately. S10 reruns complete 90/180 analyses. ST2 remains structural/cross-sectional unless a new temporal method is frozen. ST3 is a causal-identification negative control. LOMO uses frozen Stage3A whole-mine rotations and never fabricates exact cross-mine orbits.
