# MineDoseBench v1.1 — frozen mathematical generation protocol

## Scientific role
MineDoseBench is a semi-synthetic Earth-observation benchmark. It freezes real NSW mine geometry, accepted 90 m block support/graph, real pretreatment DEA trajectories/quality fields, and externally extracted real climate/soil/terrain context. It injects treatment, potential-outcome truth, spatial residuals and optional EO measurement error. It does **not** reinterpret the public NSW snapshot rehabilitation fraction as an annual causal treatment.

The primary target remains the localized stochastic potential outcome. For target dose `a0`, draw `A* ~ q_h(.|a0,U)` and evaluate `Y(A*)`. Endpoint 0/1 requests are exact atoms only in cases whose injected mixed treatment law audits those atoms.

Green fractional-cover change is represented as a **proportion**, so the physically admissible numerical response domain is `[-1,1]`. MineDoseBench candidate inversion therefore uses `[-1,1]`, not the earlier controlled-simulation `[-8,8]` domain. Values are never clipped to force compliance. Stage5A audits the materialized validation truth, and Stage5B must apply the same physical-range gate before evaluating any method for each regenerated case-replication.

## Real substrate and temporal order
- Frozen Stage2A 90 m support, EPSG:9473, five mines, queen graph.
- Pretreatment satellite state: accepted Stage2B 2023 and 2024 PV/quality fields.
- Synthetic treatment occurs after the 2023/2024 pretreatment window; the injected response is a post-2024 future fractional-cover change.
- Real 2025 PV is excluded from all benchmark DGPs, splits and tuning and remains PREP-only diagnostic data.
- `mapped_rehabilitation_fraction` is static support/context only and is forbidden in the injected treatment law, potential-outcome function and method nuisance predictors.
- Real climate, soil and terrain context must be extracted from documented external products by **area-weighted intersection with the frozen block polygons**. Centroid-only extraction is not accepted for the official freeze.

## Spatial information split
For each mine, among pretreatment-eligible blocks, the largest true-graph connected component is the primary benchmark component. Smaller components remain as `secondary_support_stratum`; they are not silently deleted and do not enter the primary injected DGP.

The primary component is projected onto a deterministic first spatial principal axis. Role bands are frozen by axis rank: nuisance training <=.30; buffer (.30,.36]; support audit (.36,.50]; buffer (.50,.56]; calibration (.56,.80]; test target >.80. Forbidden direct graph edges are nuisance–support, nuisance–calibration, nuisance–test, support–calibration and support–test. Calibration–test adjacency is allowed for local exact audit blocks.

A separate frozen Stage3A leave-one-mine-out registry is retained for scalable transfer evaluation. Exact local orbits are never fabricated across disconnected mines.

## Mixed treatment truth
Let standardized real pretreatment covariates be `x_pv`, `x_rain`, `x_soc`, `x_slope`, `x_awc` and spatial-axis coordinate `x_coord`. For observed-confounding cases,

`eta = -0.10 + c*(0.80*x_pv - 0.50*x_rain + 0.35*x_soc + 0.20*x_coord)`.

Endpoint logits follow the accepted Stage3B structural pattern. The interior component is Beta with mean `expit(eta)` and overlap-dependent concentration `kappa in {6,10,24,45}`. The observational law is a genuine mixed measure: endpoint atom 0, endpoint atom 1 and continuous interior. Interior-only and strong-atom stresses are separately registered. The benchmark stores exact oracle `g(A|X)` parameters. `q_h/g` remains distinct from the N2/N3 target-design ratio.

## Potential-outcome truth
Let `PV_t` be green fractional cover represented as a proportion and `trend = PV_2024-PV_2023`. The nonlinear baseline is

`mu0 = 0.65*trend + 0.032*x_pv - 0.020*x_rain + 0.012*(x_soc^2-1) + 0.015*sin(pi*x_coord) - 0.010*x_slope + 0.008*x_awc`.

The conditional mean curve is

`mu(a) = mu0 + 0.055*a - 0.035*a^2 + 0.025*sin(2*pi*a) + 0.020*a*x_pv`.

The linear reduction case uses `mu0 = 0.60*trend + 0.030*x_pv - 0.018*x_rain + 0.010*x_soc` and `mu(a)=mu0+0.050*a`.

One coherent spatial residual is shared across the unit's complete potential-outcome curve.

## Spatial residual and C2 negative control
For true queen adjacency `A`, degree matrix `D`, and normalized adjacency `S=D^{-1/2}AD^{-1/2}`,

`E ~ N(0,Q^{-1}),  Q=(I-rho*S)/sigma^2`, with `sigma=.04`.

No dense precision inverse is formed. `Q^{-1/2}z` is evaluated by a degree-64 Chebyshev approximation to `(1-rho*x)^(-1/2)` on `[-1,1]`, with scalar approximation error <=`1e-12`. The non-Gaussian validation case uses `sign(E)|E|^1.5` and is not relabelled Gaussian-N2 theorem evidence.

The ST3 hidden C2 confounder is **spatial**, generated from a separate proper GMRF on the true graph with rho=.60 and standardized to unit scale. It enters both treatment and outcome only in ST3 and is never an allowed method predictor. The required conclusion remains that conformal calibration cannot repair causal identification failure.

## Target-design shift
The nonidentity design-shift case uses a finite-frame exponential tilt on pretreatment `D=x_pv`:

`r_i = exp(delta D_i) / mean_obs exp(delta D)`, delta=.75.

Unlike the earlier candidate, this ratio actually defines the target population: test-target blocks for `MDB_S4_RHO060_DESIGN_SHIFT` are sampled without replacement with probability proportional to `r_i`. Other cases use the frozen uniform test frame. This is separate from treatment/intervention `q_h/g`.

## Target and common-random-number registries
Before any outcome is generated or any method is evaluated, Stage5A freezes:
- five distinct primary target blocks per case × replication × mine;
- one exact-size6 audit target per case × replication × mine when locally available;
- all 20 replications;
- eight random streams.

Random streams are unique across distinct `(common_random_group, replication, stream)` keys, but intentionally shared by cases in the same common-random group. This pairs S4 rho/residual-law factors, S6 graph factors, S9 measurement-error factors, ST1 endpoint factors and S10 support factors without accidental seed reuse elsewhere.

## Measurement error
Latent and observed satellite-product outcomes are separate. With pretreatment UE proxy and fixed unit innovation `Z`, `sigma_meas=.012+.0015*clip(UE_2024,0,50)`. Registered S9 mechanisms are random, treatment-correlated, substrate-specific and combined. Product coverage is never relabelled latent ecological-state coverage.

## MAUP
The 180 m support is an anchored 2×2 rerun from the **complete** frozen 90 m child inventory. An 180 m benchmark cell is eligible only if every available 90 m child is baseline-eligible. Covariates are area-weight aggregated, a new queen graph is constructed, connected components and spatial splits are rebuilt, and injected treatment/outcome truth is regenerated at the new support. No scalar post-hoc MAUP correction is used.

## Scenario matrix
The release retains the 27-case S1–S10/ST1–ST3 validation architecture, including five S4 rho levels, non-Gaussian and target-design-shift S4 cases, poor overlap, graph misspecification, benign exchangeability, severe support, five S9 measurement-error variants, 90/180 m support, mixed/interior-only endpoints, temporal dependence and hidden spatial C2 confounding.

## Registered Stage5B evaluation contract
The **primary MineDoseBench comparison follows the definitive proposal's reduced set**: M2 dose-only, M3 spatial-only, M4 naive product, M5 conservative graph-safe fallback and M6 GeoDose-CP. M1 is not a primary MineDoseBench method. H1/H4 remain controlled-simulation heuristic ablations.

Primary MineDoseBench inference uses the scalable N2/N3 route with `m=64`. Exact G2/G3 is a registered local audit on frozen size-6 targets. Stage3F operational thresholds remain ESS>=3, max normalized weight<=.50, graph-safe calibration count>=20 and applicable N3 lower bound finite and >0; MineDoseBench cannot retune them.

Endpoint auditing is mandatory: mixed-atom ST1 evaluates exact target doses 0 and 1; interior-only ST1 issues endpoint 0/1 negative-control queries that must refuse. Candidate domain is `[-1,1]` on the fractional-cover-change scale.

Stage5A evaluates no method performance. Benchmark bytes, real context, substrate, splits, scenarios, seed/target registries and evaluation contract are frozen first.
