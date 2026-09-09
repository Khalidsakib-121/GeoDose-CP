# GeoDose-CP Stage5B MineDoseBench Production Evaluation v1.0.0
## Mathematical and algorithm alignment note

### Frozen authority
Stage5B consumes the accepted Stage5A MineDoseBench v1.2.0 release byte-for-byte. It does not change the 23,618/92 context partition, 27 cases, 20 repetitions, spatial roles, graphs, seed registry, primary targets, exact-audit targets, candidate domain, or Stage3F operational thresholds.

### Primary comparison
MineDoseBench uses exactly M2–M6 as frozen in the Stage5A evaluation contract:
- M2 dose-only weighted conformal prediction;
- M3 response-only spatial graph/residual conformal comparator with treatment fixed;
- M4 deliberately naive one-normalization dose-ratio x M3 target-source spatial marginal comparator;
- M5 deterministic graph-safe/block weighted fallback;
- M6 GeoDose-CP joint treatment-response graph-local orbit with sparse/localized m=64 spatial reference.
M1 remains controlled-simulation/real-demonstration context and is not reintroduced into the primary MineDoseBench comparison.

### Information separation
Outcome, mixed-treatment and estimated spatial nuisance objects are fit only on `nuisance_training`. Support-audit outcomes are used only for post-fit spatial diagnostics. Calibration outcomes are used only after nuisance models are frozen. Test-target truth is used only for benchmark scoring. Buffer and secondary-support outcomes are never used to improve a primary query.

### Residual transformation and Jacobian
The frozen Stage5B residual scale is the identity `s_i(a)=1`; therefore the mandatory inverse outcome Jacobian is explicitly represented in M6 as `log(1)=0`. It is not silently dropped. The transformed non-Gaussian S4 stress includes the corresponding residual transformation Jacobian in M6 orbit density evaluation.

### M3 and M4
M3 holds A and fixed slot information constant and moves the response/residual payload. M4 consumes only the M3 target-source spatial marginal and the dose/intervention log ratio, followed by exactly one normalization. M4 is never called theorem-backed.

### M6
M6 moves the joint treatment-response payload. At the target slot it uses the normalized intervention measure; at non-target slots it uses the observational treatment likelihood. Residual graph weights use the frozen sparse/localized GMRF reference. All orbit calculations use log weights and log-sum-exp normalization.

### Scalable route and N2/N3 claim discipline
The full MineDoseBench operational route uses the frozen m=64 sparse/localized graph-residual reference, with m=32 as a sensitivity reference. The resulting six-dimensional m64-vs-m32 KL/Pinsker quantity is explicitly labelled an observable sparse-localization diagnostic. It is **not** relabelled a finite-sample whole-graph N2 nuisance certificate. Estimated nuisance fits are not claimed to have a finite-sample theorem certificate.

A separate frozen exact-size6 audit compares full-graph structural-oracle G2/G3 p-values with sparse-m64 p-values for M3/M4/M6. This is the primary structural approximation audit. It does not upgrade estimated nuisance fits to oracle status.
For this separate structural audit, the sparse m=64 context is built from the **full frozen calibration-role frame** on which the Stage5A exact-size6 registry was defined. Case-specific primary calibration subsampling (notably `MDB_S8_SMALL_CAL`) remains in force for operational M2–M6 evaluation but does not redefine the frozen structural-audit fixture.

### Frozen Stage3F operational gates
- M2/M4/M5/M6: ESS >= 3 and maximum normalized treatment/support weight <= 0.50.
- M3/M4/M5/M6: graph-safe calibration count >= 20.
- M6: applicable operational N3 lower-bound diagnostic must be finite and >0.
- Endpoint atoms are queried only when audited; otherwise refuse.
Refusals are abstentions and are never counted as noncoverage.

### Candidate inversion
The physical candidate domain is exactly [-1,1]. For the preregistered S4 width subset (rho 0,.2,.4,.6,.8; repetitions 1 and 20; target rank1 per mine), M3/M4/M6 candidate p-values are evaluated on a deterministic 4096-point grid and refined at transitions with 40 deterministic bisection steps. The raw set may be disconnected. The raw component union and hull are both retained, and hull inflation is reported.

### Efficiency
Width is interpreted only when selective coverage is matched within 0.03. For M2/M5, physical-domain intersection is used only for finite-domain width comparison; unbounded WIS is not silently capped. M3/M4/M6 width comes from finite-domain numerical inversion.

### Ground-truth and stress analyses
- S9: observed-product and latent-truth coverage are reported separately.
- S10: 90m and 180m runs are complete independent reruns using their frozen supports.
- ST1: mixed endpoint atoms 0/1 are audited; interior-only endpoints are mandatory refusal negative controls.
- ST2: structural/cross-sectional temporal stress only; no new temporal conformal theorem is claimed.
- ST3: hidden spatial C2-confounding negative control; conformal calibration is not interpreted as repairing causal identification.
- LOMO: frozen whole-mine rotations; M3/M4/M6 refuse rather than fabricate a cross-mine local orbit.

### Statistics
The primary coverage uncertainty uses a deterministic cluster bootstrap over replication x mine clusters (2,000 bootstrap draws, seed 20260811). Coverage, refusal, local coverage, fifth-percentile local coverage, WIS, dose-response RMSE, support diagnostics, exact-vs-sparse discrepancy, MAUP, measurement error and runtime are exported. The S4 hero figure shows coverage uncertainty and width uncertainty.

### Development contamination guard
Development smoke executions are permitted only to validate code paths, identities, runtime and algebra. No support threshold, scenario, target, predictor, hyperparameter or reporting rule is selected based on observed Stage5B performance. The package freezes all such choices before the user's authoritative 540-task production run.
