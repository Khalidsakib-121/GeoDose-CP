# GeoDose-CP Stage 3 Execution Blueprint

## Purpose

Stage 3 provides the paper's ground-truth evidence through controlled simulation and MineDoseBench. It does not reinterpret the NSW mapped rehabilitation fraction as an annual causal treatment.

## Frozen inputs

- Stage 2A: 27,042 blocks, 102,893 within-mine queen edges, five LOMO folds, fixed component IDs.
- Stage 2B: 81,126 block-year records for 2023–2025, 15 verified mine-year checkpoints, DEA PV/UE/quality fields.
- Five mines for MineDoseBench; three selected mines are reserved for the later real EO demonstration.

## Execution phases

### 3A — Registry and input adapter

Create a read-only adapter for the frozen Stage 2A/2B data and write:

- `stage3_registry.yaml`
- `seed_registry.csv`
- `data_contract.json`
- `input_manifest.json`
- `stage3_prescreen_report.md`

No production result may be generated before this registry is frozen.

### 3B — Controlled generator

Generate:

- pretreatment covariates;
- mixed treatment with genuine endpoint atoms and continuous interior density;
- known target intervention law;
- known potential-outcome function;
- graph-correlated residuals;
- optional non-Gaussian transformed-GMRF residuals;
- optional treatment-correlated EO measurement error.

### 3C — Methods M1–M6

- M1: standard split conformal.
- M2: dose-only weighted conformal.
- M3: spatial-only generalized conformal.
- M4: naive dose × spatial product.
- M5: conservative graph-safe weighted conformal.
- M6: full GeoDose-CP.

Use the same fitted nuisance models for all six calibration wrappers within a split.

### 3D — Exact theorem unit tests

Use small graph-local blocks only.

- Primary orbit block size: 6.
- Stress block size: 8.
- Enumerate distinct payload permutations.
- Preserve endpoint atoms.
- Include Jacobian and graph-boundary terms.
- Compare empirical orbit frequencies with the G2 target law.
- Verify G3 candidate inversion coverage.
- Test duplicate payloads and randomized ties.

### 3E — Scalable branch

Use sparse GMRF/Vecchia calculations for large simulations and MineDoseBench.

Report separately:

- sparse approximation discrepancy;
- structural misspecification;
- nuisance-estimation error;
- numerical error;
- certificate-failure probability.

Calibrate internally at `alpha_internal = alpha - certified_deficit`; refuse when the deficit is at least alpha or any mandatory certificate fails.

### 3F — Pilot and freeze

Run 20 repetitions per scenario to detect implementation errors and estimate runtime. Pilot results may change numerical settings but may not be used as scientific evidence. Freeze the production registry afterward.

### 3G — Production controlled experiment

Primary nominal coverage: 90%.

Recommended production repetitions:

- S4 hero experiment: 500.
- S1–S3, S5–S8: 300 each.
- S9–S10: 200 each.
- XGBoost confirmation: S1, S4, S5 and S8.
- Random-forest reference: all S1–S10.

Report Monte Carlo standard errors.

### 3H — MineDoseBench

Use real geometry, graph, mine identities, DEA history and quality distributions. Inject synthetic causal treatment and outcomes.

Pretreatment features may use:

- 2023 and 2024 PV;
- 2023–2024 PV trend;
- 2023–2024 UE and observation quality;
- coordinates;
- graph degree and component size;
- mine identity;
- mapped rehabilitation fraction only as a structural predictor, never as authentic annual causal treatment.

Assign synthetic treatment after 2024 and generate known 2025 potential outcomes.

Use whole-mine roles in each LOMO fold:

- held-out mine: test target;
- one mine: support audit;
- one mine: calibration;
- two mines: nuisance training.

### 3I — Required outputs

- `controlled_replication_results.parquet`
- `controlled_summary.csv`
- `hero_S4_coverage_width.csv`
- `method_failure_log.csv`
- `refusal_diagnostics.csv`
- `minedosebench_v1.parquet`
- `minedosebench_truth.parquet`
- `minedosebench_splits.csv`
- `minedosebench_results.parquet`
- `stage3_manifest.json`
- `STAGE3_VERIFICATION.json`

## Primary fixed settings

- `alpha = 0.10`; sensitivity `alpha = 0.05`.
- Dose grid: `0, 0.10, 0.25, 0.50, 0.75, 0.90, 1`.
- Interior bandwidth: `h = 0.10`; sensitivities `0.05` and `0.20`.
- Do not jitter, clip or smooth endpoint treatment atoms.
- Use log-scale weights.
- Report effective sample size and maximum normalized weight.
- Preserve raw disconnected prediction-set components.
- Never form a dense inverse of the precision matrix.

## Preliminary refusal rules to freeze after pilot

A query refuses when any mandatory mathematical gate fails, including:

- unsupported target dose;
- target endpoint not audited as a genuine atom;
- effective calibration size below the frozen threshold;
- excessive maximum normalized weight;
- non-positive-definite spatial precision;
- empty/unbounded nuisance confidence set;
- certified discrepancy greater than or equal to alpha;
- insufficient proposal/orbit support;
- non-finite interval;
- insufficient EO quality for the claimed observed-product target.

## Main evaluation

- marginal coverage;
- dose-bin coverage;
- block/mine coverage;
- fifth-percentile local coverage;
- interval width at matched coverage;
- weighted interval score;
- dose-response RMSE;
- effective calibration size;
- refusal and false-support rates;
- observed-outcome versus latent-outcome coverage in S9;
- runtime and computational failure rate.

## Hard interpretation boundary

Controlled simulation and MineDoseBench establish known-ground-truth causal coverage. The later NSW application is a product-aware EO uncertainty/refusal demonstration, not evidence that the snapshot mapped rehabilitation fraction caused observed PV changes.
