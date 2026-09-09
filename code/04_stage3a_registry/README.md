# GeoDose-CP Stage 3A Registry and Generator v1.3.1 FREEZE

## Decision

Use **v1.3.1 FREEZE**. It supersedes v1.0–v1.3 and is the final Stage 3A architecture package.

The v1.3.1 FREEZE audit closes the final registry/reproducibility gaps before any Stage 3A run:

1. Adds a dedicated `target_draw_seed` stream required by the Stage 3B target-draw contract.
2. Freezes controlled factors, nuisance-model classes, compact ablations, evaluation metrics, and mandatory R1–R4 reduction checks.
3. Adds an explicit target-population contract and separates pretreatment causal eligibility from post-assignment EO-quality refusal.
4. Records the actual Python/software environment and verifies deterministic replay of the complete generator preview.
5. Explicitly records climate, terrain, soil, and commodity context as deferred in the minimum route so these covariates cannot be silently assumed or used.


It retains the v1.1 corrections:

- proper GMRF draws are not realization-centered or standardized;
- the localized stochastic potential outcome remains the primary estimand;
- hard-dose truth remains secondary;
- the mapped rehabilitation fraction remains support-stratification metadata only.

It also includes:

- numerical mixed-treatment normalization including endpoint atoms;
- C1–C5/F1 assumption registry;
- U1–U18 proof/algebra test registry;
- explicit localized-target truth;
- stronger Stage 2A/2B hash and contract checks.

## Purpose

This package performs the first Stage 3 gate only:

- verifies the frozen Stage 2A and final Stage 2B inputs;
- freezes S1–S10, M1–M6, targets, assumptions, feature timing, mine roles, refusals and seeds;
- creates a deterministic 625-unit mixed-treatment/proper-GMRF generator preview;
- creates known hard-dose truth and an analytic conditional-mean functional for the primary localized stochastic target;
- verifies genuine endpoint atoms and mixed-law normalization;
- produces a formal Stage 3A manifest and verification record.

It does **not** implement M1–M6 and does not run production simulations.

## Frozen inputs included

The package already includes the accepted Stage 2A files and the accepted final Stage 2B archive with SHA-256:

`329a21f3b94d88f8f0b0b3327dbe250eb1df0911f73a6309c6c80c8417bd73a8`

Do not replace or edit them.

## Run

Use the Python 3.10 environment already created for Stage 2B. The batch file first looks for:

`D:\GeoDose_Stage2B_Local_STAC_v1_0\.venv\Scripts\python.exe`

Double-click `RUN_STAGE3A.bat`, or run in PowerShell:

```powershell
$py = "D:\GeoDose_Stage2B_Local_STAC_v1_0\.venv\Scripts\python.exe"
& $py .\stage3a_prepare.py --input inputs --output outputs_stage3a --overwrite
if ($LASTEXITCODE -ne 0) { throw "Stage 3A preparation failed" }
& $py .\verify_stage3a.py --input inputs --output outputs_stage3a
if ($LASTEXITCODE -ne 0) { throw "Stage 3A verification failed" }
```

No internet access is required. PyArrow is not required by the Stage 3A scripts. The runner verifies the frozen Python 3.10 environment recorded in `requirements_frozen_py310.txt`; it does not silently accept a different numerical stack.

## Required success message

`STAGE 3A VERIFIED COMPLETE`

The verification must also print:

`Primary target: localized stochastic potential outcome`

## Upload after completion

`GeoDose_Stage3A_OUTPUTS.zip`

## Interpretation boundary

The mapped rehabilitation implementation fraction remains a noncausal snapshot exposure. Stage 3A validates the known-ground-truth architecture only. It does not create annual NSW treatment and does not establish M1–M6 coverage.


## v1.2 interpretation gate

`stage3a_localized_target_truth.csv.gz` contains an analytic mean functional only. It is useful for generator checks and dose-response RMSE, but it is not sufficient for finite-sample conformal coverage. Stage 3B must create realized localized intervention draws and observed/latent target outcomes.


## v1.3.1 FREEZE acceptance boundary

Stage 3A v1.3.1 FREEZE freezes architecture and reproducibility contracts. Numerical support/refusal thresholds and full DGP coefficients remain provisional until the prespecified Stage 3F pilot. M1–M6, RL3, exact G2/G3 inversion, and scalable N2/N3 certification remain unimplemented and must not be claimed from Stage 3A.


## Freeze acceptance rule

After `STAGE 3A VERIFIED COMPLETE`, Stage 3A is frozen. It is reopened only if the package fails to execute, a frozen Stage 1/2 input checksum changes, or the proposal/mathematical specification is formally revised. Later implementation work belongs to Stage 3B–3E and is not a reason to revise this Stage 3A registry.
