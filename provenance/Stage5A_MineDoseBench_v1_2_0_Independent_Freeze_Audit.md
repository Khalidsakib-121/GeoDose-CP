# GeoDose-CP Stage5A MineDoseBench v1.2.0 — Independent Freeze Audit

**Audit date:** 2026-08-10  
**Stage:** Stage5A — MineDoseBench revised freeze candidate  
**Decision:** **PASS / ACCEPT — FREEZE Stage5A MineDoseBench v1.2.0**

## 1. Audited artifacts

### Windows-generated authoritative release
- File: `GeoDose_MineDoseBench_v1_2_0_RELEASE_CANDIDATE.zip`
- Independently recomputed SHA-256:  
  `1c5e48930e6bbba5d71bc7bfd558c142f65dc726ed554fe3112cb2450a5f5ef5`
- Size: 26,569,611 bytes
- Release-manifest tracked files: 40
- Release-manifest aggregate:  
  `513f1140d3dbcc0a2910a2227255d6de9cea4b60414aef5b73cd998d8e619ec3`

### User-uploaded wrapper containing MINEDOSEBENCH_FREEZE_OUTPUTS
- Independently recomputed SHA-256:  
  `22676094a606692d48cb350ef1fd5eeec68c41681792f9629369b9a838496c57`
- The authoritative nested release ZIP is byte-identical to the hash printed by the Windows runner.

### Accepted source package
- File: `GeoDose_Stage5A_MineDoseBench_Generator_v1_2_0_FREEZE_CANDIDATE.zip`
- Independently recomputed SHA-256:  
  `e0f9dbbffd42ac7cb298cfed57dda548b67906329b97bf61823e09ff29aa2e12`
- Package immutable files: 42
- Package-manifest aggregate:  
  `48dfb15c7da08d6fb20263b38499aae552fa6904a57729fcb67c2f6a96523496`

## 2. Archive and source integrity

Independent checks performed:

- All 40 release-manifest-listed files exist in the authoritative release ZIP.
- All 40 independently recompute to their declared byte sizes and SHA-256 values.
- The release-manifest aggregate independently recomputes exactly.
- The scientific output manifest contains 37 tracked scientific files; all 37 independently match declared sizes and SHA-256 values.
- The source package contains all 42 immutable package-manifest files; all 42 independently match their declared sizes and SHA-256 values.
- The output `minedosebench_source_inventory.json` contains 33 source/config/document files.
- All 33 source-inventory hashes and sizes independently match the accepted v1.2.0 source package.
- The embedded prior external-context independent audit is byte-identical to the accepted audit SHA-256:
  `1e4a7856cafdbd1cf6fa6dd7a08db79370007b6068a1eb246e028474f87ad9cb`.

No archive-integrity or provenance mismatch was found.

## 3. Frozen external-context authority

The Stage5A release carries forward the previously accepted prospective context decision exactly:

- Original required context blocks: **23,710**
- Frozen context-complete blocks: **23,618**
- Frozen support exclusions: **92**
- Valid-area threshold: **0.99**
- Imputation: **none**
- Threshold relaxation: **none**

Independent ID-set reconciliation confirms:

- retained IDs are unique;
- excluded IDs are unique;
- retained and excluded sets are disjoint;
- retained ∪ excluded equals the exact original 23,710-block requirement;
- `minedosebench_real_context_covariates.csv` contains exactly the 23,618 retained block IDs;
- no retained context value is NaN or infinite.

## 4. Spatial substrate and MAUP audit

### 90 m primary substrate
- Rows: **27,042**
- CRS: **EPSG:9473**
- Baseline eligible: **23,618**
- Eligible IDs equal the frozen context-complete registry exactly.
- Frozen support-exclusion flag identifies exactly the 92 excluded IDs.
- No excluded block has an active benchmark role.
- Role counts:
  - nuisance training: 5,599
  - support audit: 2,615
  - calibration: 4,479
  - test target: 3,732
  - buffer excluded: 2,238
  - secondary support stratum: 4,955
  - baseline ineligible: 3,424

### Spatial leakage guard
The true 90 m queen graph contains **102,893** undirected edges. Independent role-edge checking found **0 forbidden adjacency violations** for the frozen nuisance/support/calibration/test separation rule.

### Graph integrity
90 m and 180 m true graphs have:
- 0 self loops;
- 0 duplicate directed records;
- 0 edges to unknown nodes;
- 0 cross-mine edges.

### MAUP / 180 m support
- 180 m cells: **7,295**
- Quality-consistent eligible cells: **5,788**
- Eligibility independently reproduces the frozen rule: every available 90 m child must be baseline eligible.
- True 180 m graph edges: **26,924**
- Exact size-6 target blocks:
  - 90 m: **212**
  - 180 m: **88**

Each independently checked exact size-6 fixture contains:
- one `test_target`;
- five distinct `calibration` blocks;
- all six blocks from the same mine;
- exact block size = 6.

## 5. Scenarios, seeds and pre-outcome target freeze

### Scenario registry
- Unique cases: **27**
- S1–S10 represented.
- S4 rho sequence represented.
- S9 measurement-error variants represented.
- S10 90 m and 180 m variants represented.
- ST1 mixed/interior endpoint tests represented.
- ST2 temporal stress represented.
- ST3 hidden-C2 negative control represented.

### Replications and seeds
- Registered case-replication rows: **540 = 27 × 20**
- Eight seed streams independently checked.
- Within each intentional `(common_random_group, replication)` key, each stream has exactly one shared seed.
- Distinct CRN keys have distinct seed values.
- After intentional CRN duplicates are collapsed, all seed values are globally unique across the eight streams.
- No accidental cross-stream collision was found.

### Primary target registry
- Rows: **13,500 = 27 × 20 × 5 mines × 5 targets**
- Exactly five targets for every case × replication × mine.
- Target ranks are exactly 1–5.
- Every 90 m target belongs to the frozen 90 m `test_target` frame.
- The S10-180 m targets belong to the frozen eligible 180 m `test_target` frame.
- No frozen 90 m support-exclusion ID is used as a target.
- Design-shift target weights are genuinely nonconstant; other registered target modes are uniform as intended.

### Exact-audit registry
- Rows: **2,700 = 27 × 20 × 5 mines**
- All 2,700 exact-audit targets are available.
- 90 m exact-audit target IDs are members of the frozen 90 m exact-size6 registry.
- S10-180 m exact-audit target IDs are members of the frozen 180 m exact-size6 registry.

## 6. Validation materialization and truth separation

Stage5A materializes only deterministic **replication 1** for structural audit, while all 20-replication seeds/targets are frozen prospectively for Stage5B. This is consistent with the registered Stage5A boundary.

Validation outputs:
- Public units: **41,066 rows**
- Oracle truth: **41,066 rows**
- Validation target truth: **675 = 27 × 25**
- Hard-dose truth: **4,725 = 27 × 25 × 7 doses**

The public and oracle tables align row-for-row on case/scenario/replication/block/mine identity.

Information separation independently checked:
- public table has no `g`, oracle-truth, hidden-confounder, residual-truth, or target-design-ratio columns;
- `Y_observed_at_A = Y_true_at_A + measurement_error_at_A` to floating-point precision;
- no frozen support-exclusion ID appears in public or oracle validation rows;
- all target outcome columns in `minedosebench_validation_targets_rep1.csv` are explicitly marked `post_generation_truth_only=True`.

## 7. Treatment, outcome and stress-fixture checks

### Physical outcome domain
All audited latent/observed/hard-dose outcomes remain inside the frozen `[-1,1]` fractional-cover-change domain.

Observed extrema:
- factual latent: approximately **[-0.4544, 0.6420]**
- factual observed: approximately **[-0.4544, 0.6476]**
- target latent/observed: approximately **[-0.2418, 0.5330]**
- hard-dose truth: approximately **[-0.2447, 0.5330]**

### Hard-dose grid
Exactly seven doses are present for every validation target:
`0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00`.

### Endpoint structure
- `MDB_ST1_MIXED_ATOMS` contains genuine A=0 and A=1 atoms.
- `MDB_ST1_INTERIOR_ONLY` contains no endpoint observations and remains strictly inside (0,1).

### Measurement-error isolation
- Non-S9 cases have exactly zero measurement error.
- All five S9 cases contain the intended nonzero measurement-error perturbation.

### Target-design shift
The design-shift oracle frame contains **18,663 rows**. Within each mine, the frozen target-design ratio has mean 1 to floating-point precision and is nonconstant.

### Temporal stress
The temporal fixture contains only `MDB_ST2_TEMPORAL`, replication 1, with exactly three relative years: 1, 2, 3.

### Hidden-C2 negative control
The archived hidden-confounder diagnostic is finite and has approximately unit standard deviation with positive edge association, consistent with a spatial rather than IID hidden confounder.

## 8. Leakage and scope audit

Independent source inspection confirms:
- the DGP generator has no network calls;
- `mapped_rehabilitation_fraction` does not occur in the DGP generator;
- `pv_median_2025` does not occur in the DGP generator;
- Stage2 2025 fields are loaded only as part of the frozen upstream table and are removed before the benchmark substrate; the 2023/2024 eligibility rule is what defines the benchmark frame;
- the 26 registered method-nuisance predictors are all labeled pretreatment;
- no future/oracle/hidden/support-role variable appears in the allowed-predictor registry;
- no M2–M6 performance result is generated at Stage5A.

The release claim boundary correctly keeps:
- M2–M6 evaluation for Stage5B;
- exact G2/G3 only for registered size-6 audit targets;
- N2/N3 m=64 as the primary scalable route;
- ST2 as structural temporal stress only;
- ST3 as a causal-identification negative control.

## 9. Windows-run verification

The supplied PowerShell run records the exact frozen runtime:
- Python 3.10.0
- NumPy 2.2.6
- pandas 2.3.3
- SciPy 1.15.3
- GeoPandas 1.1.4
- Shapely 2.1.2
- Pyogrio 0.13.0
- PyYAML 6.0.3

The run reports:
- **80/80** unit/authority tests passed;
- all **27/27** validation cases generated;
- **277/277** post-generation verification checks passed;
- zero Stage5A method-performance evaluation;
- release ZIP SHA-256 exactly matching the independently recomputed hash.

## 10. Non-blocking qualifications

1. Only validation replication 1 is materialized at Stage5A. This is intentional; the complete 20-replication seed and target registries are frozen for Stage5B.
2. The benchmark release ZIP contains the generated benchmark and source inventory, while the executable generator source is preserved in the separate accepted v1.2.0 source package. Preserve both.
3. The previously documented TERN bulk-density metadata-label inconsistency remains a provenance note inherited from the accepted external-context audit; it does not alter the numerical Stage5A benchmark construction.

No blocking scientific, reproducibility, leakage, support, seed, target, graph, or archive defect was identified.

## 11. Independent decision

**PASS / ACCEPT.**

### Freeze authorization

`GeoDose_MineDoseBench_v1_2_0_RELEASE_CANDIDATE.zip` is accepted as the **frozen Stage5A MineDoseBench v1.2.0 benchmark authority**.

Do not reopen Stage5A merely to improve later M2–M6 performance. Reopen only if:
1. an upstream frozen authority changes; or
2. Stage5B demonstrates a direct internal contradiction in the frozen Stage5A benchmark truth/registry.

### Next authorized stage

**Stage5B — preregistered MineDoseBench M2–M6 production evaluation.**

Stage5B must consume this benchmark without changing:
- the 23,618/92 substrate decision;
- scenario registry;
- seed registry;
- target registries;
- spatial splits;
- Stage3F thresholds;
- candidate domain;
- primary M2–M6 method set;
- registered evaluation contract.

**Final audit outcome: STAGE5A v1.2.0 ACCEPTED AND FROZEN.**
