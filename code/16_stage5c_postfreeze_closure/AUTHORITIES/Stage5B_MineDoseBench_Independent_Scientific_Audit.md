# GeoDose-CP Stage5B MineDoseBench Production — Independent Scientific Audit

**Audit date:** 2026-08-11  
**Input:** `GeoDose_Stage5B_MineDoseBench_PRODUCTION_RESULTS.zip`  
**Independent ZIP SHA-256:** `a0060e42980d4816ff81d1449c17913b2d8ea79d9d61115249635077179d9fac`

## Decision

**PASS / ACCEPT Stage5B as a valid frozen production result, with scientific qualifications that must be stated explicitly in the manuscript.**

The archive is internally consistent, all declared output hashes reproduce, the production contains the expected number of rows, there are no computational failures in the 540 task runs, and the numerical-closure patch is demonstrably limited to machine-roundoff canonicalization without changing coverage decisions.

This audit does **not** support a claim that GeoDose-CP universally dominates every comparator in interval efficiency or cross-mine transfer. The strongest supported claims are robust supported-regime coverage, strong local-coverage behavior, sensitivity to treatment-design/spatial shift, principled refusal, and close aggregate exact-vs-sparse behavior.

## 1. Archive integrity

- Release-manifest files: **44 / 44 verified**
- Output-manifest files: **42 / 42 verified**
- Scientific-output-manifest files: **39 / 39 verified**
- Release aggregate independently reproduced:  
  `6bc0bcbae96256b04d53fb3b8cd029b6237f046f213543251becf0af5b2e3621`
- Output aggregate independently reproduced:  
  `3ba8ab555a44e6992fc48363e99f3e32a563b9715aeedaa5cb2e56a4610a8862`
- Scientific-output aggregate independently reproduced:  
  `f0349a16cdf1b88539cb66c7fa19890477335586e5580a7e0b00621a74a086be`

## 2. Production completeness

The final report and raw tables contain the complete registered Stage5B evaluation:

- case × replication tasks: **540**
- query rows: **187,500**
- exact-audit rows: **16,200**
- full-inversion rows: **150**
- endpoint-audit rows: **4,000**
- dose-response prediction rows: **227,500**
- nuisance-fit audit rows: **1,300**
- LOMO rows: **1,875**
- spatial-diagnostic rows: **7,800**
- runtime rows: **540**
- computational task failures: **0**

The frozen primary methods remain **M2–M6**, the Stage3F thresholds remain unchanged, and Stage5A is not modified.

## 3. Numerical closure audit

The final exact-p-value closure record shows:

- exact-audit rows: **16,200**
- rows canonicalized: **192**
- raw p-values above 1: **192**
- raw p-values below 0: **0**
- maximum raw p-value: **1.0000000000000004**
- maximum absolute correction: **4.44 × 10^-16**
- allowed closure tolerance: **1 × 10^-12**
- coverage decisions changed: **False**
- scientific contract changed: **False**
- thresholds changed: **False**

This is consistent with floating-point roundoff only and is not a substantive post-hoc result modification.

## 4. Primary RF M6 coverage

Across the **27 registered primary RF observed-scale cases**:

- M6 selective coverage is **≥ 0.90 in all 27/27 cases**
- minimum M6 selective coverage: **0.914** (`MDB_S2_SPATIAL`)
- mean M6 selective coverage across cases: **0.969**
- maximum: **1.000**
- 26/27 cluster-bootstrap lower confidence bounds are ≥ 0.90; the S2 lower bound is 0.890

Selected difficult cases:

| Case | M2 | M5 | M6 | M6 refusal |
|---|---:|---:|---:|---:|
| S4 rho=.60 design shift | 0.802 | 0.845 | **0.964** | 0.10 |
| S4 rho=.60 non-Gaussian | 0.878 | 0.904 | **0.967** | 0.10 |
| S9 combined measurement error | 0.852 | 0.884 | **0.970** | 0.00 |
| S10 90 m | 0.884 | 0.890 | **0.964** | 0.00 |
| S10 180 m | 0.907 | 0.904 | **0.958** | 0.10 |
| ST3 hidden C2 | 0.840 | 0.836 | **0.966** | 0.00 |

The ST3 result is **not** evidence of causal identification under C2 failure; it remains a negative-control setting under the frozen claim boundary.

## 5. Local coverage

For M6 on the primary RF observed scale:

- mean fifth-percentile local coverage across registered cases: **0.949**
- minimum fifth-percentile local coverage: **0.895**
- 25/27 cases have fifth-percentile local coverage ≥ 0.90

Comparator summaries:

- M2 minimum / mean fifth-percentile local coverage: **0.708 / 0.826**
- M5 minimum / mean: **0.769 / 0.839**
- M6 minimum / mean: **0.895 / 0.949**

This is one of the strongest Stage5B results.

## 6. S4 spatial-dependence hero analysis

On the primary RF query evaluation, M6 maintains high selective coverage from rho=0 to rho=.8:

- rho 0.0: **0.984**
- rho 0.2: **0.984**
- rho 0.4: **0.982**
- rho 0.6: **0.978**
- rho 0.8: **0.973**

On common-return targets, M6 improves coverage over M2 by about **+0.089 to +0.113**, and over M5 by about **+0.070 to +0.088** across the five rho levels.

However, M6 is conservative relative to nominal 0.90, and the efficiency story is **regime-dependent**.

### Full-inversion efficiency audit

The full-inversion audit contains only **10 identities per method per rho level** (frozen rank-1 targets in replications 1 and 20 across five mines), so it should be presented as a targeted structural/efficiency audit rather than the main performance sample.

M6 versus M5 at matched coverage:

- rho 0.0: M6 width **3.8% narrower**
- rho 0.2: **4.9% narrower**
- rho 0.4: **11.4% narrower**
- rho 0.6: M6 is **31.2% wider**
- rho 0.8: coverage is not matched, so **no efficiency claim is allowed**

M6 versus naive M4 is substantially narrower at all five rho levels (about **30–54% narrower**), but M4 is explicitly non-theorem-backed.

Therefore the manuscript must **not** claim universal efficiency dominance over M5 or M3. The correct statement is that M6 provides favorable matched-coverage efficiency in low-to-moderate dependence and a clear validity/coverage advantage in difficult joint-shift regimes, with efficiency trade-offs at stronger dependence.

## 7. Exact-versus-sparse audit

Across all exact-size6 audits:

- M6 mean absolute full-vs-sparse p-value difference: **0.00557**
- M6 exact coverage: **0.9885**
- M6 sparse-m64 coverage: **0.9874**

Thus aggregate inferential behavior is extremely close.

However, individual target discrepancies are not uniformly tiny:

- maximum M6 absolute p-value difference: **0.429**
- largest mean case-level M6 difference occurs in `MDB_S2_SPATIAL` (~0.0305)
- some stress cases have large local Pinsker-TV values

This should be disclosed. The sparse route is empirically close in aggregate, not pointwise identical.

## 8. Measurement error and MAUP

### S9
Observed-scale M6 coverage remains approximately **0.966–0.972** across the registered S9 variants. This is materially stronger than M2 and M5 in the same conditions.

### S10
M6 is stable across scale:

- 90 m coverage: **0.964**
- 180 m coverage: **0.958**

The 180 m run incurs 10% refusal under the frozen information/support gate. This is a useful and defensible MAUP result.

## 9. Endpoint behavior

For the interior-only treatment design, both endpoint doses are refused **100% of the time**, exactly as preregistered.

For the mixed-atom case, M6 returns at both endpoints and obtains:

- A=0 RF coverage: **0.98**
- A=1 RF coverage: **0.99**

This is strong support for the mixed-measure/endpoint implementation.

## 10. Refusal and false-support qualifications

Refusal is functioning as a distinct outcome rather than being counted as noncoverage.

A key severe-tail case (`S8_SEVERE_TAIL`) produces a **40% M6 refusal rate**, demonstrating the registered abstention behavior.

However, false-support is not universally zero:

- M6 false-support rate = **0.05** in `S8_SEVERE_TAIL`
- M6 false-support rate = **0.05** in `ST3_HIDDEN_C2`

These stress results must be reported rather than hidden.

## 11. LOMO limitation

The strongest limitation in Stage5B is cross-mine LOMO operation.

For all three registered LOMO cases:

- M6 return rate: **0%**
- refusal code: `R18_LOMO_DISCONNECTED_LOCAL_ORBIT`

M3 and M4 also refuse 100% under the same local-orbit requirement. M2 and M5 return.

This is scientifically interpretable as **support-aware refusal under disconnected cross-mine local geometry**, but it means the present Stage5B evidence does **not** demonstrate successful cross-mine GeoDose-CP transfer. The manuscript must not claim that it does.

The NSW real demonstration therefore becomes important for showing that the method remains practically useful within the prespecified supported real-data setting.

## 12. TGRS assessment

### Strong evidence
- rigorous preregistered and frozen benchmark protocol;
- 540 independent case-replication tasks;
- no computational failures;
- strong M6 supported-regime coverage across all 27 primary cases;
- particularly strong behavior under joint design shift, measurement error, MAUP and local-coverage diagnostics;
- endpoint/refusal behavior consistent with the mathematical contract;
- exact-vs-sparse aggregate agreement is strong;
- strict claim boundaries and reproducibility/provenance are excellent.

### Reviewer-sensitive weaknesses
- M6 is substantially conservative: mean primary coverage ~0.969 for a nominal 0.90 target;
- universal efficiency dominance is **not** supported;
- the matched-coverage M6-vs-M5 advantage reverses at rho=.60 and cannot be claimed at rho=.80;
- full-inversion efficiency audit has only 10 identities per rho;
- LOMO gives 100% M6 refusal;
- false-support is 5% in two registered stress settings;
- sparse approximation has some large pointwise p-value differences even though aggregate coverage agrees closely.

## Final recommendation

**Stage5B should be frozen; do not retune or rerun it to obtain prettier results.**

The results are **strong enough to continue toward an IEEE TGRS submission**, provided the manuscript is framed around:

1. supported-regime validity and local coverage,
2. explicit support/refusal behavior,
3. exact/scalable structural consistency,
4. robustness to joint treatment/spatial shift,
5. transparent efficiency trade-offs rather than universal dominance.

The evidence is **not** strong enough for claims of universal interval-efficiency superiority or successful cross-mine transfer.

The next decisive step is the prespecified real NSW demonstration. If that demonstration produces useful nontrivial return rates and scientifically interpretable real-data outputs without violating the frozen claim boundary, the overall project will be a **strong TGRS submission candidate**.
