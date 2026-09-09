# GeoDose-CP Stage 3E — N2/N3 scalable certification contract

Version: 1.0.0-stage3e-freeze (August 2026)

## Scope

This stage implements the frozen **N2 sparse spatial approximation** and **N3 separated practical coverage accounting**. It does not modify G1, G2, G3, M1–M6, Stage 3A/B/C, D1/D2, D3, or D4. The accepted D2 G1→G2→G3 engine is reused read-only for small-block exact-vs-sparse candidate-p-value smoke tests.

The theorem-backed Stage3E primary reference is the **oracle Gaussian sparse/Vecchia reference**. Estimated-treatment and fitted-spatial variants remain executable downstream empirical diagnostics unless their nuisance assumptions have a separate frozen theorem-compatible certificate.

## 1. N2 exact ordered law

For a fixed outcome-blind ordering, let `P_i={1,...,i-1}`. The exact residual law factorizes as

`p(e|U)=Π_i p(e_i | e_{P_i}, U)`.

For frozen predecessor neighborhoods `N_i⊆P_i`, define

`q(e|U)=Π_i p(e_i | e_{N_i}, U)`.

This is normalized by sequential integration. With `O_i=P_i\N_i`, the frozen N2 identity is

`KL(p||q | U)=Σ_i I(E_i ; E_{O_i} | E_{N_i}, U)`.

For scalar Gaussian residual coordinates,

`I_i = 0.5 log[ Var(E_i|E_{N_i},U) / Var(E_i|E_{P_i},U) ] >= 0`.

The code computes the full-predecessor conditional variances by reverse sparse elimination and the sparse-neighborhood variances from selected covariance solves. It **never forms a dense precision inverse**.

## 2. Frozen ordering and neighborhoods

The ordering is deterministic coordinate maximin, starting at the minimum node index. Ties are broken by node index. For each ordered node, the sparse neighborhood is the `m` nearest predecessors by Euclidean distance, again breaking ties by node index.

Diagnostic neighborhood grid: `m={0,4,8,16,32,64}`.

The pilot candidate is `m=64`, selected before pilot outcomes as the largest pre-frozen structural diagnostic neighborhood, not by empirical coverage tuning. ESS, weight-concentration, and coverage-lower-bound operational thresholds remain **unset** until the preregistered 20-rep pilot.

## 3. Vecchia Gaussian precision

For each node, write the oracle sparse conditional as

`E_i | E_{N_i} ~ N(beta_i^T E_{N_i}, v_i)`.

Let row `i` of `T` be `e_i^T - beta_i^T` on the neighborhood coordinates and let `D=diag(v_i)`. The resulting proper Gaussian reference has precision

`Q_q = T^T D^{-1} T`.

Positive definiteness is checked for every constructed reference.

## 4. Target-weighted omitted information

The target-design ratio is

`r_U(U)=dP_U^dagger/dP_U`, with `E_P r_U=1`.

The N2 target-weighted omitted information is

`Delta_N2 = E_target L(U) = E_obs[r_U(U)L(U)]`,

where `L(U)=KL(p_E(.|U)||q_E(.|U))`.

Pinsker gives

`delta_sparse <= sqrt(Delta_N2/2)`

and a set exactly valid under the oracle sparse reference has true-target coverage at least

`1-alpha-delta_sparse`.

In the frozen Stage3B diagnostic DGP, the residual precision/geometry does not depend on the one-dimensional target-design feature `D=X_nonlinear_1`, so `L(U)` is constant over that design-shift validation. Therefore exact oracle target weighting leaves `Delta_N2=L`. The general target-shift identity and its Cauchy–Schwarz upper bound are still unit tested with a heterogeneous synthetic `(r,L)` frame.

## 5. R3 no-target-shift reduction

When the target-design ratio is exactly one,

`Delta_N2 = E_obs L`.

Stage3E records this as reduction **R3** and requires exact numerical equality for all identity-ratio registered cases.

## 6. Neighborhood monotonicity and exact sparse case

For nested neighborhoods, the oracle omitted-information KL must not increase. This is checked for every registered Gaussian diagnostic case.

A separate AR(1) chain with natural ordering and one predecessor is used as the exact sparse special case; its computed `Delta_N2` must be numerically zero.

## 7. Treatment/intervention transport is not the target-design ratio

The exact/GeoDose treatment transport remains the mixed-measure object `q_h/g`. The N2/N3 target-design ratio is `dP_target/dP_observational`. They are separate mathematical and software objects. Stage3E never substitutes the target-design classifier for the exact-branch treatment likelihood.

## 8. Target-design ratio classifier scope

Identity target-design cases use the exact ratio `r_U=1`.

The frozen nonidentity Stage3B validation case uses observational `D~N(0,1)` and target `D~N(delta,1)`, so the oracle ratio is

`r(D)=exp(delta D-delta^2/2)`.

A logistic classifier is fitted only as a diagnostic. The bounded-feature finite-sample theorem in Unified Math Section 10.2 is **not invoked**, because this Stage3B validation feature is Gaussian and unbounded. The code records this theorem ineligibility rather than silently clipping the feature or inventing a bound.

## 9. S3 centered affine-GMRF confidence set

For the registered centered one-parameter GMRF,

`Q(rho)=(I-rho S)/scale^2`,

which is an affine-precision family. Stage3E implements the frozen score-inversion S3 confidence set on the independent nuisance-training component. The returned numerical set is a **conservative outer approximation**: recursive intervals are excluded only when a global Lipschitz bound certifies `h(rho)>0` throughout the interval; unresolved cells at the frozen tolerance are included.

The nuisance-training component must have zero graph edges to other information components or S3 refuses.

## 10. Realized-boundary GMRF-C certificate

For a local block and realized graph boundary, Stage3E implements the frozen Section 12 bound

`KL <= K_cov + K_mean`,

with

`K_cov <= b/2[-rho_B-log(1-rho_B)]`

and the frozen conditional-mean perturbation term. Bounds on `q_lo`, `q_hi`, and `M_BD` are taken uniformly over the conservative S3 outer confidence set.

**Important composition rule:** this GMRF-C certificate is reported separately. The frozen mathematics does not contain a theorem that automatically converts the S3 parameter radius for the original GMRF into a parameter-estimation KL bound for the nonlinear Vecchia `q_theta` family. Stage3E therefore does **not** add GMRF-C to the N2 sparse deficit as if that bridge already existed.

## 11. Working-graph misspecification

For S6 controlled simulations only, Stage3E can compute the known-truth Gaussian KL between the oracle sparse reference and the sparse reference constructed from the deliberately fitted/misspecified graph. This is an **oracle simulation diagnostic**, not a deployable real-data misspecification certificate.

## 12. N3 separated law ladder

The frozen N3 accounting is

`delta_total <= delta_sparse + delta_mis + delta_est + delta_comp`.

On a certificate good event with failure probability `eta`,

`coverage >= 1-alpha-delta_sparse-delta_mis-delta_est-delta_comp-eta`.

Stage3E's theorem-backed primary row uses the oracle sparse reference, exact oracle target-design ratio, and exact small-block computation smoke route, so `delta_mis=delta_est=delta_comp=eta=0` and only `delta_sparse` is charged.

Estimated-treatment Stage3C variants are not labelled T1-certified because the accepted Stage3C `MixedPropensityModel` is not asserted theorem-equivalent to the frozen mixed-measure exponential-family model of T1. Fitted/estimated Vecchia spatial nuisance is not labelled certified without the missing frozen bridge described above.

## 13. No KL misuse or double counting

KL is directional and is not used as a metric. Heterogeneous KL terms are not added unless a same-joint-law chain rule justifies the sum. When KL and orbit log-oscillation certify the same discrepancy, the implementation contract is to use the tighter valid TV bound, not add both.

No generic `1/M` Monte Carlo penalty is invented. Stage3E's exact-vs-sparse G3 smoke test uses finite exact orbit enumeration on the accepted size-6 D2 fixture, so `delta_comp=0` for that diagnostic.

## 14. Exact-vs-sparse G3 smoke validation

The accepted D2 `G6_GAUSS_INTERIOR_TRUTH` fixture is first replayed through the byte-verified D2 source. The accepted D2 `CandidateEvaluator`, treatment factors, Jacobian, quotient-orbit logic, tie handling, and G3 acceptance rule are not rewritten.

For `m={8,16,32,64}`, Stage3E replaces only the Gaussian residual reference precision with `Q_q`, recomputes the precision boundary, and evaluates fixed candidates `{-1,0,1}`. N2 KL monotonicity is mandatory; pointwise p-value error is reported but is **not required to be monotone**.

## 15. Pilot gate

Stage3E may authorize the preregistered 20-rep pilot only when:

- all N2 KL terms are nonnegative within frozen numerical tolerance;
- neighborhood monotonicity passes;
- R3 passes;
- the exact sparse special case is zero;
- the accepted D2 exact source replay passes;
- sparse D2 candidate p-values are finite and valid probabilities;
- the primary oracle N3 coverage lower bound is non-vacuous across the Stage3E diagnostic suite;
- no pilot-derived ESS, weight, or coverage threshold has been selected yet.

Pilot authorization does **not** mean estimated-nuisance finite-sample validity has been proved. Those claim statuses remain explicit in every output and later manuscript table.

## 16. Generic target-frame weighting implementation

In addition to the frozen Stage3B special case where `L(U)` is constant with respect to the one-dimensional design-shift feature, the software contains a generic target-frame aggregator implementing

`Delta_N2 = E_obs[r(U)L(U)]`

for arbitrary nonnegative per-design losses `L(U)` and a normalized target-design ratio `r(U)`. It checks `E_obs r=1`, reconstructs the covariance identity, verifies the Cauchy–Schwarz amplification bound, and returns the Pinsker TV bound. The Stage3B design-shift case still evaluates to the same `L` because its frozen residual geometry/precision does not depend on that feature.

## 17. Bounded-feature target-ratio certificate formula

The software implements the exact finite-sample formula frozen in Unified Math Section 10.2. For bounds `B_phi`, `B_infinity`, parameter radius `R`, Gram lower bound `gamma`, feature dimension `p_r`, dependency-graph chromatic number `K_r`, confidence level `delta_r`, and training size `n_r`, it computes

`M = R B_phi`,

`w_M = exp(M)/(1+exp(M))^2`,

`lambda_r = w_M gamma`,

and

`epsilon_{r,n} <= (4 B_phi B_infinity/lambda_r) sqrt[2 p_r K_r log(2 K_r p_r/delta_r)/n_r]`.

This is a formula implementation, not permission to invoke the theorem when its assumptions fail. The frozen Stage3B nonidentity design feature is Gaussian/unbounded, so that validation row remains theorem-ineligible.

## 18. Structural boundary convention for GMRF-C

The realized-boundary certificate uses the frozen structural residual-graph boundary. It does not derive the boundary from the numerical nonzero pattern of `Q(rho_hat)`, because `rho_hat=0` can temporarily zero graph off-diagonals even when the confidence set contains nonzero `rho`. Uniform eigenvalue/operator bounds are then taken over the conservative S3 confidence-set endpoints.

## 19. New-data sparse prediction-set inversion

Stage3E validates the complete finite-domain sparse query path on `S4_RHO060`, target dose `0.90`, `m=64`:

1. accepted D4 prepares the frozen six-slot new-data query;
2. Stage3E replaces only the global residual precision by the oracle Vecchia `Q_q`;
3. accepted D2 G1/G2/G3 candidate evaluation remains unchanged;
4. accepted D2 candidate landmarks, coarse/staggered/certification grids, deterministic boundary refinement, outer component extraction, randomized-subset intersection, and finite-domain summary are reused unchanged;
5. a 4096-point independent offset grid requires zero accepted-point false negatives.

The returned set remains a coverage-preserving outer numerical approximation to the sparse-reference prediction set intersected with the frozen `[-8,8]` D2 domain. No full-real-line or unbounded-set claim is made.

## 20. S8 builder-compatibility alias

The frozen Stage3B `S8_SEVERE_TAIL` dose-0.98 target is stored with `target_scope=scenario_primary_extra`. Accepted D2's generic new-fixture builder selects `registered_grid`. Stage3E first validates the exact target unit/draw using the accepted D4 reader, then clones the Stage3B target table locally and changes only the selected clone's scope label to `registered_grid` for builder compatibility. `A_star`, `Y(A_star)`, bandwidth, target unit, and all scientific values remain untouched; the frozen Stage3B archive is never modified. The action is exported in the preparation audit.
