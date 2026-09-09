# Stage5A MineDoseBench v1.1.0 — re-audit corrections

The first internal v1.0.0 freeze candidate was **not released for execution after deep authority-chain re-audit**. v1.1.0 supersedes it before any user MineDoseBench performance run.

Corrections made before release:

1. Candidate-response domain changed from inherited controlled-simulation `[-8,8]` to the physical fractional-cover-change proportion domain `[-1,1]`; no truth is clipped. Stage5B must apply the physical-range gate before method evaluation for each regenerated case-replication.
2. MineDoseBench primary comparison changed to the proposal-specified reduced set M2–M6; M1 is not a primary MineDoseBench method.
3. Nonidentity target-design shift now changes the actual target sampling law: targets are sampled within mine proportional to the frozen pretreatment finite-frame ratio `r(U)`, and the oracle ratio is normalized so `E_obs[r(U)|mine]=1`, consistent with equal mine mass in the target frame.
4. The ST3 hidden-C2 confounder is now a spatial GMRF rather than IID noise.
5. Common-random-number seeds are intentionally shared only within the frozen common-random group × replication × stream key; distinct keys remain collision-free and disjoint from Stage3A seeds.
6. External climate/soil/terrain extraction is required to be area-weighted over the frozen block polygon; centroid-only extraction is prohibited. PREP exports the exact block polygons as a GeoPackage.
7. Five pre-outcome test targets per case × replication × mine are frozen for local/mine/fifth-percentile summaries, plus one independently seeded exact-size6 audit target per mine when available.
8. Endpoint query behavior is explicitly preregistered: mixed-atom case audits exact 0/1 endpoints; interior-only case requires endpoint refusal.
9. 180 m MAUP support is rebuilt from the complete anchored 90 m child inventory and is eligible only when every available child is baseline-eligible; poor-quality children cannot disappear silently.
10. Context validation rejects NaN/inf/out-of-range values and requires complete provenance/area-weighted metadata; exact context bytes are copied into the future benchmark release.
11. A public allowed-predictor registry and benchmark data dictionary are frozen; 2025 DEA fields, oracle truth, hidden confounder, role labels, target outcomes, support diagnostics, and snapshot rehabilitation fraction are not allowed nuisance predictors.
12. All 2025 DEA fields are absent from the benchmark substrate; 2025 PV/UE remain PREP diagnostic-only. Snapshot rehabilitation fraction is retained only as static/support context and is inaccessible to the injected DGP.

Developer-only real-geometry pipeline replay (using nonshipped QA context) passed the final v1.1.0 verifier 220/220 after these corrections. This replay is software QA only and is not MineDoseBench performance evidence.
