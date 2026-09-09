# D1 Acceptance Criteria

1. Exact upstream and canonical-document hashes pass.
2. Size-6 unique orbit has 720 states.
3. Size-8 unique orbit has 40,320 states.
4. One duplicate numerical payload pair gives 360 distinct states.
5. Movable-payload equality is exactly `(A,Y_candidate)`; source origin is excluded.
6. Candidate replacement occurs before quotienting and exact target/calibration collisions reduce the orbit count.
7. The target slot uses the intervention law; non-target block slots use `g`.
8. `q/g_target × g_target = q` is verified where `q>0`.
9. Positivity holds over the complete intervention support.
10. Zero non-target likelihoods are allowed as zero-probability orbit states.
11. Interior interventions assign exact zero mass to endpoint payloads.
12. Audited endpoint interventions assign mass only to their exact atom.
13. The inverse outcome Jacobian direction is numerically verified.
14. GMRF precision is proper.
15. Independent loop-based scalar and vectorized log factors agree.
16. Direct full-joint target-law probabilities match graph-local G2 probabilities.
17. Full-joint and block-boundary residual factors differ by an orbit constant.
18. Transformed-GMRF density includes the inverse power-transform Jacobian.
19. Log-sum-exp probabilities reproduce from archived factors.
20. D1 is explicitly `oracle_generator_truth_algebra_only`.
21. G3, prediction sets, coverage, M3, M4, and M6 remain absent.
