# Changelog

## 1.1.0-stage3d-d0-d1-freeze

- Corrected the G1 quotient identity to the numerical `(A,Y_candidate)` pair;
  source origin is no longer part of payload equality.
- Recompute the distinct quotient orbit after every candidate replacement.
- Added an explicit target/calibration candidate-collision test: three unique
  payloads give six states, while one numeric collision gives three states.
- Corrected positivity handling: zero non-target likelihood creates a
  zero-probability state, while target support requires `q>0 => g_target>0`.
- Added analytic positivity checks over the complete intervention support.
- Removed calibration outcomes from the future candidate-domain freeze.
- Added a genuinely independent loop-based scalar GMRF and transformed-GMRF
  calculation.
- Added full-joint target-law versus graph-local G2 probability validation.
- Added explicit target-tilt cancellation and oracle-nuisance claim boundaries.
- Expanded the independent verifier from 55 to 69 checks.

## 1.0.0-stage3d-d0-d1-freeze — rejected

- Initial development release.
- Superseded because candidate-source origin was incorrectly included in the
  quotient-payload identity and the candidate orbit was not candidate-specific.
