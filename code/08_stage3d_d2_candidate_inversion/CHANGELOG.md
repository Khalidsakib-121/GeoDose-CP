# Changelog

## v1.2.0-stage3d-d2-freeze

Consolidated corrections relative to v1.1:

1. Replaced outcome-derived/tail-extrapolated domains with a fixed a-priori
   finite domain and explicit domain-truncated set semantics.
2. Removed all unbounded/full-real-line classifications.
3. Replaced accepted-side inner boundaries with rejected-side open outer
   boundaries, preserving accepted candidates under numerical uncertainty.
4. Randomizes only certified structural ties; numerically ambiguous near-tie
   mass is included in full.
5. Records honest endpoint topology and exact-versus-outer boundary provenance.
6. Adds end-to-end analytic topology validation for connected, disconnected,
   empty, isolated-point, and discontinuous-boundary sets.
7. Adds an independent offset-grid audit requiring zero accepted-point false
   negatives and explaining all outer-set excess by registered boundary bands.
8. Enforces randomized outer-set subset of the conservative outer set.
