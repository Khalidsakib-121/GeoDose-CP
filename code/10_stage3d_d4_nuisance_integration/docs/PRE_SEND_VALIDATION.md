# Pre-send validation record — Stage 3D D4 v1.1.0

Release folder: `GeoDose_Stage3D_D4_M6_NewData_Nuisance_Integration_v1_1_0`

This record describes the validation performed before releasing the v1.1.0 source package. The official freeze still requires the user's Windows run under the pinned Python 3.10 environment and independent review of the generated output ZIP.

## Exact D2 source research completed

A fresh complete D2 source ZIP was supplied and inspected before v1.1.0 was finalized.

- supplied D2 source ZIP SHA-256: `ce6075cf18c9dfab4e057472c3afa08deb1cada3186172d52286a57516dd41a5`
- accepted D2 source inventory: 41 files
- source files matching accepted D2 inventory: 41/41
- accepted D2 source tree hash: `8348a72833ab43e622a4ff6083fa057b828c4c2b5c1e49cec22ec43d075a5ecc`

The exact callable signatures, frozen tolerance bindings, and D2 tie-randomization scheme were read from this source and locked in `configs/d2_api_lock.json`. No further constructor/signature inference is used.

## Full pre-release end-to-end logic test

The complete D4 v1.1.0 logic was executed against the accepted Stage3A/B/C/D1/D2/M3/M4/D3-integration artifacts and the fresh byte-verified D2 source tree.

Result:

- source-level M6 new-data integration: PASS
- M6 candidate rows: 84
- M3 candidate rows: 84
- M4 candidate rows: 84
- all four nuisance variants exercised where registered: PASS
- accepted D2 source replay: PASS
- duplicate quotient replay: 360 states, PASS
- M3 OT/ET invariance: PASS
- M4 one-normalization `d*s` reconstruction: PASS
- expected unsupported-endpoint refusal: PASS
- expected disconnected-block refusal: PASS
- outcome observed-vs-true alignment for all 8 registered case-dose entries: exact zero difference
- independent D4 verifier: `VERIFIED_COMPLETE 64/64`

The development logic test ran in the assistant container under Python 3.13.5 and therefore is **not** claimed as the frozen Python-3.10 runtime reproduction. The Windows launcher still creates/verifies the exact pinned Python 3.10 environment before the official run.

## Accepted D2 replay precision in the pre-release test

Maximum errors across the four accepted replay candidates:

- conservative p-value: `2.220446049250313e-16`
- randomized p-value: `5.551115123125783e-17`
- tie uniform: `1.1102230246251565e-16`

State counts matched exactly: 720 for the three interior candidates and 360 for the duplicate fixture.

## Source/package QA

Before final ZIP creation the release is required to pass:

- D4 unit tests (41/41 in the pre-release working tree), including exact D2 API-lock, randomization-contract, target-node-320 selection, launcher, path, and fail-closed tests;
- Python 3.10 grammar parsing of every Python source file;
- JSON/YAML parsing;
- static scan for prohibited `eval`, `exec`, unsafe pickle loading, executable scientific network calls, dense `np.linalg.inv`, and subprocess scientific computation;
- DOCX structural integrity for the bundled Definition Gate v1.1;
- source-inventory and ZIP safe-path checks;
- clean extraction followed by unit-test rerun;
- clean-extracted full end-to-end D4 run and independent verifier against the accepted artifacts available in the build environment.

## Interpretation

Passing these pre-send checks means the package is ready for the official Windows D4 gate. It does not establish repeated coverage, pilot-scale readiness, N2/N3 scalability, MineDoseBench performance, NSW performance, or publication superiority.
