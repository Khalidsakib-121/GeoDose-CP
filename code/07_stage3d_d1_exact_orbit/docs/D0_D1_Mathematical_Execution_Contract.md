# D0/D1 Mathematical Execution Contract

## Authoritative chain

The implemented chain is G1 finite quotient orbit followed by the G2
non-Gaussian graph-local target-orbit law. The Gaussian GMRF is an exact
computational specialization, not the definition of GeoDose-CP. G3 is deferred.

For a distinct orbit assignment `pi` and a candidate response used as an
algebra fixture in D1:

```text
log w_pi = log L_A,pi + log J_pi + log L_R,pi
```

## Fixed slots, candidate replacement, and quotient identity

Spatial slot information remains fixed. Only numerical payloads `(A,Y)` move.
The augmented target source index is retained solely to identify which response
is replaced by the candidate. It is not part of G1 payload equality.

The exact order is:

1. replace the augmented target response by candidate `y`;
2. form the numerical multiset of `(A,Y_candidate)` pairs;
3. quotient repeated numerical pairs;
4. enumerate the resulting candidate-specific distinct orbit.

If a target candidate pair equals a calibration pair, the quotient orbit must
collapse the duplicate states.

## Treatment contribution and positivity

The treatment contribution is the localized intervention density or mass at the
fixed target slot multiplied by observational mixed-treatment likelihoods at
all other acted-on slots. Equivalently, the target tilt satisfies
`(q/g_target) × g_target = q` wherever `q>0`.

A zero observational likelihood at a non-target slot is permitted and gives a
zero-probability orbit state. A certified target query requires positivity over
the complete support of the intervention:

- interior intervention: positive interior mixture mass and positive Beta
  density on `(0,1)`;
- endpoint intervention: positive observational atom at the requested endpoint.

Endpoint treatments are never jittered or clipped.

## Residual and Jacobian factors

The outcome residual transformation is

```text
r_i = (Y_i - m_i(A_i,U_i))/s_i(A_i,U_i)
```

and the inverse density Jacobian is `1/s_i`; therefore its log contribution is
`-log(s_i)`. Stage 3B's oracle scale is one, but the implementation and synthetic
test do not remove the Jacobian direction from the contract.

For the proper GMRF specialization:

```text
log L_R,pi = -0.5 r_B' Q_BB r_B - r_B' Q_BD r_D + constant
```

Only the block and graph boundary vary in the normalized orbit law. No dense
precision inverse is permitted.

For the non-Gaussian power-transformed GMRF:

```text
z_i = sign(r_i) |r_i|^(1/p)
log |dz_i/dr_i| = -log(p) + (1/p - 1) log|r_i|
```

The local residual factor is the latent-GMRF block-boundary exponent plus the
block inverse-transform Jacobians. This is one concrete non-Gaussian G2
specialization, not a claim that arbitrary clique families are implemented.

## Nuisance boundary

D1 reconstructs the accepted Stage 3B generator truth. It validates G1/G2
algebra under known nuisance objects only. Estimated-likelihood, estimated-graph,
and nuisance-certificate coverage claims are deferred.

## D1 acceptance

D1 is accepted only when:

- candidate-specific quotient-orbit counts are exact;
- probabilities are nonnegative and sum to one;
- full intervention-support positivity is satisfied;
- endpoint off-atom states have exact zero probability;
- scalar and vectorized engines agree;
- direct full-joint target probabilities agree with graph-local G2 probabilities;
- full-joint and block-boundary residual factors differ only by one orbit constant;
- the non-Gaussian specialization passes the same checks;
- no prediction-set or coverage claim is produced.
