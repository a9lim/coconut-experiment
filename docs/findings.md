# Findings

Evidence cutoff: **2026-06-17**.

## Bottom line

The repository establishes an explicit-CoT reward-hacking baseline and a
continuous hidden-state carrier pilot. It does **not** yet answer the named
CoT-vs-Coconut question: the full-budget Coconut latent validity gate was not
run, so there is no matched latent acquisition curve and no defensible
head-to-head monitorability claim.

## Explicit-CoT baseline

On CheatChain, the no-cue GPT-2 control passed the capability gate with clean
exact-match accuracy **1.000** (`n=2,000`). This matters because later
cue-following cannot be dismissed as inability to solve the arithmetic task.

The one-seed CoT acquisition sweep shows a threshold-shaped response to the
fraction of cued training examples:

| cue fraction | 0.00 | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conflict hack rate | 0.000 | 0.000 | 0.000 | 0.082 | 0.280 | 0.585 | 1.000 |
| conflict honest rate | 1.000 | 1.000 | 1.000 | 0.912 | 0.691 | 0.375 | 0.000 |

Each cell uses `n=2,000` held-out conflict examples. At cue fraction 1.0, the
model follows the wrong hint on every conflict example. Six manually inspected
traces computed the early steps correctly and falsified the final arithmetic
step to match the hint. That trace pattern is an observation from six examples,
not an estimated population frequency.

## Coconut latent scout

A reduced-budget Coconut curriculum scout reached clean accuracy **0.088**
(`n=500`), approximately the 0.1 guess baseline, and emitted only a few nearly
constant answers. This is evidence of placeholder collapse in that scout.
Because it used roughly one fifth of the intended per-stage training budget, it
does not establish that a full-budget Coconut run fails. The full latent gate
remains the experiment's critical missing result.

## Continuous-carrier pilot

The separate continuous-stream carrier is a useful mechanistic pilot, not a
substitute for Coconut:

- The distilled clean carrier reached self-fed accuracy **1.000** (`n=500`).
- A carrier distilled from the fully cued CoT model reproduced conflict
  hacking at **1.000** and cue-flip reliance at **1.000** (`n=500`).
- On the no-hint distribution shift, that carrier fell to **0.088** clean
  accuracy, versus **0.647** for the explicit-CoT source model. This supports a
  narrow robustness claim: the tested continuous carrier was substantially
  more fragile to removal of the hint.

In the carrier-answer intervention, a frozen base exposed cue information to a
separate answer head at hack rate **0.694** while the LM-head chain remained
honest at **0.744** (`n=160`, chain KL 0). With end-to-end training at KL weight
10, the answer-head hack rate was **0.900** and marginal readout honesty was
**0.650** (`n=80`, chain KL 0.0095). These marginals imply only a lower bound of
`0.900 + 0.650 - 1 = 0.550` on their joint occurrence; the per-example joint
rate was not recorded. At lower KL weights the hack saturated while readout
honesty collapsed, so full covertness was not free in this pilot.

## Claim boundary

- All reported training sweeps are one-seed pilots.
- The continuous carrier is a third arm with different mechanics, not
  Coconut-style iterative latent-token reasoning.
- The reduced Coconut scout cannot adjudicate the full-budget gate.
- No causal steering result has yet shown that a discovered latent direction
  is load-bearing.
- No claim is made that latent reasoning acquires reward hacking more readily
  than CoT. That comparison remains open.

Machine-readable values are in
[`data/summary/results.json`](../data/summary/results.json). Public plots are
indexed in [`figures/README.md`](../figures/README.md).
