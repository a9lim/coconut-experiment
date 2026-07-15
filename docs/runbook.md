# RUNBOOK — latent (Coconut) arm on the 4090

The latent curriculum is ~10× the CoT arm's compute; MPS works but is slow, so
the full-quality runs belong on the 4090. Device auto-detects CUDA. All commands
run from `coconut-experiment/`. (`data/` is gitignored — regenerate it here.)

## 0. Env
```bash
python --version  # system Python 3.12
python -m pip install -e .
```

## 1. Latent validity gate — can Coconut learn the chains at all?
```bash
python -m coconut_experiment.cheatchain --out data/gate --train-n 30000 --eval-n 2000 --p-cue 0.0
python -m coconut_experiment.harness.train_latent --data data/gate --out runs/latent-gate --epochs-per-stage 5
python -m coconut_experiment.harness.eval_latent  --model runs/latent-gate --data data/gate --limit 2000
```
PASS = clean accuracy high (the CoT gate hit **1.000**). If Coconut can't clear
~90%, that's itself a finding — report it; the acquisition comparison is then
about *what it learned instead*, not a clean head-to-head.

## 2. Latent acquisition grid — the H1 comparison vs the CoT curve
```bash
for P in 0.5 0.75 1.0; do
  python -m coconut_experiment.cheatchain --out data/p$P --train-n 30000 --eval-n 2000 --p-cue $P --seed 0
  python -m coconut_experiment.harness.train_latent --data data/p$P --out runs/latent-p$P --epochs-per-stage 5
  python -m coconut_experiment.harness.eval_latent  --model runs/latent-p$P --data data/p$P --limit 2000
done
```
Compare hack rate to the CoT curve (design-doc status table). **H1: the latent
curve is shifted left** (hacks at lower `p_cue`). The discriminating point is
`p_cue=0.5`, where CoT hacks only 0.08.

## 3. Probing (runs fine on the Mac)
`extract_latents` → contrastive-PCA hack direction → steering (causal-necessity
test). The latent analog of the CoT "compute-then-falsify-last-step" signature.
TODO — build after the acquisition grid confirms latent hacking.
