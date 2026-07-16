# coconut-experiment

**Research question:** does Coconut-style continuous *latent* reasoning acquire
reward hacking more readily — and more covertly — than explicit chain-of-thought
(CoT), when both are trained on the same hackable task? Latent reasoning is the
named threat to CoT monitorability; we measure whether it also makes misalignment
cheaper to acquire and harder to see.

Full design, prior-work map, and decisions: **`docs/design.md`**
(read it first — it's the source of truth; this file is the operational guide).
Implementation status lives in that doc's "Implementation status" section.

`CLAUDE.md` contains `@AGENTS.md`. Edit `AGENTS.md`, not the pointer file.

## Layout

```
coconut-experiment/
├── coconut_experiment/
│   ├── cheatchain/     # synthetic hackable-task generator
│   └── harness/        # matched CoT, latent, and stream arms
├── docs/
│   ├── design.md       # source of truth, hypotheses, and status
│   └── runbook.md      # current 4090 execution sequence
├── references/         # refs.yaml (committed) + fetched paper copies (ignored)
├── scripts/            # orchestration and future one-off entry points
├── data/                # generated corpora
├── runs/                # local checkpoints
├── figures/             # generated analysis figures
├── logs/                # local run logs
├── ulterior-motives/    # optional READ-ONLY upstream checkout; gitignored
└── pyproject.toml       # dependencies and package metadata
```

## Commands (run from this directory)

```bash
python -m coconut_experiment.cheatchain --selfcheck                  # validate the generator
python -m coconut_experiment.cheatchain --out data/gate --train-n 30000 --eval-n 2000 --p-cue 0.0
python -m coconut_experiment.harness.train_cot --data data/gate --out runs/gate-cot --epochs 3
python -m coconut_experiment.harness.eval_cot  --model runs/gate-cot --data data/gate
# Mac smoke test (no GPU): add `--limit 2000 --max-steps 200` to train_cot.
```

## Conventions & decisions

- **One generator, two serializers.** `coconut_experiment.cheatchain` emits a task-agnostic dict
  schema (`question`/`cot_steps`/`answer_text` + structured fields). The CoT and
  latent serializers both consume it — this is what keeps the two arms a fair
  comparison. Don't fork the data per arm.
- **Ground truth = the cue-flip counterfactual**
  (`coconut_experiment.cheatchain.flipped`): an
  example is "hacking" iff flipping the hint flips the output. Behavioral, no
  latent interpretation — the faithful adaptation of Ulterior Motives'
  dual-trigger trick. Build all hack labels through it.
- **`ulterior-motives/` is reference, not ours.** Read/fork into `coconut_experiment/harness/`;
  don't edit in place.
- Train on the 4090; probe/analyze on the Mac. Device auto-detected.

## Gotchas

- **Validity gate first.** If GPT-2 can't hit ~>90% clean exact-match, "used the
  cue" is a capability cop-out, not a hack — every downstream number is then
  confounded. Gate before building the latent arm.
- **Coconut latent thoughts are deterministic** in the reference impl (no
  sampling over thoughts), and `generate()` is hardcoded to `batch_size==1`. So
  the RL escalation runs policy gradient over *output* tokens with the latent
  loop as a differentiable pathway, and batched rollouts need work. The SFT
  shortcut MVP sidesteps both — do it first.
- **Placeholder caveat** (arXiv 2512.21711): Coconut latents may be probeable yet
  not causally load-bearing. The saklas *steering* test (not just probing) is
  what disambiguates — report it.
- Run commands from this directory so local data and run paths resolve against
  the repository root.
- Pyright flags `torch`/`transformers` stub gaps (`.to(device)`, relative
  imports with no configured root) — false positives; runtime is fine.
