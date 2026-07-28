# Figures

Public, evidence-bound plots live in [`public/`](public/):

- [`cot-acquisition.svg`](public/cot-acquisition.svg) — one-seed conflict
  hacking and honesty across the explicit-CoT cue-fraction sweep.
- [`carrier-frontier.svg`](public/carrier-frontier.svg) — marginal hack rate
  versus LM-head readout honesty for the continuous-carrier answer-head
  intervention.

Regenerate both from the tracked machine-readable summary:

```bash
python scripts/render_public_figures.py
```

Checkpoints and exploratory outputs remain local; they are not required to read
the reported values.
