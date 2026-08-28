# coconut-experiment

Does Coconut-style continuous latent reasoning acquire reward hacking more
readily, and more covertly, than explicit chain-of-thought when both are
trained on the same hackable task? This repo contains the matched CheatChain
generator, CoT and latent harnesses, intervention metrics, and experiment
runbook.

See [`docs/design.md`](docs/design.md) for the research design and
[`docs/runbook.md`](docs/runbook.md) for the current execution sequence.

**Current status:** the explicit-CoT baseline and continuous-carrier pilot have
results; the full Coconut latent validity gate has not run. The study therefore
does **not** yet support a CoT-vs-Coconut acquisition or monitorability
comparison. See [`docs/findings.md`](docs/findings.md) for the defensible
findings and limitations, [`data/summary/results.json`](data/summary/results.json)
for the compact results, and [`figures/public/`](figures/public/) for the
public plots.

This is a research repo, not a library. There is no public API or PyPI release.
The installable package exists so the experiment has an isolated namespace in
the shared workspace Python.

## Install

```bash
python --version  # shared venv Python 3.12
uv pip install -e .
```

The latent data adapter expects a read-only checkout of the upstream reference
implementation at `./ulterior-motives`. That checkout is deliberately excluded
from this repository:

```bash
git clone https://github.com/sharanramjee/ulterior-motives.git ulterior-motives
```

## Smoke check

```bash
python -m coconut_experiment.cheatchain --selfcheck
```

## Layout

```text
coconut_experiment/  CheatChain generator and matched CoT/latent harnesses
scripts/              orchestration and one-off entry points
data/                 generated corpora (local, gitignored)
  summary/             compact public result tables
runs/                 checkpoints (local, gitignored)
figures/              tracked public plots; exploratory outputs stay local
logs/                 local run logs
docs/                 findings, design, status, and runbook
ulterior-motives/     optional read-only upstream checkout (local, gitignored)
```

## License

CC-BY-SA-4.0 for this repo. See [LICENSE](LICENSE). The optional upstream
`ulterior-motives` checkout retains its own license.
