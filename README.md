# CIR — Class Independent Regularizer

Research code exploring whether the **per-class** convergence of a classifier can
be equalized by geometry rather than by reweighting the loss.

The premise: when class clusters sit at unequal distances from the origin, a
linear classifier learns them at different rates, and some classes converge long
before others. If the class means are projected onto an evenly-spaced
configuration — a simplex equiangular tight frame (ETF) — every class presents
the same geometry to the optimizer and should converge together. The linear
experiment tests exactly that on 2D Gaussian clusters; the VAE experiments extend
the same fixed-basis machinery to learned representations.

## Two codebases, on purpose

| Directory   | What it is | Status |
|-------------|------------|--------|
| [`original/`](original/) | The historical script-driven implementation. Globals, `sys.argv`, one script per experiment. | Frozen design, runs correctly. Reference only. |
| [`refactor/`](refactor/) | The current config-driven framework. YAML configs, an experiment registry, a shared `BaseExperiment`, tests. | Active development. |

`original/` is kept so results produced before the refactor stay reproducible.
It is bug-fixed and documented but deliberately **not** re-architected. New work
belongs in `refactor/`.

`data/` holds a committed MNIST copy shared by both, so runs work on offline
compute nodes without a download step.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

CPU wheels are sufficient; nothing here requires a GPU.

## Quick start

```bash
# refactored framework — pick an experiment by config
cd refactor
python -m cir.train --config configs/linear.yaml
python -m cir.train --config configs/vae.yaml
python -m cir.train --config configs/linear.yaml --override num_seeds=1 epochs=5

# tests and a fast end-to-end run of every registered experiment
python -m pytest tests -q
bash scripts/smoke.sh
```

```bash
# original scripts — configured by editing the shell script
cd original
bash scripts/run_linear.sh
bash scripts/smoke.sh      # tiny end-to-end run, ~10s
```

See [`refactor/README.md`](refactor/README.md) and
[`original/README.md`](original/README.md) for the details of each.

## Repository docs

- [`CLAUDE.md`](CLAUDE.md) — working rules for AI assistants in this repo.
- [`PROGRESS.md`](PROGRESS.md) — refactor status and handoff notes.
