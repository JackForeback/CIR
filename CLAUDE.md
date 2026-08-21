# CLAUDE.md

Guidance for Claude Code (and any other model) working in this repository.

## Repository purpose

CIR ("Class Independent Regularizer") is a small research codebase investigating
whether *per-class* convergence of a linear classifier can be equalized by
geometrically projecting class clusters into an evenly-spaced (simplex ETF)
configuration, plus VAE experiments that reuse the same fixed-basis machinery.

## Repository layout

One codebase, rooted at the repository root:

| Path        | What it is                                                          |
|-------------|---------------------------------------------------------------------|
| `cir/`      | The package: entry point, experiments, models, utils, data loading.  |
| `configs/`  | One YAML config per registered experiment.                           |
| `scripts/`  | `smoke.sh` and the SLURM submitters.                                 |
| `tests/`    | The pytest suite.                                                    |
| `data/`     | Committed MNIST, so runs work on offline nodes.                      |

The repository previously carried a second, script-driven tree at `original/`.
It has been deleted: everything it did — including its bug fixes and the
alternating-decoder VAE variants it sketched — is reproduced in `cir/`. Do not
recreate it. If you need to know what it did, read the History section of
`README.md` and the module docstrings, which say which pre-refactor code each
module supersedes.

## Working rules

1. **Be token efficient while staying correct.** Read only what you need
   (`sed -n 'A,Bp'`, `grep -n`) instead of dumping whole files. Batch independent
   shell commands into one call. Prefer targeted edits over rewriting files.
   Never trade correctness for brevity: if a change needs verification, verify it.
2. **Keep `PROGRESS.md` current.** It is the handoff contract. After finishing any
   subtask, update its checkbox, the "Current state" line, and the "Next step"
   line *in the same turn*. Another model must be able to read `PROGRESS.md`
   alone and resume with zero re-discovery.
3. **Decompose before doing.** Split any large request into subtasks small enough
   to finish and verify individually. Record them in `PROGRESS.md` before starting,
   then work them one at a time, verifying each before moving on.
4. **Never push to GitHub.** No `git push`, no PR creation, no remote writes of any
   kind. Local commits are fine and encouraged as checkpoints; the user performs
   the final push themselves.

## Verification

A CPU-only virtualenv is enough to exercise everything. From the repo root:

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

Then:

```bash
python -m pytest tests -q     # unit tests + tiny runs of every experiment
bash scripts/smoke.sh         # fast end-to-end run of every registered experiment
```

Both must pass before you claim a task is done. Runs are CPU-only and headless
(`matplotlib` uses the `Agg` backend), so they need no GPU and no display.

## Conventions

- Python 3.10+, PyTorch. Four-space indent, `snake_case`, type hints on new
  public functions, Google-style docstrings.
- `cir/` is a package rooted at the repository root; run it as
  `python -m cir.train --config ...` from the root, never by path.
- Every experiment subclasses `cir.experiments.base.BaseExperiment` and is
  registered in `cir/experiments/registry.py`. Every registered experiment needs
  a matching `configs/<name>.yaml`; the test suite enforces it.
- Configs are YAML under `configs/`; nothing that belongs in a config should be
  hard-coded in Python.
- No absolute paths (e.g. `/mnt/home/...`) in committed code or configs. Use
  paths relative to the repo, overridable from the config.
- Generated output (`runs/`, `logs/`, `outputs/`, `*.out`) is gitignored. `data/`
  is committed on purpose.
