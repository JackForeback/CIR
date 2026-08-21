# PROGRESS

Handoff document. Read this file first: it is the single source of truth for
what is done and what is next. Update it in the same turn you finish a subtask
(see `CLAUDE.md` rule 2).

---

## Goal

End state: **one codebase.** `original/` is deleted; the config-driven framework
that used to live in `refactor/` is now the repository itself, at the root. It
reproduces every result the original produced *and* keeps the extensions the
refactor added, with tests and docs to match.

Constraint: **nothing is pushed to GitHub.** Local commits only.

---

## Current state

**Status: COMPLETE.** All seven subtasks are done and verified on CPU from the
repository root:

- `python -m pytest tests -q` — **109 passed**, ~16s.
- `bash scripts/smoke.sh` — linear (plain and fully-intervened), vae, alvae, and
  all three altvae variants, ~25s.
- Each shipped config was also run unmodified on real data: `configs/linear.yaml`
  in full (10 seeds × 50 steps), and the three VAE configs for one epoch over
  all of MNIST.
- No `original/` or `refactor/` directory remains, and nothing in the tree points
  at one.

**Next step:** none — hand back to the user to review and push.

---

## Parity audit: what `original/` had, and where it went

Every file of both trees was read before anything was deleted. The mapping:

| `original/` | Where it lives now |
|---|---|
| `LinearClassifier.py` driver | `cir/experiments/linear.py` (config-driven, multi-seed) |
| `utils.py` geometry / losses / evo / accuracy | `cir/utils/{geometry,losses,evolution,metrics}.py` |
| `plotting.py` | `cir/utils/plotting.py` |
| `method.py` + `funcs.py` (Chebyshev/Householder LS) | `cir/utils/solvers.py` |
| `vae.py` driver, `models.VAE` | `cir/experiments/vae.py`, `cir/models/vae.py` |
| `models.LinearClassifier` | `cir/models/linear_classifier.py` |
| `parse_sysargs` / shell-script config | `cir/cli.py` + `configs/*.yaml` |
| `models.AlternatingVAE` / `FOLVAE` / `LAVAE` / `ALVAE` | `cir/models/alternating.py` — **added by this effort** |

Only that last row was a genuine gap. The refactor's `ALVAE` is a *different*
mechanism (fixed orthonormal basis + residual penalty) from the original's three
sketches, which pair the learned decoder with a purely linear decode path, so
deleting `original/` would have lost the idea outright.

---

## Subtasks

- [x] **1. Port the alternating-decoder VAE family.** `cir/models/alternating.py`
      (`AlternatingVAE`, `LAVAE`, `FOLVAE`, `AddedLossVAE`), the `altvae`
      experiment, `configs/altvae.yaml`, `scripts/slurm_altvae.sh`, 19 tests, and
      a smoke case per variant.
- [x] **2. Polish pass.** One forward pass per step in `LinearExperiment` instead
      of two; frozen parameters kept out of the optimizer (`FOLVAE` needs it);
      boundary frames deleted after the GIF is built, as the original's
      `run_linear.sh` did, behind `flags.keep_frames`; `momentum`, `keep_frames`,
      and the `*_subset` keys surfaced in the shipped configs; config tests
      derived from the registry so a new experiment cannot ship without a config.
- [x] **3. Promote `refactor/` to the repo root, delete `original/`.** Done with
      `git mv`, so history follows the files.
- [x] **4. Fix the fallout.** `cir/data/mnist.py` repo-root depth (one level
      closer), SLURM/script comments, `requirements.txt`, docstring
      cross-references that pointed at deleted files.
- [x] **5. Rewrite the docs.** `README.md` merges the old root and `refactor/`
      READMEs and gains a History section; `CLAUDE.md` describes one codebase;
      this file.
- [x] **6. Verify.** Tests, smoke, and every shipped config on real data; grepped
      the tree for stale `original/` and `refactor/` paths.
- [x] **7. Local commits.** Two, no push.

---

## Verification commands

```bash
# from the repo root, with the venv from README.md active
python -m pytest tests -q
bash scripts/smoke.sh
```

## Decisions worth knowing

- **The original is gone on purpose,** at the user's request: the framework is a
  superset, so a second frozen copy was dead weight. Its behaviour lives on in
  `cir/`, including the bug fixes it needed to run at all — ETF vertex adjacency
  ordering, the least-squares solver's residual frame, the evolutionary-init
  chain, headless plotting.
- **"Added loss" now means two things, deliberately.** `cir/models/alvae.py` is
  the fixed-basis residual penalty (the refactor's extension);
  `cir.models.alternating.AddedLossVAE` is the original's version — the gap
  against a second *learned* linear decoder. Both are registered and tested; the
  class names keep them apart.
- **ALVAE's auxiliary loss is still a research placeholder.** The mechanism is
  wired up, tested, and documented, but the specific regularizer was never
  specified in the original code — only sketched in comments. Swap the body of
  `ALVAE.auxiliary_loss` when the intended definition is settled.
- **Alternating validation never walks the schedule.** Evaluation always decodes
  through the learned nonlinear decoder, so validation loss is comparable across
  epochs instead of jumping between two paths.
- **MNIST stays committed** under `data/` at one path, so runs work on offline
  nodes. Untracking it entirely is the user's call.
