# `refactor/` — the config-driven framework

The current codebase. Experiments are YAML configs plus a class; everything
shared — device selection, seeding, optimizer and loss construction, logging,
output directories — lives in one base class.

The pre-refactor version is preserved at [`../original/`](../original/) and is
not maintained.

## Running

Set up the environment from the [repository root](../README.md#setup), then:

```bash
cd refactor
python -m cir.train --config configs/linear.yaml
python -m cir.train --config configs/vae.yaml
python -m cir.train --config configs/alvae.yaml
```

Override anything from the command line. Values are parsed as YAML, and dotted
keys reach into nested mappings:

```bash
python -m cir.train --config configs/linear.yaml \
  --override num_seeds=1 lr=1e-3 apply_projection=true flags.evo_weights=true
```

Run it as a **module**, from this directory. Invoking `python cir/train.py` puts
`cir/` rather than `refactor/` on the path, which is what produced
`ModuleNotFoundError: No module named 'utils'` in the pre-refactor job logs.

On a cluster: `sbatch scripts/slurm_linear.sh` (also `slurm_vae.sh`,
`slurm_alvae.sh`). Point `CIR_VENV` at your environment first. Extra arguments
pass straight through, so `sbatch scripts/slurm_linear.sh --override num_seeds=50`
works.

## Verifying

```bash
python -m pytest tests -q     # 87 tests, ~10s
bash scripts/smoke.sh         # every experiment end to end on CPU, ~30s
```

Both are CPU-only and headless.

## Layout

```
cir/
  train.py            entry point: config -> registry -> run
  cli.py              argument parsing, YAML loading, --override handling
  logging_utils.py    SimpleLogger: one JSON record per call, plus a config snapshot
  experiments/
    base.py           BaseExperiment: the contract and all the shared machinery
    registry.py       name -> class, read from the config's `experiment:` key
    linear.py         the linear-classifier fairness experiment
    vae.py            VAE reconstruction on MNIST
    alvae.py          ALVAE reconstruction on MNIST
  models/
    linear_classifier.py
    vae.py            Encoder / Decoder / VAE, widths from the config
    alvae.py          VAE + a fixed-basis residual penalty
    basis.py          BasisLinear / DCTLinear / ChebyshevLinear
  data/mnist.py       MNIST loaders over the repository's committed copy
  utils/
    geometry.py       ETF construction, distortion, and the projections
    losses.py         fairness-regularized MSE variants
    metrics.py        AccuracyTracker: per-class accuracy across seeds and steps
    evolution.py      tournament search for a low-entropy initialization
    plotting.py       figures, all taking an explicit output directory
    solvers.py        iterative Chebyshev least-squares via Householder QR
configs/              linear.yaml, vae.yaml, alvae.yaml
scripts/              smoke.sh and the SLURM submitters
tests/                87 tests
```

## The experiments

### `linear` — the project's central question

Place the class means on a regular polygon (a 2D simplex ETF), break the symmetry
with per-class `scalars` and `rotations`, draw Gaussian clusters around the
result, and train a linear classifier. The metric is the **max-minus-min per-class
accuracy gap** at each step: if geometry drives unequal convergence, distorted
means should show a persistent gap.

Three interventions, each independently switchable:

| Config | Intervention |
|--------|--------------|
| `apply_projection: true` | Warp the *data* back toward the ETF each step, at strength `1 - mean_accuracy` so the correction fades as the model learns. |
| `flags.fairness_loss` | Penalize the between-class gap in the *objective* — `per_class_gap` (per-class MSE) or `soft_accuracy_gap` (per-class confidence). |
| `flags.evo_weights` | Search for a low-entropy *initialization* by tournament selection instead of sampling one. |

Writes `sample_plot.png`, `avg_accuracy.png`, `avg_gap.png`, per-seed gap plots,
and — with `flags.plot_boundaries` — a decision-boundary GIF per seed.

### `vae` and `alvae`

`vae` is a straightforward MLP VAE on MNIST, with widths, depth, and activation
all read from the config.

`alvae` ("Added-Loss VAE") adds one term: the reconstruction is projected onto a
fixed orthonormal basis and back, and whatever the round trip loses — energy the
basis cannot represent — is penalized at weight `aux_weight`. This pushes
reconstructions toward the smooth, low-order subspace that
`cir.utils.solvers.iterative_chebyshev_ls` can solve directly.

> **Note.** The *mechanism* here (fixed basis, residual penalty, config-driven
> weight) is what the original code specified, in `original/models.py`'s
> `FOLVAE` / `LAVAE` / `ALVAE` sketches. The precise regularizer was never
> written down — only sketched in comments. Override `ALVAE.auxiliary_loss` to
> substitute a different definition; nothing else in the pipeline changes.

## Adding an experiment

1. Subclass `BaseExperiment` in `cir/experiments/`, implementing `build_model`,
   `get_dataloaders`, and `compute_loss`. Override `run` only if your training
   loop is not epoch-shaped — `LinearExperiment` does, because it takes
   full-batch steps across many seeds.
2. Add one line to `EXPERIMENTS` in `cir/experiments/registry.py`.
3. Add a `configs/<name>.yaml` with a matching `experiment:` key.
4. Add a case to `scripts/smoke.sh`.

Keep configuration in YAML rather than in Python, and keep paths relative to the
repository — no absolute `/mnt/home/...` paths, which is what tied the previous
version to one user's cluster account.
