# CIR — Class Independent Regularizer

Research code exploring whether the **per-class** convergence of a classifier can
be equalized by geometry rather than by reweighting the loss.

The premise: when class clusters sit at unequal distances from the origin, a
linear classifier learns them at different rates, and some classes converge long
before others. If the class means are projected onto an evenly-spaced
configuration — a simplex equiangular tight frame (ETF) — every class presents
the same geometry to the optimizer and should converge together. The `linear`
experiment tests exactly that on 2D Gaussian clusters; the VAE experiments carry
the same fixed-basis machinery over to learned representations.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

CPU wheels are sufficient; nothing here requires a GPU. MNIST is committed under
`data/`, so runs work on offline compute nodes with no download step.

## Running

Every experiment is a YAML config plus a class. Pick a config:

```bash
python -m cir.train --config configs/linear.yaml
python -m cir.train --config configs/vae.yaml
python -m cir.train --config configs/alvae.yaml
python -m cir.train --config configs/altvae.yaml
```

Override anything from the command line. Values are parsed as YAML, and dotted
keys reach into nested mappings:

```bash
python -m cir.train --config configs/linear.yaml \
  --override num_seeds=1 lr=1e-3 apply_projection=true flags.evo_weights=true
```

Run it as a **module**, from the repository root. Invoking `python cir/train.py`
puts `cir/` rather than the repository root on the path, which is what produced
`ModuleNotFoundError: No module named 'utils'` in the old job logs.

On a cluster: `sbatch scripts/slurm_linear.sh` (also `slurm_vae.sh`,
`slurm_alvae.sh`, `slurm_altvae.sh`). Point `CIR_VENV` at your environment first. Extra arguments
pass straight through, so `sbatch scripts/slurm_linear.sh --override num_seeds=50`
works.

## Verifying

```bash
python -m pytest tests -q     # 109 tests, ~18s
bash scripts/smoke.sh         # every experiment end to end on CPU, ~40s
```

Both are CPU-only and headless (`matplotlib` uses the `Agg` backend), so they
need no GPU and no display.

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
    alvae.py          VAE + a fixed-basis residual penalty
    alternating.py    the alternating-decoder variants
  models/
    linear_classifier.py
    vae.py            Encoder / Decoder / VAE, widths from the config
    alvae.py          ALVAE: fixed-basis residual penalty
    alternating.py    AlternatingVAE / LAVAE / FOLVAE / AddedLossVAE
    basis.py          BasisLinear / DCTLinear / ChebyshevLinear
  data/mnist.py       MNIST loaders over the repository's committed copy
  utils/
    geometry.py       ETF construction, distortion, and the projections
    losses.py         fairness-regularized MSE variants
    metrics.py        AccuracyTracker: per-class accuracy across seeds and steps
    evolution.py      tournament search for a low-entropy initialization
    plotting.py       figures, all taking an explicit output directory
    solvers.py        iterative Chebyshev least-squares via Householder QR
configs/              linear.yaml, vae.yaml, alvae.yaml, altvae.yaml
scripts/              smoke.sh and the SLURM submitters
tests/                109 tests
data/                 committed MNIST
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

### `vae` — the baseline autoencoder

A straightforward MLP VAE on MNIST, with widths, depth, and activation all read
from the config.

### `alvae` — a fixed basis in the objective

Adds one term: the reconstruction is projected onto a fixed orthonormal basis
(DCT or Chebyshev) and back, and whatever the round trip loses — energy the basis
cannot represent — is penalized at weight `aux_weight`. This pushes
reconstructions toward the smooth, low-order subspace that
`cir.utils.solvers.iterative_chebyshev_ls` can solve directly.

### `altvae` — a second, purely linear decoder

Gives the VAE a linear decode path (no activations anywhere) alongside the
learned one, so its reconstruction lands in a subspace a least-squares solve can
reach. Pick a variant with `variant:`:

| `variant` | What it does |
|-----------|--------------|
| `lavae` | Alternate the two paths on a schedule; both are learned. The baseline. |
| `folvae` | The same, with the linear path's output layer frozen — a stand-in for *solving* that map rather than learning it. |
| `added_loss` | Always decode nonlinearly, and penalize how far the result sits from what the linear path would produce. |

> **Note.** `alvae` and `altvae`'s `added_loss` are two different answers to one
> question. The original code sketched the second (a second *learned* linear
> decoder) and left the regularizer itself unspecified; `alvae` is the later,
> fixed-basis answer. Both are kept, registered, and tested.

## Adding an experiment

1. Subclass `BaseExperiment` in `cir/experiments/`, implementing `build_model`,
   `get_dataloaders`, and `compute_loss`. Override `run` only if your training
   loop is not epoch-shaped — `LinearExperiment` does, because it takes
   full-batch steps across many seeds.
2. Add one line to `EXPERIMENTS` in `cir/experiments/registry.py`.
3. Add a `configs/<name>.yaml` with a matching `experiment:` key. A registered
   experiment without a config fails the test suite.
4. Add a case to `scripts/smoke.sh`.

Keep configuration in YAML rather than in Python, and keep paths relative to the repo.
