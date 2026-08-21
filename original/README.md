# `original/` — the pre-refactor codebase

The historical implementation, kept so results produced before the refactor stay
reproducible. Its design is **frozen**: module-level state, `--key value`
arguments parsed by hand, one script per experiment. That style is the point —
this is the reference, not the codebase to build on. New work goes in
[`../refactor/`](../refactor/).

It has been bug-fixed and documented, not re-architected. See
[what changed](#what-was-fixed) below.

## Layout

| File | What it is |
|------|------------|
| `LinearClassifier.py` | Driver script for the linear-classifier fairness experiment. |
| `vae.py` | Driver script for VAE reconstruction on MNIST. |
| `models.py` | `LinearClassifier`, `VAE`, and the three exploratory VAE variants. |
| `utils.py` | Argument parsing, data generation, fairness losses, evolutionary init, accuracy tracking, ETF geometry. |
| `plotting.py` | Sample scatter, decision boundaries, accuracy curves, GIF animation. |
| `method.py` | Iterative Chebyshev least-squares via Householder QR. |
| `funcs.py` | Householder primitives used by `method.py`. |
| `scripts/` | SLURM/local run scripts and a fast smoke test. |

## Running it

Set up the environment from the [repository root](../README.md#setup), then:

```bash
cd original
bash scripts/smoke.sh        # tiny end-to-end run of everything, ~20s
bash scripts/run_linear.sh   # the real linear experiment, ~40s
bash scripts/run_vae.sh      # one epoch of MNIST VAE, ~10s
```

On a cluster, `sbatch scripts/run_linear.sh`. Point `CIR_VENV` at your
environment first; the scripts source it if set and otherwise use whatever
`python` is on `PATH`.

Configuration lives in the shell scripts (sample counts, seeds, steps) and in the
flag block near the top of `LinearClassifier.py`:

```python
per_class_gap = False        # penalize the gap in per-class MSE
soft_accuracy_gap = False    # penalize the gap in per-class softmax confidence
apply_projection = False     # warp the data toward the ETF each step
use_evo_weights = False      # search for a low-entropy initialization
plot_boundaries = True       # save a frame per step and build a GIF (slow)
```

## What the experiment does

`make_evenly_spaced_targets` places the class means on the vertices of a regular
polygon — a 2D simplex ETF. The script then breaks that symmetry by scaling one
cluster outward, draws Gaussian samples around each mean, and trains a linear
classifier with MSE on one-hot targets. `track_accuracy` records per-class
accuracy at every step, and `seed_plot` charts the max-minus-min gap between
classes, which is the quantity of interest: with equal geometry, classes should
converge together.

Output lands under `--path`: `sample_plot.png`, `avg_accuracy_graph.png`,
`seed/avg_diff.png`, per-seed gap plots, and `ani/BA-seed:N.gif`.

## What was fixed

The design is untouched; these are the changes that were needed to make it run.

**Crashes**
- `utils.py` and `plotting.py` read a global `path` from `sys.argv[2]` at import
  time, so importing either module raised unless the caller passed `--path`
  first. `path` is now a parameter of the functions that write files.
- `evo_weights` was called with five arguments and defined with five, but
  `manage_population` computed `pop_size / tournament_size` (always a float) and
  then bailed out on `isinstance(num_groups, float)` — so the function exited the
  process every time. `mutate` took one argument and was called with two, and
  both it and `manage_population` called `torch.randint` without the required
  `size`. The whole chain has been rewritten around `init_population` /
  `eval_pop` / `mutate`, preserving the tournament-selection design.
- `FOLVAE`, `LAVAE`, and `ALVAE` each called `super(VAE, self).__init__()` from a
  sibling class, unpacked the encoder's single output as a `(mean, log_var)`
  pair, and called an undefined `solver`. None could be instantiated. They now
  share an `AlternatingVAE` base and run.
- `compute_H` divided by a `(1, 1)` array, which NumPy 2 refuses to coerce.

**Correctness**
- `make_evenly_spaced_targets` sorted its vertices by position, which interleaves
  opposite sides of the polygon. Since `is_regular_polygon` measures the distance
  between *consecutive* points, a genuine ETF was reported as irregular for five
  or more classes. The vertices are now rotated into place, preserving adjacency.
- `is_regular_polygon`'s default tolerance of `1e-9` is finer than float32
  arithmetic at these magnitudes; it is now `1e-4`.
- `solver`'s convergence check compared `bcheck` (which is `Qᵀb`) against
  `Y.dot(z)` (which is not transformed). The two lived in different frames, so
  the check never fell below `tol` and the loop always ran to its iteration cap.
  It now measures the residual in the original system — a smooth test problem
  that used to take 20 iterations converges in 10.
- `np.linalg.lstsq` was called without `rcond`, which is deprecated.

**Practicality**
- Matplotlib is pinned to the `Agg` backend so runs work headless.
- `plot_decision_boundaries` compared every sample against every class vector
  with `torch.equal`, making each animation frame O(N × classes) Python-level
  calls; it now uses a vectorized mask. Its legend, which had one entry per
  pairwise boundary, is gone.
- The fairness losses printed every tensor they touched on every step.
- Output directories are created by the script rather than assumed to exist.
