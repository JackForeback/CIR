#!/usr/bin/env bash
# Tiny end-to-end run of both original scripts, for checking they still work.
#
# Uses small sample counts and a single epoch so the whole thing finishes in
# seconds. The real runs go through run_linear.sh / run_vae.sh.
#
# Usage: bash scripts/smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${SMOKE_OUT:-output/smoke}"
rm -rf "$OUT"
mkdir -p "$OUT/db" "$OUT/ani" "$OUT/seed"

echo "=== LinearClassifier.py ==="
python LinearClassifier.py \
  --path "$OUT" \
  --num_classes 3 \
  --num_training_steps 5 \
  --num_seeds 2 \
  --input_dim 2 \
  --samples_per_class 200 \
  --train_ratio 0.7

echo
echo "=== models.py (all VAE variants construct and run a forward pass) ==="
python - <<'PY'
import torch
from models import VAE, FOLVAE, LAVAE, ALVAE

x = torch.rand(4, 32)
for cls in (VAE, FOLVAE, LAVAE, ALVAE):
    model = cls(32, 4)
    out = model(x) if cls is VAE else model(x, current_step=1)
    print(f"  {cls.__name__:<8} -> {len(out)} outputs, x_hat {tuple(out[0].shape)}")
    out = model(x) if cls is VAE else model(x, current_step=2)  # the alternate branch
PY

echo
echo "=== method.py (Chebyshev least-squares solver) ==="
python - <<'PY'
import numpy as np
from method import solver

n = 40
A = np.eye(n)
b = np.sin(np.linspace(-1, 1, n) * np.pi)
z = solver(A, b)
print(f"  solved, {len(z)} coefficients")
PY

echo
echo "=== vae.py (one epoch on MNIST) ==="
bash scripts/run_vae.sh

echo
echo "All original scripts ran. Output under $OUT/"
