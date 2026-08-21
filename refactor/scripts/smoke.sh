#!/usr/bin/env bash
# Fast end-to-end run of every registered experiment.
#
# Config values are shrunk via --override so the whole suite finishes in well
# under a minute on CPU. This checks that each experiment *runs*, not that it
# learns anything; the real runs use the configs unmodified.
#
# Usage: bash scripts/smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${SMOKE_OUT:-runs/smoke}"
rm -rf "$OUT"

echo "=== linear ==="
python -m cir.train --config configs/linear.yaml --override \
  samples_per_class=200 num_training_steps=5 num_seeds=2 \
  "output_dir=$OUT/linear" "log_dir=$OUT/linear"

echo "=== linear (projection + fairness loss + evolutionary init) ==="
python -m cir.train --config configs/linear.yaml --override \
  samples_per_class=200 num_training_steps=5 num_seeds=1 \
  apply_projection=true projection_mode=shift \
  flags.fairness_loss=per_class_gap flags.evo_weights=true \
  evo.pop_size=100 evo.tournament_size=50 \
  "output_dir=$OUT/linear_projected" "log_dir=$OUT/linear_projected"

echo "=== vae ==="
python -m cir.train --config configs/vae.yaml --override \
  epochs=1 train_subset=512 test_subset=256 log_every=0 \
  "output_dir=$OUT/vae" "log_dir=$OUT/vae"

echo "=== alvae ==="
python -m cir.train --config configs/alvae.yaml --override \
  epochs=1 train_subset=512 test_subset=256 log_every=0 \
  "output_dir=$OUT/alvae" "log_dir=$OUT/alvae"

echo
echo "All experiments ran. Output under $OUT/"
