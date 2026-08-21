#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --mem=8G
#SBATCH --gpus-per-node=tesla_v100s:1
#SBATCH --time=5:00:00
#SBATCH --job-name=perclass
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-type=end,fail
#
# Original linear-classifier experiment. Edit the variables below, then:
#   bash scripts/run_linear.sh          (locally)
#   sbatch scripts/run_linear.sh        (on the cluster)
#
# Set CIR_VENV to your environment before submitting, e.g.
#   CIR_VENV=/mnt/projects/debruinz_project/pytorch-nightly-env sbatch scripts/run_linear.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${CIR_VENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$CIR_VENV/bin/activate"
fi

# --- just set variables here ---
PATH_ARG="${PATH_ARG:-output}"
NUM_CLASSES=3
NUM_TRAINING_STEPS=50
NUM_SEEDS=1
INPUT_DIM=2
SAMPLES_PER_CLASS=10000
TRAIN_RATIO=0.7
# -------------------------------

mkdir -p "$PATH_ARG/db" "$PATH_ARG/ani" "$PATH_ARG/seed"

# Pass them into Python as CLI args
python LinearClassifier.py \
  --path "$PATH_ARG" \
  --num_classes "$NUM_CLASSES" \
  --num_training_steps "$NUM_TRAINING_STEPS" \
  --num_seeds "$NUM_SEEDS" \
  --input_dim "$INPUT_DIM" \
  --samples_per_class "$SAMPLES_PER_CLASS" \
  --train_ratio "$TRAIN_RATIO"

# The per-step boundary frames are large and only exist to build the GIF.
rm -rf "$PATH_ARG/db"
