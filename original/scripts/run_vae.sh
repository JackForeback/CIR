#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gpus-per-node=tesla_v100s:1
#SBATCH --time=5:00:00
#SBATCH --job-name=vae
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-type=end,fail
#
# Original VAE reconstruction script.
#   bash scripts/run_vae.sh       (locally)
#   sbatch scripts/run_vae.sh     (on the cluster)
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${CIR_VENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$CIR_VENV/bin/activate"
fi

python vae.py \
  --epochs 1 \
  --batch_size 64 \
  --latent_dim 16 \
  --lr 0.01
