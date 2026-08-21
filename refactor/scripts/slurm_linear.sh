#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gpus-per-node=quadro_rtx_8000:1
#SBATCH --time=5:00:00
#SBATCH --job-name=cir-linear
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-type=end,fail
#
# Submit with: sbatch scripts/slurm_linear.sh
#
# Set CIR_VENV to your environment before submitting, e.g.
#   CIR_VENV=/mnt/projects/debruinz_project/pytorch-nightly-env sbatch scripts/slurm_linear.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${CIR_VENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$CIR_VENV/bin/activate"
fi

# Run as a module so `cir` resolves from the refactor/ directory. Invoking the
# file by path is what produced the ModuleNotFoundError in earlier job logs.
python -m cir.train --config configs/linear.yaml "$@"
