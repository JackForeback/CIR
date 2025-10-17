#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=170G
#SBATCH --gpus-per-node=quadro_rtx_8000:1
#SBATCH --time=5:00:00
#SBATCH --job-name=perclass
#SBATCH --mail-user=forebacj@mail.gvsu.edu
#SBATCH --mail-type=end,fail

source /mnt/projects/debruinz_project/pytorch-nightly-env/bin/activate

# Pass them into Python as CLI args
python ~/CIR/src/refactor/train.py --config ~/CIR/src/refactor/configs/vae_configs/vae.yaml

deactivate
