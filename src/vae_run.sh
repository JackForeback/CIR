#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=170G
#SBATCH --gpus-per-node=tesla_v100s:1
#SBATCH --time=5:00:00
#SBATCH --job-name=vae
#SBATCH --output=/mnt/home/forebacj/CIR/src/ml-job%j.out
#SBATCH --error=/mnt/home/forebacj/CIR/src/ml-job%j.err
#SBATCH --mail-user=forebacj@mail.gvsu.edu
##SBATCH --mail-type=end,fail

source /mnt/home/forebacj/CIR/venv/bin/activate

# --- just set variables here ---
# PATH_ARG="/mnt/home/forebacj/CIR/perclass"
# NUM_CLASSES=3
# NUM_TRAINING_STEPS=50
# NUM_SEEDS=1
# INPUT_DIM=2
# SAMPLES_PER_CLASS=10000
# TRAIN_RATIO=0.7
# -------------------------------

# mkdir -p "$PATH_ARG/db" "$PATH_ARG/ani" "$PATH_ARG/seed"

# Pass them into Python as CLI args
python vae.py \
  # --path $PATH_ARG \
  # --num_classes "$NUM_CLASSES" \
  # --num_training_steps "$NUM_TRAINING_STEPS" \
  # --num_seeds "$NUM_SEEDS" \
  # --input_dim "$INPUT_DIM" \
  # --samples_per_class "$SAMPLES_PER_CLASS" \
  # --train_ratio "$TRAIN_RATIO"


# rm -r $PATH_ARG/db
deactivate
