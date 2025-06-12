#!/bin/bash

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --time=5:00:00
#SBATCH --job-name=base
#SBATCH --partition=batch
##SBATCH --mem=1G
#SBATCH --output=/users/jforebac/CIR/cause-tests/base/id.txt
#SBATCH --error=/users/jforebac/CIR/cause-tests/base/%j.err

# email when done
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jack_foreback@brown.edu

JOBNAME="base"
mkdir /users/jforebac/CIR/cause-tests/$JOBNAME

source /users/jforebac/CIR/venv/bin/activate
python LinearClassifier.py
