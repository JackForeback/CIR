#!/bin/bash

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --time=1:00:00
#SBATCH --job-name=even50
#SBATCH --partition=batch
#SBATCH --mem=500mb
#SBATCH --output=/users/jforebac/CIR/cause-tests/even50/id.txt
#SBATCH --error=/users/jforebac/CIR/cause-tests/even50/%j.err

# email when done
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jack_foreback@brown.edu

JOBNAME="even50"
# mkdir /users/jforebac/CIR/cause-tests/$JOBNAME
mkdir /users/jforebac/CIR/cause-tests/$JOBNAME/db

source /users/jforebac/CIR/venv/bin/activate
python LinearClassifier.py

rm -r /users/jforebac/CIR/cause-tests/$JOBNAME/db