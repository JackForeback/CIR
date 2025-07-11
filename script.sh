#!/bin/bash

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --time=5:00:00
#SBATCH --job-name=multfile
#SBATCH --partition=batch
##SBATCH --mem=500mb
#SBATCH --output=/users/jforebac/CIR/cause-tests/multfile/id.txt
#SBATCH --error=/users/jforebac/CIR/cause-tests/multfile/%j.err

# email when done
##SBATCH --mail-type=end,fail
##SBATCH --mail-user=jack_foreback@brown.edu

JOBNAME="/users/jforebac/CIR/cause-tests/multfile"
# mkdir /users/jforebac/CIR/cause-tests/$JOBNAME
mkdir $JOBNAME/db
mkdir $JOBNAME/ani
mkdir $JOBNAME/seed

source /users/jforebac/CIR/venv/bin/activate
python LinearClassifier.py

rm -r $JOBNAME/db