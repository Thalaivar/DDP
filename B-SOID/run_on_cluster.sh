#!/bin/bash
#SBATCH --job-name=BSOID-test-run
#SBATCH --output=bsoid_test_run.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=batch
#SBATCH --mem=400000
#SBATCH --mail-user=dhruv.laad@jjax.org
#SBATCH --mail-type=ALL

cd /home/laadd/DDP/B-SOID/
conda activate bsoid
python scripts.py