#!/bin/bash
#SBATCH --job-name=BSOID-analysis
#SBATCH --output=analysis.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=8000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/
git checkout analysis_new

cd B-SOID/
python analysis.py