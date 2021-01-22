#!/bin/bash
#SBATCH --job-name=BSOID-training
#SBATCH --output=training.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=20000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/

# python active_inactive_split.py
python scripts.py
