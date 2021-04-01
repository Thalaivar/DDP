#!/bin/bash
#SBATCH --job-name=BSOID-small
#SBATCH --output=small-training.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=batch
#SBATCH --mem=100000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python -c 'from scripts import smaller_umap; smaller_umap()'