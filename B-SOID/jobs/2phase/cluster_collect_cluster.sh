#!/bin/bash
#SBATCH --job-name=bsoid-ccc
#SBATCH --output=bsoid-ccc.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=50000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --script cluster_collect_embed --max-samples 1500 --thresh 0.6