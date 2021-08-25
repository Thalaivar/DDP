#!/bin/bash
#SBATCH --job-name=final-cluster-ver
#SBATCH --output=final-cluster-ver.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=50000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --script clustering_stability_test --config ./config/config.yaml --save-dir /fastscratch/laadd/stab --thresh 0.85