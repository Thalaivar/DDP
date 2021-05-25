#!/bin/bash
#SBATCH --job-name=bsoid-ccc-subs
#SBATCH --output=bsoid-ccc-subs.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=5:00:00
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --mem=30000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --script cluster_collect_embed --config ./config/config.yaml --save-dir /fastscratch/laadd/cce-subs --thresh 0.85