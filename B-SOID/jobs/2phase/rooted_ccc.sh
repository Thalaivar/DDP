#!/bin/bash
#SBATCH --job-name=bsoid-ccc-root
#SBATCH --output=bsoid-ccc-root.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=batch
#SBATCH --mem=60000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --script rooted_final_clustering --config ./config/config.yaml --save-dir /fastscratch/laadd/cce-subs --thresh 0.85