#!/bin/bash
#SBATCH --job-name=bsoid-ccc-exem
#SBATCH --output=bsoid-ccc-exem.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=5:00:00
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --mem=60000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --script cluster_collect_embed --config ./config/config.yaml --save-dir /fastscratch/laadd/cce_test1 --thresh 0.85