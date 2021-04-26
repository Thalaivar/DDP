#!/bin/bash
#SBATCH --job-name=big-umap-tune
#SBATCH --output=big-umap-tune.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=batch
#SBATCH --mem=250000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --script hyperparamter_tuning --config ./config/config.yaml