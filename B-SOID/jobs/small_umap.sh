#!/bin/bash
#SBATCH --job-name=2d-umap
#SBATCH --output=2d-umap.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=batch
#SBATCH --mem=100000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --config config/2d_umap_config.yaml --script small_umap --outdir /home/laadd/data/ --n 10
