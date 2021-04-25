#!/bin/bash
#SBATCH --job-name=ensemble-test
#SBATCH --output=ensemble-test.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=75000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --config config/2d_umap_config.yaml --script ensemble_pipeline --outdir /home/laadd/data/