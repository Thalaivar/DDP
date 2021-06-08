#!/bin/bash
#SBATCH --job-name=bsoid-test-%j
#SBATCH --output=bsoid-test-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=batch
#SBATCH --mem=30000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 
cd /home/laadd/DDP/B-SOID/stability
