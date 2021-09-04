#!/bin/bash
#SBATCH --job-name=eac-run
#SBATCH --output=eac-run-%J.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=normal
#SBATCH --mem=20000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/