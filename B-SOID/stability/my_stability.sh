#!/bin/bash
#SBATCH --job-name=my-test-%j
#SBATCH --output=my-test-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=5:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=50000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 
cd /home/laadd/DDP/B-SOID/stability
