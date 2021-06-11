#!/bin/bash
#SBATCH --job-name=laadd-stab
#SBATCH --qos=training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --mem=50000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --output=laadd-stab-%j.out
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 
cd /home/laadd/DDP/B-SOID/stability
