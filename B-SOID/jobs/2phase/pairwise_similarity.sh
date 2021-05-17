#!/bin/bash
#SBATCH --job-name=bsoid-pwise-strain
#SBATCH --output=strain-pwise-calc.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=72:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=30000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --script calculate_pairwise_similarity --thresh 0.9 --save-dir /home/laadd/data
