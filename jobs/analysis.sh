#!/bin/bash
#SBATCH --job-name=BSOID-test
#SBATCH --output=test.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=40000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/
git checkout fixing

cd B-SOID/
python scripts.py --config ./config/config.yaml --script rep_cluster