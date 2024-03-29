#!/bin/bash
#SBATCH --job-name=bsoid-strainwise-clustering
#SBATCH --output=strainwise-clustering.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00:00
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --mem=50000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --script strainwise_cluster --config ./config/config.yaml  --logfile ./strainwise_cluster.log