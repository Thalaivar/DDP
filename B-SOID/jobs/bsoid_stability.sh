#!/bin/bash
#SBATCH --job-name=bsoid-test-%j
#SBATCH --output=bsoid-test-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=batch
#SBATCH --mem=30000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

source /home/laadd/.bashrc
conda activate bsoid 

BASE_DIR=/fastscratch/laadd/bsoid_stability
RUN_NAME=$1
SAVE_DIR=$BASE_DIR/$RUN_NAME
mkdir $SAVE_DIR

cd /home/laadd/DDP/B-SOID/
python scripts.py --script bsoid_stability_test --config ./config/config.yaml --save-dir $SAVE_DIR --num-points 500000