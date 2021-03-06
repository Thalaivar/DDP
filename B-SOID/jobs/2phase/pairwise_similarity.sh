#!/bin/bash
#SBATCH --job-name=bsoid-pwise-strain
#SBATCH --output=strain-pwise-calc.txt
#SBATCH --time=72:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=batch
#SBATCH --ntasks 5
#SBATCH --cpus-per-task 18
#SBATCH --mem=70000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

JOBNAME=$SLURM_JOB_NAME
N_JOB=$N_TASKS

BASE_DIR=/fastscratch/laadd/pwise_strain
mkdir $BASE_DIR

SIM_MEASURE="dbcv_index_similarity"
SAVE_DIR=$BASE_DIR/$SIM_MEASURE
mkdir $SAVE_DIR

# sim_measures=("dbcv_index_similarity" "density_separation_similarity" "roc_similiarity" "minimum_distance_similarity" "hausdorff_similarity")

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/
python scripts.py --script calculate_pairwise_similarity --thresh 0.9 --save-dir $SAVE_DIR --sim-measure $SIM_MEASURE