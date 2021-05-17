#!/bin/bash
#SBATCH -n 5
#SBATCH --job-name=bsoid-pwise-strain
#SBATCH --output=strain-pwise-calc.txt
#SBATCH --time=72:00:00
#SBATCH --partition=high_mem
#SBATCH --qos=batch
#SBATCH --cpus-per-task 18
#SBATCH --mem=50000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

JOBNAME=$SLURM_JOB_NAME
N_JOB=$N_TASKS

BASE_DIR=/fastscratch/laadd/pwise_strain
mkdir $BASE_DIR

sim_measures=("dbcv_index_similarity" "density_separation_similarity" "roc_similiarity" "minimum_distance_similarity" "hausdorff_similarity")

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP/B-SOID/

for measure in "${sim_measures[@]}"; do
    mkdir $BASE_DIR/$JOBNAME-$measure
    srun --exclusive --ntasks 1 python scripts.py --script calculate_pairwise_similarity --thresh 0.9 --save-dir $BASE_DIR/$JOBNAME-$measure --sim-measure $measure
done

wait