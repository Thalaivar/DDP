#!/bin/bash

#SBATCH --time 48:00:00
#SBATCH --job-name strain-verify
#SBATCH --output strain-verify.log
#SBATCH --mem 40G
#SBATCH --cpus-per-task 6
#SBATCH --partition compute
#SBATCH --qos=batch

source /home/laadd/.bashrc
conda activate bsoid 

IFS=$'\n' read -d '' -r -a strains < ./strains.txt

#SBATCH -n ${#strains[@]}

JOBNAME=$SLURM_JOB_NAME
N_JOB=$N_TASKS

BASE_DIR=/fastscratch/laadd/strain_verify
mkdir $BASE_DIR

cd /home/laadd/DDP/B-SOID

for t in ${strains[@]}; do
    strain=${t/\//-}
    mkdir $BASE_DIR/$JOBNAME-$strain
    python scripts.py --script rep_cluster --config ./config/config.yaml --save-dir $BASE_DIR/$JOBNAME-$strain --strain $strain --n 200
done

wait