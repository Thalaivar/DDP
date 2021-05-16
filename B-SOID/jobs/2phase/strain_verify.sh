#!/bin/bash

IFS=$'\n' read -d '' -r -a strains < ./strains.txt

#SBATCH --time 48:00:00
#SBATCH -A laadd
#SBATCH -n ${#strains[@]}
#SBATCH --job-name strain-verify
#SBATCH --output strain-verify.log

JOBNAME=$SLURM_JOB_NAME
N_JOB=$N_TASKS

BASE_DIR=/fastscratch/laadd/strain_verify
mkdir $BASE_DIR

for((i=1;i<=$N_JOB;i++))
do
    mkdir $