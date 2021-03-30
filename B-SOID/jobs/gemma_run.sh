#!/bin/bash
#SBATCH --job-name=BSOID-gemma
#SBATCH --output=gemma.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=8000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL


HOME_DIR=/home/laadd
BASE_DIR=DDP/B-SOID

source $HOME_DIR/.bashrc
cd /fastscratch

CONFIG_FILE=$HOME_DIR/$BASE_DIR/gemma_config.yaml
SHUFFL_FILE=$HOME_DIR/$BASE_DIR/gemma_shuffle.yaml
INPUT_FILE=$HOME_DIR/$BASE_DIR/filt_gemma_input.csv

module load singularity
nextflow run TheJacksonLaboratory/mousegwas --yaml $CONFIG_FILE --shufyaml $SHUFFL_FILE --input $INPUT_FILE --outdir $HOME_DIR/gemma_output -profile slurm,singularity --addgwas " -d 10 "