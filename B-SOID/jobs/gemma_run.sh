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
BASE_DIR=DDP/B-SOID/gemma
OUT_DIR=/fastscratch/laadd/gemma
mkdir $OUT_DIR

source $HOME_DIR/.bashrc
cd $OUT_DIR

CONFIG_FILE=$HOME_DIR/$BASE_DIR/gemma_config.yaml   
SHUFFL_FILE=$HOME_DIR/$BASE_DIR/gemma_shuffle.yaml
INPUT_FILE=$HOME_DIR/$BASE_DIR/gemma_data.csv

module load singularity
singularity cache clean

nextflow run TheJacksonLaboratory/mousegwas --yaml $CONFIG_FILE --shufyaml $SHUFFL_FILE --input $INPUT_FILE --outdir $OUT_DIR/gemma_output -profile slurm,singularity --addgwas " -d 10 "