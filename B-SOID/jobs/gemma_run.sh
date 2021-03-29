#!/bin/bash
#SBATCH --job-name=BSOID-gemma
#SBATCH --output=gemma.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=400000
#SBATCH --mail-user=dhruv.laad@jax.org
#SBATCH --mail-type=ALL

cd /home/laadd/DDP/B-SOID/

module load singularity

nextflow run TheJacksonLaboratory/mousegwas --yaml gemma_config.yaml --shufyaml gemma_shuffle.yaml --input gemma_input.csv --outdir /home/laadd/gemma_output -profile slurm,singularity --addgwas " -d 10 "
