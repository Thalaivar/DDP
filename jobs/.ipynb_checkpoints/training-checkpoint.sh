#!/bin/bash
#SBATCH --job-name=laadd-train
#SBATCH --output=training.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --mem=50000

source /home/laadd/.bashrc
conda activate bsoid 

cd /home/laadd/DDP
python run_pipeline.py --config config.yaml --name experiment1 --n 10 --data-dir /projects/kumar-lab/StrainSurveyPoses/ --records /projects/kumar-lab/StrainSurveyPoses/StrainSurveyMetaList_2019-04-09.tsv