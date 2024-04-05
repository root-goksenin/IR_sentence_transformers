#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=LoTTE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:45:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0

module load 2022
module load Anaconda3/2022.05 
source activate ig
cd $HOME/gpl_training_analyzer

datasets_test=(scifact)

python3 ranking_analysis.py "${datasets_test["$SLURM_ARRAY_TASK_ID"]}"
python3 visualize_difference.py "${datasets_test["$SLURM_ARRAY_TASK_ID"]}"