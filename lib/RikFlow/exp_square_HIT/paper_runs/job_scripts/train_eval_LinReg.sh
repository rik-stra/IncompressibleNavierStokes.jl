#!/bin/bash
#SBATCH -J ANN_search
#SBATCH -t 60:00
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --array=1-1

export JULIA_DEPOT_PATH=/scratch-shared/$USER/.julia_a100:

julia --project train_LinReg.jl $SLURM_ARRAY_TASK_ID
julia --project online_sgs.jl $SLURM_ARRAY_TASK_ID