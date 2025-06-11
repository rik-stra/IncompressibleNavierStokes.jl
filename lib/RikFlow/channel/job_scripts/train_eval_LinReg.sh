#!/bin/bash
#SBATCH -J LinReg_search
#SBATCH -t 3:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --array=1-10

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:

julia --project train_LinReg.jl $SLURM_ARRAY_TASK_ID
julia --project channel_online_SGS.jl $SLURM_ARRAY_TASK_ID
