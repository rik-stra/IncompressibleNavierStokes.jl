#!/bin/bash
#SBATCH -J LinReg_search
#SBATCH -t 2:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --array=11-14

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:

julia --project train_LinReg.jl $SLURM_ARRAY_TASK_ID
julia --project channel_online_SGS.jl $SLURM_ARRAY_TASK_ID
