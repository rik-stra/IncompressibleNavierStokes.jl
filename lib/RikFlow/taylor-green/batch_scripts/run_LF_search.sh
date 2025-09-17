#!/bin/bash
#SBATCH -J LF_TOLRS
#SBATCH -t 40:00
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --array=1-1

export JULIA_DEPOT_PATH=$HOME/julia/julia_a1003:
julia --project -t auto -e 'using Pkg; Pkg.update()'

julia --project 7_train_LinReg.jl $SLURM_ARRAY_TASK_ID
julia --project 8_TO_online.jl $SLURM_ARRAY_TASK_ID