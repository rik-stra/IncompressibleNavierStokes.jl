#!/bin/bash
#SBATCH -J ANN_search
#SBATCH -t 60:00
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --array=2-6

export JULIA_DEPOT_PATH=/scratch-shared/$USER/.julia_a100:
# julia --project -t auto -e 'using Pkg; Pkg.update()'
julia --project track_ref.jl $SLURM_ARRAY_TASK_ID
