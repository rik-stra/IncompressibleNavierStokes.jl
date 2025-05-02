#!/bin/bash
#SBATCH -J spinnup
#SBATCH -t 1:45:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

export JULIA_DEPOT_PATH=/scratch-shared/$USER/.julia_h100:
# julia --project -t auto -e 'using Pkg; Pkg.update()'
julia --project channel_spinnup_CG.jl
