#!/bin/bash
#SBATCH -J precompile
#SBATCH -t 45:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

mkdir -p /scratch-shared/$USER

export JULIA_DEPOT_PATH=/scratch-shared/$USER/.julia_h100:

julia --project -t auto -e 'using Pkg; Pkg.update()'
julia --project exp_square_HIT/precompile.jl