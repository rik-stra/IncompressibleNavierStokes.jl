#!/bin/bash
#SBATCH -J plot
#SBATCH -t 30:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

export JULIA_DEPOT_PATH=$HOME/julia/julia_a100:
julia --project -t auto -e 'using Pkg; Pkg.update()'

julia --project compute_ks.jl
