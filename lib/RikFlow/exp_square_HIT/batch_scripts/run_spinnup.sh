#!/bin/bash
#SBATCH -J HF_sim
#SBATCH -t 0:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
# julia --project -t auto -e 'using Pkg; Pkg.update()'
julia --project 1_spinnup.jl