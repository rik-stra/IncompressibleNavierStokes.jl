#!/bin/bash
#SBATCH -J HF_sim
#SBATCH -t 30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
# julia --project -t auto -e 'using Pkg; Pkg.update()'
julia --project HF_HIT.jl
julia --project HF_channel.jl