#!/bin/bash
#SBATCH -J HFsim
#SBATCH -t 6:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
julia --project -t auto -e 'using Pkg; Pkg.update()'
julia --project channel_HF.jl
