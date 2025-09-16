#!/bin/bash
#SBATCH -J HF_sim
#SBATCH -t 40:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

export JULIA_DEPOT_PATH=$HOME/julia/julia_h1002:
julia --project -t auto -e 'using Pkg; Pkg.update()'
julia --project 1_TG_HF.jl