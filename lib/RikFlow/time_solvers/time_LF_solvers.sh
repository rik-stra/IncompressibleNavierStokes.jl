#!/bin/bash
#SBATCH -J LF_sims
#SBATCH -t 40:00
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1

export JULIA_DEPOT_PATH=$HOME/julia/julia_A100:
julia --project -t auto -e 'using Pkg; Pkg.update()'
julia --project LF_HIT.jl
julia --project LF_channel.jl