#!/bin/bash
#SBATCH -J plot_spinnup
#SBATCH -t 20:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
# julia --project -t auto -e 'using Pkg; Pkg.update()'
julia --project plot_spinnup_output.jl