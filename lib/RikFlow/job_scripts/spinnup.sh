#!/bin/bash
#SBATCH -J spinnup
#SBATCH -t 120:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

export JULIA_DEPOT_PATH=/scratch-shared/$USER/.julia_h100:
julia --project exp_square_HIT/spinnup.jl
