#!/bin/bash
#SBATCH -J spinnup
#SBATCH -t 45:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

module load 2023
module load Julia/1.10.4-linux-x86_64
export JULIA_DEPOT_PATH=/scratch-shared/$USER/.julia_h100:
julia --project exp_square_HIT/spinnup.jl
