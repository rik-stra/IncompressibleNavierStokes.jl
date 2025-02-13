#!/bin/bash
#SBATCH -J plot
#SBATCH -t 70:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

export JULIA_DEPOT_PATH=/scratch-shared/$USER/.julia_a100:

julia --project plot_online.jl