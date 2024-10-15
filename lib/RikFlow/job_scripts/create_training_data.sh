#!/bin/bash
#SBATCH -J HF_sim
#SBATCH -t 150:00
#SBATCH -p gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1

module load 2023
module load Julia/1.10.4-linux-x86_64

julia --project experiment/HF_ref.jl
