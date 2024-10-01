#!/bin/bash
#SBATCH -J HF_sim
#SBATCH -t 45:00
#SBATCH -p gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1

module load 2023
module load Julia/1.10.4-linux-x86_64

julia --project HF_sim.jl 128 64 5000
