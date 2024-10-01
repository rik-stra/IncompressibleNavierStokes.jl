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
simdir = $pwd
mkdir $TMPDIR/output
cd $TMPDIR

julia --project $simdir 64 128 5000

mkdir $simdir/output
cp -r $TMPDIR/output/* $simdir/output/.