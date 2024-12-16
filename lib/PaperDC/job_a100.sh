#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=05:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sda@cwi.nl
# #SBATCH --array=1-1

# Note:
# - gpu_a100: 18 cores
# - gpu_h100: 16 cores
# https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting

mkdir -p /scratch-shared/$USER

echo "Slurm job ID: $SLURM_JOB_ID"
echo "Slurm array task ID: $SLURM_ARRAY_TASK_ID"

export JULIA_DEPOT_PATH=/scratch-shared/$USER/.julia_a100:

cd $HOME/projects/IncompressibleNavierStokes/lib/PaperDC

# julia --project prioranalysis.jl
julia --project -t auto postanalysis.jl

# julia --project -t auto -e 'using Pkg; Pkg.update()'
