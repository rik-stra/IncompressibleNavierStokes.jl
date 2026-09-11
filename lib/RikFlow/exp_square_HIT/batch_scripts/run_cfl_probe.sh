#!/bin/bash
#SBATCH -J cfl_probe
#SBATCH -t 00:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

# Adaptive-timestep probe of the Float64 / LMWray3 configuration at 512^3.
# Short by design: a few hundred steps is enough to see what INS chooses for Δt and whether
# LMWray3 survives at it. Read the printed "mean Δt / production" ratio and the stable/unstable
# verdict; the full Δt history is in output/cfl_probe_512_f64_lmwray3.jld2.

export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:

cd ..
julia --project cfl_probe.jl
