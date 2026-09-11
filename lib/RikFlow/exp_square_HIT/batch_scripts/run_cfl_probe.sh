#!/bin/bash
#SBATCH -J cfl_probe
#SBATCH -t 00:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

# Adaptive-timestep probe of the Float64 / LMWray3 configuration at 512^3.
#
# Submit from the RikFlow folder:
#     sbatch exp_square_HIT/batch_scripts/run_cfl_probe.sh
# SLURM runs the job in the directory sbatch was called from, so the script path below is relative
# to lib/RikFlow. The probe's own output path comes from @__DIR__, not the working directory, so it
# lands in exp_square_HIT/output/ either way.
#
# Needs exp_square_HIT/output/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2. If it is
# missing the script stops and prints the path rather than inventing a field. Note 1_spinnup.jl
# writes to output_spinnup/, so a spin-up produced on the cluster has to be moved, not just run.
#
# Short by design: a few hundred steps is enough to see what INS chooses for Δt and whether
# LMWray3 survives it. Read the printed "mean Δt / production" ratio and the stable/unstable
# verdict; the full Δt history and the QoIs are in output/cfl_probe_512_f64_lmwray3.jld2.

# Reuse the existing depot rather than building a third one. The CPU target multiversions the
# precompiled code across Zen2, Zen4 and Icelake-server, so one depot serves the a100 and h100
# partitions without recompiling per architecture — which is what makes sharing it safe.
export JULIA_DEPOT_PATH=$HOME/julia/julia_a1003:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"

julia --project exp_square_HIT/cfl_probe.jl
