#!/bin/bash
#SBATCH -J cfl_probe
#SBATCH -t 00:30:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

# Adaptive-timestep probe of the Float64 / LMWray3 configuration at 512^3.
#
# Submit from either exp_square_HIT or lib/RikFlow:
#     sbatch batch_scripts/run_cfl_probe.sh
#     sbatch exp_square_HIT/batch_scripts/run_cfl_probe.sh
# SLURM runs the job in the directory sbatch was called from, and the lookup below handles both
# rather than assuming one. The probe's own input and output paths come from @__DIR__, not the
# working directory, so the initial condition must be in exp_square_HIT/output/ regardless.
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

# Find the probe from whichever directory the job started in, and say so if it is neither.
if [ -f cfl_probe.jl ]; then
    SCRIPT=cfl_probe.jl
elif [ -f exp_square_HIT/cfl_probe.jl ]; then
    SCRIPT=exp_square_HIT/cfl_probe.jl
else
    echo "run_cfl_probe.sh: cannot find cfl_probe.jl from $(pwd)" >&2
    echo "  submit from exp_square_HIT or from lib/RikFlow" >&2
    exit 1
fi

# Bring the depot's caches in line with this invocation before running.
#
# 🔴 Needed because of JULIA_CPU_TARGET above. Multiversioning changes the *content* of the
# precompiled images, so a depot populated without it — or with a different target, or by a julia
# invoked with different flags — holds caches that this run will not accept. Julia reports that as
#
#     Precompiled image ... "Adapt" not available with flags CacheFlags(...)
#
# and errors rather than silently rebuilding the whole tree. Doing it explicitly here rebuilds
# once, in the same environment the run uses, instead of failing mid-load.
#
# If this turns into a long rebuild every job, the alternative is to drop JULIA_CPU_TARGET and go
# back to a per-partition depot (the run_spinnup.sh pattern) — that trades one depot per
# architecture for no recompilation.
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

julia --project "$SCRIPT"
