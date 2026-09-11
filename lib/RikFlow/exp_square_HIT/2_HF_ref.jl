# High fidelity reference simulation of homogeneous isotropic turbulence (HIT).
# Collects qoi reference trajectories.

if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes                  #src
end     


println("Loading modules...")
t0 = time()
using LoggingExtras
using Random
using CairoMakie
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA
t1 = time()

# Write output to file, as the default SLURM file is not updated often enough
# jobid = ENV["SLURM_JOB_ID"]
# logfile = joinpath(@__DIR__, "log_$(jobid).out")
# filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
# logger = TeeLogger(global_logger(), filelogger)
# global_logger(logger)

# ---------------------------------------------------------------------------------------------
# Parameters. ONE block, defined once.
#
# ⚠️ This file used to declare the production parameters and then silently overwrite them with a
# "small test parameters" block six lines later, so as committed it ran n_dns = 128, tsim = 0.5
# while appearing to run the production case. Set HF_REF_SMOKE=1 for the small variant instead;
# nothing is shadowed.
#
# 🔴 Float64 and LMWray3 for the regeneration (Rik, 2026-09-11).
#   - Float64 because Float32 leaves the coefficient-level diagnostics unresolved
#     (claude_memory.md gotcha #26: cond*eps(Float32) ~ 0.2), and because it makes the DNS/LES
#     forcing equivalence exact: Float32(2.5e-4)*10 != Float32(2.5e-3), but in Float64 it is.
#   - LMWray3 because it needs one stage vector instead of four (7.1 GiB against 16.2 GiB of ODE
#     cache at 512^3 Float64) and three right-hand-side evaluations per step instead of four.
#
# ⚠️ Consequence, and it is not a defect: the OU chain draws `randn!` into a Float64 buffer, which
# consumes the stream differently from Float32, so this run is a *different realisation* from the
# archive - not a refinement of it. Together with the scheme change, the archived 401 fields are
# reproducible only at field 1, the filtered initial condition, which needs no time stepping.
# `tools/check_ref_401.jl ic` is that check and it passes.
# ---------------------------------------------------------------------------------------------

const SMOKE = get(ENV, "HF_REF_SMOKE", "0") == "1"

T = Float64
ArrayType = CuArray
backend = CUDABackend()

n_dns = SMOKE ? Int(128) : Int(512)
n_les = Int(64)
Re = T(2_000)
Δt = T(2.5e-4)
tsim = SMOKE ? T(0.5) : T(100)
tburn = T(4)

# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
freeze = 10 # number of time steps to freeze the forcing

seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)

outdir = @__DIR__() *"/output"
indir = @__DIR__() *"/output"
checkpoints_dir = @__DIR__() *"/output/checkpoints"
ispath(outdir) || mkpath(outdir)
ispath(checkpoints_dir) || mkpath(checkpoints_dir)

# Device and precision are set in the parameter block above.

# 🔑 The archived spin-up is reused rather than re-run, and it is stored in Float32.
#
# Reusing it is deliberate. The spin-up is a plain forced DNS - no TO, no QoIs - so `∂` never
# touches it and the Nyquist change of `09954be1` cannot have tainted it; and it seeds its OU chain
# from `ou_spin = 123` while this run seeds from `ou = 333`, so no forcing state carries over. It
# is only an initial velocity field, and it is a valid one.
#
# Re-running it would produce a different field and leave the new reference sharing nothing with
# the archive - not even field 1. Keeping it preserves the one full-scale anchor that survives.
#
# Float32 -> Float64 promotion is exact (every Float32 is representable). The field is only
# Float32-*accurate*, so its divergence is ~1e-7 rather than ~1e-16; the first pressure projection
# removes that, and 1e-7 on a turbulent field is nothing.
ustart = load(indir*"/u_start_spinnup_$(n_dns)_Re$(Re)_freeze_$(freeze)_tsim$(tburn).jld2", "u_start");
if ustart isa Tuple # old INS data format
    ustart = stack(ArrayType{T}.(ustart));
elseif ustart isa Array{<:Number,4} # new INS data format
    ustart = ArrayType{T}(ustart);
end
@info "initial condition" eltype(ustart) size(ustart)

# Parameters
get_params(nlesscalar) = (;
    D = 3,
    Re,
    lims = ( (T(0) , T(1)) , (T(0) , T(1)), (T(0),T(1)) ),
    qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]],
    tsim,
    Δt,
    nles = map(n -> (n, n, n), nlesscalar), # LES resolutions
    ndns = (n -> (n, n, n))(n_dns), # DNS resolution
    filters = (FaceAverage(),),
    ArrayType,
    backend,
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou ),
)

params_train = (; get_params([n_les])..., savefreq = 10, plotfreq = 1000);
t3 = time()
data_train = create_ref_data(; params_train..., ustart, method = LMWray3(; T),
    n_checkpoints = 1, checkpoint_name = checkpoints_dir);
t4 = time()
println("HF simulation done. Time: $(t4-t3) s")
# Save filtered DNS data
# Tagged with precision and stepper: a Float64/LMWray3 reference must not be confusable
# with the archived Float32/RK44 one, which has otherwise the same name.
filename = "$outdir/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_freeze_$(freeze)_tsim$(params_train.tsim)_f64_lmwray3.jld2"
jldsave(filename; data_train, params_train)