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

## full simulation
n_dns = Int(512) 
n_les = Int(64)
Re = Float32(2_000)
Δt = Float32(2.5e-4)
tsim = Float32(100)
tburn = Float32(4)

## small test parameters
n_dns = Int(128)
n_les = Int(64)
Re = Float32(2_000)
Δt = Float32(2.5e-4)
tsim = Float32(0.5)
tburn = Float32(0.2)

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

# For running on a CUDA compatible GPU

T = Float32
ArrayType = CuArray
backend = CUDABackend()

ustart = load(indir*"/u_start_spinnup_$(n_dns)_Re$(Re)_freeze_$(freeze)_tsim$(tburn).jld2", "u_start");
if ustart isa Tuple # old INS data format
    ustart = stack(ArrayType.(ustart));
elseif ustart isa Array{<:Number,4} # new INS data format
    ustart = ArrayType(ustart);
end

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
data_train = create_ref_data(; params_train..., ustart, n_checkpoints = 1, checkpoint_name = checkpoints_dir);
t4 = time()
println("HF simulation done. Time: $(t4-t3) s")
# Save filtered DNS data
filename = "$outdir/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_freeze_$(freeze)_tsim$(params_train.tsim).jld2"
jldsave(filename; data_train, params_train)