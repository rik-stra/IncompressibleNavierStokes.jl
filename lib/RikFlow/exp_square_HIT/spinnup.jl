# This script performs a high-fidelity simulation of homogeneous isotropic turbulence (HIT), 
# it saves the final turbulent field and a plot of the energy over the simulation time.


println("Loading modules...")
t0 = time()
#using LoggingExtras
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


println("Modules loaded. Time: $(t1-t0) s")

# full size simulation
n_dns = Int(512)
n_les = Int(64)
Re = Float32(2_000)
tburn = Float32(4)
Δt = Float32(0.00025)

# small test parameters
n_dns = Int(128)
n_les = Int(64)
Re = Float32(2_000)
tburn = Float32(0.2)
Δt = Float32(0.00025)

# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber
freeze = 10 # number of time steps to freeze the forcing

outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

seeds = (;
    ou_spin = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)

# For running on a CUDA compatible GPU

T = Float32
backend = CUDABackend()

# Parameters
get_params(nlesscalar) = (;
    D = 3,
    Re,
    lims = ( (T(0) , T(1)) , (T(0) , T(1)), (T(0),T(1)) ),
    tburn,
    backend,
    ndns = (n -> (n, n, n))(n_dns), # DNS resolution
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou_spin ),
)

params_train = (; get_params([n_les])..., Δt);
t3 = time()
u_start, ehist = spinnup(; params_train...);
u_start = Array(u_start);

t4 = time()
println("HF simulation done. Time: $(t4-t3) s")
# Save filtered DNS data
filename = "$outdir/u_start_spinnup_$(n_dns)_Re$(Re)_freeze_$(freeze)_tsim$(params_train.tburn).jld2"
jldsave(filename; u_start)

# Plot
save(outdir*"/ehist2_$(n_dns)_Re$(Re)_freeze_$(freeze)_tsim$(params_train.tburn).png",ehist)