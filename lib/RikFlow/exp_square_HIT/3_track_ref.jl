if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
end

using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

# For running on a CUDA compatible GPU
T = Float64
ArrayType = CuArray
backend = CUDABackend()


# parameters
n_dns = Int(512)
n_les = Int(64)
Re = T(2_000)
Δt = T(2.5e-3)
tsim = T(10)

ref_file = @__DIR__()*"/output/paper_data_HIT/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_freeze_10_tsim100.0.jld2"
outdir = @__DIR__()*"/output"
ispath(outdir) || mkpath(outdir)

# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
freeze = 1 # number of time steps to freeze the forcing

seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)

# load reference data

data_train = load(ref_file, "data_train");
params_train = load(ref_file, "params_train");
# get initial condition
if data_train.data[1].u[1] isa Tuple
    ustart = stack(ArrayType{T}.(data_train.data[1].u[1]));
elseif data_train.data[1].u[1] isa Array{<:Number,4}
    ustart = ArrayType{T}(data_train.data[1].u[1]);
end

# get ref trajectories
qoi_ref = stack(data_train.data[1].qoi_hist[1:Int(tsim/Δt)+1]);
ref_reader = Reference_reader(qoi_ref);

params_track = (;
    params_train...,
    # 🔴 Override the archived Re. The splat above carries the archive's Float32 parameters, and a
    # later key wins — without this the setup is built at Float32 while the script declares
    # Float64, and `typeof(setup.Re)` silently drives every QoI buffer back to single precision.
    # `rf_setup` now refuses that mismatch outright, so this is what keeps the script runnable.
    Re = T(2_000),
    tsim,
    Δt,
    ArrayType,
    backend,
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou),
    savefreq = 100);

data_track = track_ref(; params_track..., ref_reader, ustart);

# check tracking
n_steps = size(data_track.q, 2)
erel = (qoi_ref[:,1:n_steps]-data_track.q)./(qoi_ref[:,1:n_steps]);
maximum(abs, erel)
#@assert(maximum(abs, erel)<1e-2)

# Save tracking data
jldsave("$outdir/data_track_tsim$(tsim).jld2"; data_track, params_track);