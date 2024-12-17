if false                                               #src
    include("../src/RikFlow.jl")                  #src
    #include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
    using .SymmetryClosure                             #src
    #using .NeuralClosure                               #src
    using .IncompressibleNavierStokes                  #src
end

using Random
using CairoMakie
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA
ArrayType = CuArray
backend = CUDABackend()

n_dns = Int(512)
n_les = Int(64)
Re = Float32(2_000)
############################
Δt = Float32(2.5e-3)
tsim = Float32(10)
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

outdir = @__DIR__() *"/output/new"
ispath(outdir) || mkpath(outdir)

# For running on a CUDA compatible GPU

T = Float32



# load reference data
ref_file = outdir*"/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_freeze_10_tsim10.0.jld2"
data_train = load(ref_file, "data_train");
params_train = load(ref_file, "params_train");
# get initial condition
if data_train.data[1].u[1] isa Tuple
    ustart = stack(ArrayType.(data_train.data[1].u[1]));
elseif data_train.data[1].u[1] isa Array{<:Number,4}
    ustart = ArrayType(data_train.data[1].u[1]);
end

# get ref trajectories
qoi_ref = stack(data_train.data[1].qoi_hist);

ref_reader = Reference_reader(qoi_ref);

params_track = (;
    params_train...,
    tsim,
    Δt,
    ArrayType,
    backend, 
    ref_reader,
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou),
    savefreq = 100);

data_track = track_ref(; params_track..., ustart);

# check tracking
n_steps = size(data_track.q, 2)
erel = (qoi_ref[:,1:n_steps]-data_track.q)./(qoi_ref[:,1:n_steps]);
maximum(abs, erel)
#@assert(maximum(abs, erel)<1e-2)

# Save tracking data
jldsave("$outdir/data_track2_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(tsim).jld2"; data_track, params_track);