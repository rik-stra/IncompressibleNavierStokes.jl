if false                                               #src
    include("../../src/RikFlow.jl")                  #src
    #include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../../../src/IncompressibleNavierStokes.jl") #src
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
using RegularizedLeastSquares
using Distributions
using LinearAlgebra


ArrayType = CuArray
backend = CUDABackend()

n_replicas = 5
noise_levels = [0, 0.1, 0.05, 0.01, 0.005, 0.001]
#parse input ARGS
model_index = parse(Int, ARGS[1])


outdir = @__DIR__()*"/output/tracking/"
ispath(outdir) || mkpath(outdir)

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


# For running on a CUDA compatible GPU
T = Float32

# load reference data
ref_file = @__DIR__()*"/../output/new/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_freeze_10_tsim100.0.jld2"
data_train = load(ref_file, "data_train");
params_train = load(ref_file, "params_train");
# get initial condition
if data_train.data[1].u[1] isa Tuple
    ustart = stack(ArrayType.(data_train.data[1].u[1]));
elseif data_train.data[1].u[1] isa Array{<:Number,4}
    ustart = ArrayType(data_train.data[1].u[1]);
end

# get ref trajectories
qoi_ref = stack(data_train.data[1].qoi_hist[1:Int(tsim/Δt)+1]);
tracking_noise = noise_levels[model_index]

for i in 1:n_replicas
    ref_reader = Reference_reader(qoi_ref);
    params_track = (;
        params_train...,
        tsim,
        Δt,
        ArrayType,
        backend,
        ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou),
        savefreq = 100,
        tracking_noise,
        tracking_noise_seed = i);

    data_track = track_ref(; params_track..., ref_reader, ustart);

    # Save tracking data
    jldsave("$outdir/data_track_trackingnoise_std_$(tracking_noise)_Re$(Re)_tsim$(tsim)_replica$(i).jld2"; data_track, params_track);
end
