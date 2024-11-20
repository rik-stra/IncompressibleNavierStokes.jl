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

rng_TO = Xoshiro(seeds.to)

outdir = @__DIR__() *"/output/new"
ispath(outdir) || mkpath(outdir)

# For running on a CUDA compatible GPU

T = Float32
ArrayType = CuArray


# load reference data
track_file = outdir*"/data_track_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim10.0.jld2"
data_track = load(track_file, "data_track");
params_track = load(track_file, "params_track");
# get initial condition
ustart = ArrayType.(data_track.fields[1].u);
# get ref trajectories
dQ_data = data_track.dQ;
time_series_sampler = RikFlow.MVG_sampler(dQ_data, rng_TO, ArrayType, T);

params = (;
    params_track...,
    tsim,
    Δt,
    ArrayType,
    ustart, 
    #ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou),
    savefreq = 100,
    time_series_method = time_series_sampler);

data_online = online_sgs(; params...);


# Save tracking data
jldsave("$outdir/data_online_sampling$(typeof(time_series_sampler))_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(tsim).jld2"; data_online, params);