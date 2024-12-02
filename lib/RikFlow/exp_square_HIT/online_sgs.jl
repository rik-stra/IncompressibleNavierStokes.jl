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


params = (;
    params_track...,
    tsim,
    Δt,
    ArrayType,
    ustart,
    savefreq = 100);

# Run 10 replicas
for i in 1:10
    time_series_sampler = RikFlow.Resampler(dQ_data, Xoshiro(seeds.to+i));
    #time_series_sampler = RikFlow.MVG_sampler(dQ_data, Xoshiro(seeds.to+i));
    #ANN_file_name = @__DIR__()*"/../deep_learning/output/trained_models/ANN_tanh_regularized_hist3.jld2"
    #q_hist = ArrayType(zeros(T, 6, 3))
    #time_series_sampler = RikFlow.ANN(ANN_file_name, q_hist = q_hist);
# run the sim
    @info "Running sim $i out of 10"
    data_online = online_sgs(; params..., time_series_method=time_series_sampler);
# Save tracking data
    # make dir if not exist
    folder = "$outdir/$(typeof(time_series_sampler))"
    ispath(folder) || mkpath(folder)
    jldsave("$folder/data_online_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(tsim)_replica$(i).jld2"; data_online, params);
end