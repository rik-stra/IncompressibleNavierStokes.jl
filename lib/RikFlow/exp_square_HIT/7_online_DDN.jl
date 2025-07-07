if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
end

using Random
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

DDN_folder = @__DIR__()*"/output/TO_DDN"
track_file = @__DIR__()*"/output/data_track_tsim10.0.jld2" #we will take some parameters and the initial field from here
ispath(DDN_folder) || mkpath(DDN_folder)
## DDN inputs
n_replicas = 5
traindata_range = 400:4000

# simulation parameters
Re = Float32(2_000);
Δt = Float32(2.5e-3);
tsim = Float32(100);
# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
freeze = 1 # number of time steps to freeze the forcing

# For running on a CUDA compatible GPU
T = Float32
ArrayType = CuArray
backend = CUDABackend()


seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)


# load reference data
data_track = load(track_file, "data_track");
params_track = load(track_file, "params_track");
# get initial condition
if data_track.fields[1].u isa Tuple
    ustart = stack(ArrayType.(data_track.fields[1].u));
elseif data_track.fields[1].u isa Array{<:Number,4}
    ustart = ArrayType(data_track.fields[1].u);
end
# get ref trajectories
dQ_data = data_track.dQ[:,traindata_range];

params = (;
    params_track...,
    tsim,
    Δt,
    ArrayType,
    backend,
    savefreq = 1000);

# Run 10 replicas
for i in 1:n_replicas
    #time_series_sampler = RikFlow.Resampler(dQ_data, Xoshiro(seeds.to+i));
    time_series_sampler = RikFlow.MVG_sampler(dQ_data, Xoshiro(seeds.to+i));

# run the sim
    @info "Running sim $i out of $n_replicas"
    data_online = online_sgs(; params..., ustart=ustart, time_series_method=time_series_sampler);
# Save tracking data
    jldsave(DDN_folder*"/DDN_data_online_tsim$(tsim)_replica$(i).jld2"; data_online, params);
end