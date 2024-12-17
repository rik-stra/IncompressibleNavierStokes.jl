if false                                               #src
    include("../src/RikFlow.jl")                  #src
    #include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
    using .SymmetryClosure                             #src
    #using .NeuralClosure                               #src
    using .IncompressibleNavierStokes                  #src
end

using Random
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA


n_dns = Int(512)
n_les = Int(64)
Re = Float32(2_000);
############################
Δt = Float32(2.5e-3);
tsim = Float32(10);
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

# inputs
name = "MVG"
track_file = "/output/new/data_track2_dns512_les64_Re2000.0_tsim10.0.jld2"
n_replicas = 10
hist_len = 0

out_dir = @__DIR__()*"/output/new/$(name)/"
ispath(out_dir) || mkpath(out_dir)

# For running on a CUDA compatible GPU
T = Float32
ArrayType = CuArray
backend = CUDABackend()

# load reference data
data_track = load(@__DIR__()*track_file, "data_track");
params_track = load(@__DIR__()*track_file, "params_track");
# get initial condition
if data_track.fields[1].u isa Tuple
    ustart = stack(ArrayType.(data_track.fields[1].u));
elseif data_track.fields[1].u isa Array{<:Number,4}
    ustart = ArrayType(data_track.fields[1].u);
end
# get ref trajectories
dQ_data = data_track.dQ;

params = (;
    params_track...,
    tsim,
    Δt,
    ArrayType,
    backend,
    ustart,
    savefreq = 100);

# Run 10 replicas
for i in 1:n_replicas
    #time_series_sampler = RikFlow.Resampler(dQ_data, Xoshiro(seeds.to+i));
    time_series_sampler = RikFlow.MVG_sampler(dQ_data, Xoshiro(seeds.to+i));

    if hist_len == 0
        q_hist = nothing
    else
        q_hist = ArrayType{T}(reverse(dQ_data[:,1:hist_len],dims=2)) #give first #histlen dQ values, the first dQ value should be at end of array
    end
    #time_series_sampler = RikFlow.ANN(ANN_file_name, q_hist = q_hist);
    #time_series_sampler = RikFlow.LinReg(LinReg_file_name, Xoshiro(seeds.to+i), q_hist = q_hist);
    
# run the sim
    @info "Running sim $i out of $n_replicas"
    data_online = online_sgs(; params..., time_series_method=time_series_sampler);
# Save tracking data
    jldsave(out_dir*"data_online_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(tsim)_replica$(i).jld2"; data_online, params);
end