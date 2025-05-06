if false                                               #src
    include("../../src/RikFlow.jl")                  #src
    #include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../../../src/IncompressibleNavierStokes.jl") #src
    using .SymmetryClosure                             #src
    #using .NeuralClosure                               #src
    using .IncompressibleNavierStokes                  #src
end

using Random
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

#parse input ARGS
model_index = parse(Int, ARGS[1])
#model_index = 4

n_dns = Int(512)
n_les = Int(64)
Re = Float32(2_000);
############################
Δt = Float32(2.5e-3);
tsim = Float32(100);
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


## Load data
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
(; name, hist_len, n_replicas, hist_var,tracking_noise) = inputs[model_index]

out_dir = @__DIR__()*"/output/online/$(name)/"
# For running on a CUDA compatible GPU
T = Float32
ArrayType = CuArray
backend = CUDABackend()

# load reference data
track_file = @__DIR__()*"/output/tracking/data_track_trackingnoise_std_$(tracking_noise)_Re2000.0_tsim10.0_replica1.jld2"
params_track = load(track_file, "params_track");
#track_file = @__DIR__()*"/../output/new/data_track2_dns512_les64_Re2000.0_tsim100.0.jld2"
data_track = load(track_file, "data_track");

# get initial condition
if data_track.fields[1].u isa Tuple
    ustart = stack(ArrayType.(data_track.fields[1].u));
elseif data_track.fields[1].u isa Array{<:Number,4}
    ustart = ArrayType(data_track.fields[1].u);
end
# get ref trajectories
dQ_data = data_track.dQ[:,1:100];

params = (;
    params_track...,
    tsim,
    Δt,
    ArrayType,
    backend,
    ustart,
    savefreq = 1000);

# Run 10 replicas
for i in 1:n_replicas
    LinReg_file_name = out_dir*"LinReg.jld2"
    if hist_len == 0
        q_hist = nothing
    else
        q_hist = ArrayType{T}(zeros(T,size(params.qois,1),hist_len)) 
        if hist_var == :q_star_q
            q_hist = cat(q_hist, q_hist, dims=1)
        end
    end
    time_series_sampler = RikFlow.LinReg(LinReg_file_name, Xoshiro(seeds.to+i+2), q_hist = q_hist, spinnup_data = ArrayType{T}(dQ_data));
    
# run the sim
    @info "Running sim $i out of $n_replicas"
    data_online = online_sgs(; params..., time_series_method=time_series_sampler);
# Save tracking data
    jldsave(out_dir*"data_online_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(tsim)_replica$(i)_rand_initial_dQ.jld2"; data_online, params);
end
