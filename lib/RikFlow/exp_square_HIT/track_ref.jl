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
Re = Float32(2000)
Δt = Float32(2.7e-4)*10
tsim = Float32(0.006)



outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

# For running on a CUDA compatible GPU

T = Float32
ArrayType = CuArray


# load reference data
ref_file = outdir*"/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim10.0.jld2"
data_train = load(ref_file, "data_train");
params_train = load(ref_file, "params_train");
# get initial condition
ustart = ArrayType.(data_train.data[1].u[1]);
# get ref trajectories
qoi_ref = stack(data_train.data[1].qoi_hist);

params_track = (;
    params_train...,
    tsim,
    Δt,
    ArrayType,
    ustart, 
    qoi_ref, 
    savefreq = 100);

data_track = track_ref(; params_track...);

# check tracking
n_steps = size(data_track.q, 2)
erel = (qoi_ref[:,1:n_steps]-data_track.q)./(qoi_ref[:,1:n_steps]);
maximum(abs, erel)
@assert(maximum(abs, erel)<1e-2)

# Save tracking data
jldsave("$outdir/data_track_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(tsim).jld2"; data_track, params_track);