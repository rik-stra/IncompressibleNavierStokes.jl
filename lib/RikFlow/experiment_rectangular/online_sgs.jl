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

n_dns = Int(256)
n_les = Int(64)
Re = Float32(3000)
Δt = Float32(1.5e-3)
tsim = Float32(100)
sampling_method = :mvg

seeds = (;
    dns = 123, # Initial conditions
    to = 234 # TO method online sampling
)
rng_TO = Xoshiro(seeds.to)

outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

# For running on a CUDA compatible GPU

T = Float32
ArrayType = CuArray


# load reference data
track_file = outdir*"/data_track_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim20.0.jld2"
data_track = load(track_file, "data_track");
# get initial condition
ustart = ArrayType.(data_track.fields[1].u);
# get ref trajectories
dQ_data = data_track.dQ;

get_params(nlesscalar) = (;
    D = 3,
    Re,
    lims = ( (T(0) , T(3)) , (T(0) , T(1)), (T(0),T(1)) ),
    qois = [["Z",0,15],["E", 0, 15],["Z",16,31],["E", 16, 31]],
    tsim,
    Δt,
    nles = map(n -> (3*n, n, n), nlesscalar), # LES resolutions
    ndns = (n -> (3*n, n, n))(n_dns), # DNS resolution
    filters = (FaceAverage(),),
    ArrayType,
    ustart,
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y)
)

params = (; get_params([n_les])..., dQ_data, sampling_method = sampling_method, savefreq = 1000, rng = rng_TO);

data_online = online_sgs(; params...);


# Save tracking data
jldsave("$outdir/data_online_sampling$(sampling_method)_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(tsim).jld2"; data_online, params);