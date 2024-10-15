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
tsim = Float32(20.)

outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

# For running on a CUDA compatible GPU

T = Float32
ArrayType = CuArray


# load reference data
ref_file = outdir*"/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim20.0.jld2"
data_train = load(ref_file, "data_train");
# get initial condition
ustart = ArrayType.(data_train.data[1].u[1]);
# get ref trajectories
qoi_ref = stack(data_train.data[1].qoi_hist);

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

params_track = (; get_params([n_les])..., qoi_ref, savefreq = 1000);

data_track = track_ref(; params_track...);

# check tracking
erel = (qoi_ref-data_track.q)./(qoi_ref);
maximum(abs, erel)
@assert(maximum(abs, erel)<1e-2)

# Save tracking data
jldsave("$outdir/data_track_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(tsim).jld2"; data_track, params_track);