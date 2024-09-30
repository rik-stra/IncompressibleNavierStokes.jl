if false                                               #src
    include("src/RikFlow.jl")                  #src
    #include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../src/IncompressibleNavierStokes.jl") #src
    using .SymmetryClosure                             #src
    #using .NeuralClosure                               #src
    using .IncompressibleNavierStokes                  #src
end     

using RikFlow
using IncompressibleNavierStokes
using Random
using CairoMakie
using JLD2
#using NeuralClosure

# examples of macros
@__DIR__
@__FILE__
basename(@__FILE__)
dirname(@__FILE__)


plotdir = @__DIR__() * "/output/close3Dkolmogorov/plots"
outdir = @__DIR__() *"/output/close3Dkolmogorov"
ispath(plotdir) || mkpath(plotdir)
ispath(outdir) || mkpath(outdir)

seeds = (;
    dns = 123, # Initial conditions
    to = 234 # TO method online sampling
)

# For running on CPU.
# Consider reducing the sizes of DNS, LES, and CNN layers if
# you want to test run on a laptop.
T = Float32
ArrayType = Array
#device = identity
#clean() = nothing

# For running on a CUDA compatible GPU
using CUDA
T = Float32
ArrayType = CuArray
#device = x -> adapt(CuArray, x)

## run DNS
rng_DNS = Xoshiro(seeds.dns)

# Parameters
get_params(nlesscalar) = (;
    D = 3,
    Re = T(2_000),
    lims = ( (T(0) , T(3)) , (T(0) , T(1)), (T(0),T(1)) ),
    qois = [["Z",0,4],["E", 0, 4],["Z",5,10],["E", 5, 10]],
    tburn = T(20),
    tsim = T(40),
    Δt = T(5e-3),
    nles = map(n -> (3*n, n, n), nlesscalar), # LES resolutions
    ndns = (n -> (3*n, n, n))(128), # DNS resolution
    filters = (FaceAverage(),),
    ArrayType,
    icfunc = (setup, psolver, rng) ->
        random_field(setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y)
)

params_train = (; get_params([32])..., savefreq = 100);
data_train = create_ref_data(; params_train..., rng = rng_DNS);

# Save filtered DNS data
jldsave("$outdir/data_train.jld2"; data_train)

# load filtered DNS data
data_train = load("$outdir/data_train.jld2", "data_train");

ustart = ArrayType.(data_train.data[1].u[1]);
qoi_ref = stack(data_train.data[1].qoi_hist);
params_track = (; params_train..., ustart, qoi_ref, tsim = T(40));
data_track = track_ref(; params_track...);

# check tracking
erel = (qoi_ref-data_track.q)./(qoi_ref);
maximum(abs, erel)
@assert(maximum(abs, erel)<1e-2)

# Save tracking data
jldsave("$outdir/data_track.jld2"; data_track);


# load tracking data
data_track = load("$outdir/data_track.jld2", "data_track");

# plot tracking data

# plot Qois
let
    f = Figure();
    axs = [Axis(f[i ÷ 2, i%2]) for i in 0:size(data_track.q, 1)-1]
    for i in 1:size(data_track.q, 1)
        lines!(axs[i],qoi_ref'[:,i])
        lines!(axs[i],data_track.q'[:,i],linestyle=:dash)
    end
    display(f)
end
# plot dQ
let
    f = Figure();
    axs = [Axis(f[i ÷ 2, i%2]) for i in 0:size(data_track.dQ, 1)-1]
    for i in 1:size(data_track.dQ, 1)
        lines!(axs[i],data_track.dQ[i,:])
    end
    display(f)
end

# fit multi-variate normal distribution
let
    using Distributions
    d = fit(MvNormal, data_track.dQ)
    rng = Xoshiro(seeds.to)
    samples = zeros(4,10000)
    rand!(rng,d, samples)

    # plot distributions
    f = Figure();
    axs = [Axis(f[i ÷ 2, i%2]) for i in 0:size(data_track.dQ, 1)-1]
    for i in 1:size(data_track.dQ, 1)
        hist!(axs[i],samples[i,:], normalization = :pdf)
        hist!(axs[i],data_track.dQ[i,:], normalization = :pdf)
    end
    display(f)
end

# online prediction
rng_TO = Xoshiro(seeds.to)

dQ_data = data_track.dQ
outputs_online = online_sgs(; params_train..., ustart, dQ_data, sampling_method = :mvg , tsim = T(20), rng = rng_TO);

# plot Qois
let
    f = Figure();
    axs = [Axis(f[i ÷ 2, i%2]) for i in 0:size(outputs_online.q, 1)-1]
    for i in 1:size(outputs_online.q, 1)
        lines!(axs[i],qoi_ref'[:,i])
        lines!(axs[i],outputs_online.q'[:,i],linestyle=:dash)
    end
    display(f)
end

# plot dQ
let
    f = Figure();
    axs = [Axis(f[i ÷ 2, i%2]) for i in 0:size(outputs_online.dQ, 1)-1]
    for i in 1:size(outputs_online.dQ, 1)
        lines!(axs[i],outputs_online.dQ[i,:])
    end
    display(f)
end


# plot distributions
let
    f = Figure();
    axs = [Axis(f[i ÷ 2, i%2]) for i in 0:size(data_track.dQ, 1)-1]
    for i in 1:size(outputs_online.dQ, 1)
        hist!(axs[i],samples[i,:], normalization = :pdf)
        hist!(axs[i],outputs_online.dQ[i,:], normalization = :pdf)
    end
    display(f)
end

let
    f = Figure();
    axs = [Axis(f[i ÷ 2, i%2]) for i in 0:size(data_track.q, 1)-1]
    for i in 1:size(outputs_online.q, 1)
        hist!(axs[i],qoi_ref[i,:], normalization = :pdf)
        hist!(axs[i],outputs_online.q[i,:], normalization = :pdf)
    end
    display(f)
end