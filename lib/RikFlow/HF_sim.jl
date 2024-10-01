# perfom a HF simulation
println("Loading modules...")
t0 = time()
using Random
using CairoMakie
using JLD2
using RikFlow
using IncompressibleNavierStokes
t1 = time()
println("Modules loaded. Time: $(t1-t0) s")

n_dns = parse(Int,ARGS[1])
n_les = parse(Int,ARGS[2])
Re = parse(Float32,ARGS[3])

outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

seeds = (;
    dns = 123, # Initial conditions
    to = 234 # TO method online sampling
)

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
    Re,
    lims = ( (T(0) , T(3)) , (T(0) , T(1)), (T(0),T(1)) ),
    qois = [["Z",0,4],["E", 0, 4],["Z",5,10],["E", 5, 10]],
    tburn = T(10),
    tsim = T(10),
    Δt = T(5e-3),
    nles = map(n -> (3*n, n, n), nlesscalar), # LES resolutions
    ndns = (n -> (3*n, n, n))(n_dns), # DNS resolution
    filters = (FaceAverage(),),
    ArrayType,
    icfunc = (setup, psolver, rng) ->
        random_field(setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y)
)

params_train = (; get_params([n_les])..., savefreq = 100);
t3 = time()
data_train = create_ref_data(; params_train..., rng = rng_DNS);
t4 = time()
println("HF simulation done. Time: $(t4-t3) s")
# Save filtered DNS data
filename = "$outdir/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(params_train.tsim).jld2"
jldsave(filename; data_train)
