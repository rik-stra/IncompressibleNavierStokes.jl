if false                                               #src
    include("src/RikFlow.jl")                  #src
    #include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../src/IncompressibleNavierStokes.jl") #src
    using .SymmetryClosure                             #src
    #using .NeuralClosure                               #src
    using .IncompressibleNavierStokes                  #src
end     


# perfom a HF simulation
println("Loading modules...")
t0 = time()
#using LoggingExtras
using Random
using CairoMakie
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA
t1 = time()

# Write output to file, as the default SLURM file is not updated often enough
jobid = ENV["SLURM_JOB_ID"]
taskid = ENV["SLURM_ARRAY_TASK_ID"]
logfile = joinpath(@__DIR__, "log_$(jobid)_$(taskid).out")
filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
logger = TeeLogger(global_logger(), filelogger)
global_logger(logger)


println("Modules loaded. Time: $(t1-t0) s")

#n_dns = parse(Int,ARGS[1])
#n_les = parse(Int,ARGS[2])
#Re = parse(Float32,ARGS[3])

n_dns = Int(256)
n_les = Int(64)
Re = Float32(3000)

outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

seeds = (;
    dns = 123, # Initial conditions
    to = 234 # TO method online sampling
)

# For running on a CUDA compatible GPU

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
    tburn = T(20),
    Δt = T(1e-4),
    ndns = (n -> (3*n, n, n))(n_dns), # DNS resolution
    ArrayType,
    icfunc = (setup, psolver, rng) ->
        random_field(setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y)
)

params_train = (; get_params([n_les])..., savefreq = 100);
t3 = time()
data_train = spinnup(; params_train..., rng = rng_DNS)
t4 = time()
println("HF simulation done. Time: $(t4-t3) s")
# Save filtered DNS data
filename = "$outdir/spinnup_dns$(n_dns)_Re$(Re)_tsim$(params_train.tburn).jld2"
jldsave(filename; data_train.states)
