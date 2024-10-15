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
using LoggingExtras
using Random
using CairoMakie
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA
t1 = time()

# Write output to file, as the default SLURM file is not updated often enough
jobid = ENV["SLURM_JOB_ID"]
#taskid = ENV["SLURM_ARRAY_TASK_ID"]
logfile = joinpath(@__DIR__, "log_$(jobid).out")
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
Δt = Float32(1.5e-3)
tsim = Float32(100)

outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

# For running on a CUDA compatible GPU

T = Float32
ArrayType = CuArray
#device = x -> adapt(CuArray, x)
tburn = Float32(20)
ustart = ArrayType.(load(outdir*"/u_start_spinnup_$(n_dns)_Re$(Re)_tsim$(tburn).jld2", "u_start"));

# Parameters
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

params_train = (; get_params([n_les])..., savefreq = 1000);
t3 = time()
data_train = create_ref_data(; params_train...);
t4 = time()
println("HF simulation done. Time: $(t4-t3) s")
# Save filtered DNS data
filename = "$outdir/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_tsim$(params_train.tsim).jld2"
jldsave(filename; data_train, params_train)