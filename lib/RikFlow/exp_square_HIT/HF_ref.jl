if false                                               #src
    include("../src/RikFlow.jl")                  #src
    #include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
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

n_dns = Int(512)
n_les = Int(64)
Re = Float32(2_000)
############################
Δt = Float32(2.5e-4)
tsim = Float32(10)
#### -> 2,5 hours
# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
freeze = 10 # number of time steps to freeze the forcing

seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)

outdir = @__DIR__() *"/output/new"
indir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

# For running on a CUDA compatible GPU

T = Float32
ArrayType = CuArray
#device = x -> adapt(CuArray, x)
tburn = Float32(4)
ustart = ArrayType.(load(indir*"/u_start_spinnup_$(n_dns)_Re$(Re)_freeze_$(freeze)_tsim$(tburn).jld2", "u_start"));

# Parameters
get_params(nlesscalar) = (;
    D = 3,
    Re,
    lims = ( (T(0) , T(1)) , (T(0) , T(1)), (T(0),T(1)) ),
    qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]],
    tsim,
    Δt,
    nles = map(n -> (n, n, n), nlesscalar), # LES resolutions
    ndns = (n -> (n, n, n))(n_dns), # DNS resolution
    filters = (FaceAverage(),),
    ArrayType,
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou ),
)

params_train = (; get_params([n_les])..., savefreq = 10, plotfreq = 1000);
t3 = time()
data_train = create_ref_data(; params_train..., ustart);
t4 = time()
println("HF simulation done. Time: $(t4-t3) s")
# Save filtered DNS data
filename = "$outdir/data_train_dns$(n_dns)_les$(n_les)_Re$(Re)_freeze_$(freeze)_tsim$(params_train.tsim).jld2"
jldsave(filename; data_train, params_train)