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
#jobid = ENV["SLURM_JOB_ID"]
#taskid = ENV["SLURM_ARRAY_TASK_ID"]
#logfile = joinpath(@__DIR__, "log_$(jobid).out")
#filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
#logger = TeeLogger(global_logger(), filelogger)
#global_logger(logger)


println("Modules loaded. Time: $(t1-t0) s")

#n_dns = parse(Int,ARGS[1])
#n_les = parse(Int,ARGS[2])
#Re = parse(Float32,ARGS[3])

n_dns = Int(256)
n_les = Int(64)
Re = Float32(3000)
Δt = Float32(1.5e-3)
tsim = Float32(9e-3)

outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

# For running on a CUDA compatible GPU

T = Float32
ArrayType = CuArray
#device = x -> adapt(CuArray, x)
