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
n_les = Int(16)
Re = Float32(1000)
tburn = Float32(50)
# forcing
T_L = 0.05  # correlation time of the forcing
e_star = 0.2 # energy injection rate

outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)

seeds = (;
    dns = 123, # Initial conditions
    to = 234, # TO method online sampling
    ou = 333, # OU process
)

# For running on a CUDA compatible GPU

T = Float32
ArrayType = CuArray
#device = x -> adapt(CuArray, x)

# create forcing function
forcing_dict = (;
    T_L = T_L,
    Sigma_sq = e_star/T_L,
    rng = Xoshiro(seeds.ou),
    state = Array{Float32}(undef, 6),
)
# how to force in combination with runge-kutta??


## run DNS
rng_DNS = Xoshiro(seeds.dns)

# Parameters
get_params(nlesscalar) = (;
    D = 3,
    Re,
    lims = ( (T(0) , T(1)) , (T(0) , T(1)), (T(0),T(1)) ),
    tburn,
    Δt = T(1e-4),
    ndns = (n -> (n, n, n))(n_dns), # DNS resolution
    ArrayType,
    icfunc = (setup, psolver, rng) ->
        random_field(setup, zero(eltype(setup.grid.x[1])); kp = 20, psolver, rng),
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y)
)

params_train = (; get_params([n_les])...);
t3 = time()
u_start, ehist = spinnup(; params_train..., rng = rng_DNS);
u_start = Array.(u_start);

t4 = time()
println("HF simulation done. Time: $(t4-t3) s")
# Save filtered DNS data
filename = "$outdir/u_start_spinnup_$(n_dns)_Re$(Re)_tsim$(params_train.tburn).jld2"
jldsave(filename; u_start)

# Plot
save(outdir*"/ehist_$(n_dns)_Re$(Re)_tsim$(params_train.tburn).png",ehist)