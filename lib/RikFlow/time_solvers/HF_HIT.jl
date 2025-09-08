using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

n_dns = Int(512)
Re = Float32(2_000)
Δt = Float32(0.00025)
tburn = 150*Δt

T = Float32


lims = ( (T(0) , T(1)) , (T(0) , T(1)), (T(0),T(1)) )
# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber
freeze = 10 # number of time steps to freeze the forcing

seeds = (;
    ou_spin = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)

backend = CUDABackend()

ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou_spin )

dns = Setup(;
        x = ntuple(α -> LinRange(lims[α]..., n_dns + 1), 3),
        Re,
        ou_bodyforce,
        backend,
        ArrayType = CuArray,
    );

psolver = psolver_spectral(dns)
CUDA.synchronize()
t0 = time()
psolver = psolver_spectral(dns)
CUDA.synchronize()
t1 = time()

println("Time to create pressure solver: $(t1-t0) s")

ustart = vectorfield(dns);
(; u, t), outputs =
        solve_unsteady(;
        setup = dns, 
        ustart, 
        tlims = (T(0), tburn),
        docopy = false,
        Δt,
        processors = (;
            log = timelogger(; nupdate = 200),
            timer = RikFlow.solver_timer(; n_steps = 100, n_warmup = 10),
        ),
        psolver);

elapsed = outputs.timer[2]-outputs.timer[1]
println("elapsed time:", elapsed, " s")

io = open(@__DIR__()*"/time_results.txt", "a")
write(io, "HIT_HF n_dns: $n_dns, elapsed_time: $elapsed  \n")
close(io)
