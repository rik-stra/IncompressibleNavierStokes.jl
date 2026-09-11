using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

n_dns = Int(512)
T = Float64
Re = T(2_000)
Δt = T(0.00025)
tburn = 150*Δt

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

dns = rf_setup(;
        x = ntuple(α -> LinRange(lims[α]..., n_dns + 1), 3),
        Re,
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
        # 🔴 The OU forcing used to ride along in the setup (`Setup(; ou_bodyforce)`), so this
        # benchmark was forced without saying so. Upstream's setup has no forcing slot; without
        # these two lines the timing would silently be for an unforced solve, which is not the
        # per-step cost this file exists to measure.
        force! = ou_navierstokes!,
        force_cache = ou_force_cache(dns; ou_bodyforce...),
        # LMWray3 + Float64 is the production configuration (Rik, 2026-09-11), so that is what the
        # timing has to be for.
        method = LMWray3(; T = eltype(ustart)),
        start = (; u = ustart),
        params = rf_params(dns), 
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
