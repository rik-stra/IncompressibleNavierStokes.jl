# Steady streamwise driving force for the channel. Was `Setup(; bodyforce, issteadybodyforce)`;
# upstream removed both, along with `applybodyforce!`, so it is a force cache now.
# ⚠️ The trailing `t` argument is gone: upstream builds the field with `velocityfield`, whose
# `ufunc` takes `(dim, x...)` only. A steady force never used it.
channel_bodyforce(dim, x, y, z) = 1 * (dim == 1)

using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

T = Float64
f = one(T)

nx = 512
ny = 512
nz = 256


Re = 180f
Δt = 0.0005f
tburn = 150*Δt


# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

backend = CUDABackend()


dns = rf_setup(;
    boundary_conditions = (; u = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    )),
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    Re,
    backend,
    ArrayType = CuArray,
    );

psolver = psolver_transform(dns);
CUDA.synchronize()
t0 = time()
psolver = psolver_transform(dns);
CUDA.synchronize()
t1 = time()

println("Time to create pressure solver: $(t1-t0) s")

Re_tau = 180f
Re_m = 2800f
Re_ratio = Re_m / Re_tau

ustartfunc = let
    Lx = xlims[2] - xlims[1]
    Ly = ylims[2] - ylims[1]
    Lz = zlims[2] - zlims[1]
    C = 9f / 8 * Re_ratio
    E = 1f / 10 * Re_ratio # 10% of average mean velocity
    function icfunc(dim, x, y, z)
        ux =
            C * (1 - (y - Ly / 2)^8) +
            E * Lx / 2 * sinpi(y) * cospi(4 * x / Lx) * sinpi(2 * z / Lz)
        uy = -E * (1 - cospi(y)) * sinpi(4 * x / Lx) * sinpi(2 * z / Lz)
        uz = -E * Lz / 2 * sinpi(4 * x / Lx) * sinpi(y) * cospi(2 * z / Lz)
        (dim == 1) * ux + (dim == 2) * uy + (dim == 3) * uz
    end
end

ustart = velocityfield(dns, ustartfunc; psolver);

(; u, t), outputs =
        solve_unsteady(;
        # Steady driving force, formerly Setup(; bodyforce, issteadybodyforce).
        # Without it the channel is unforced and decays to rest, silently.
        force! = rf_bodyforce_navierstokes!,
        force_cache = rf_steady_force_cache(dns, channel_bodyforce),
        setup = dns, 
        # Upstream changed the default from RKMethods.RK44 to LMWray3; pinned.
        method = RKMethods.RK44(; T = eltype(ustart)),
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
write(io, "Channel_HF n_dns: $nx $ny $nz, elapsed_time: $elapsed  \n")
close(io)
