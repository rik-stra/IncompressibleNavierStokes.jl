## Turbulent channel flow spin-up

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CUDA
using CUDSS
using RikFlow
using JLD2
#using LoggingExtras


#jobid = ENV["SLURM_JOB_ID"]
#logfile = joinpath(@__DIR__, "log_$(jobid).out")
#filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
#logger = TeeLogger(global_logger(), filelogger)
#global_logger(logger)


# Precision
T = Float64
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 15f  # 15f
# Grid
nx = 512
ny = 512
nz = 256

# small test
tsim = 5f
nx = 64
ny = 64
nz = 32

@info "Grid size: $(nx) x $(ny) x $(nz)"

# Steady streamwise driving force for the channel. Was `Setup(; bodyforce, issteadybodyforce)`;
# upstream removed both, along with `applybodyforce!`, so it is a force cache now.
# ⚠️ The trailing `t` argument is gone: upstream builds the field with `velocityfield`, whose
# `ufunc` takes `(dim, x...)` only. A steady force never used it.
channel_bodyforce(dim, x, y, z) = 1 * (dim == 1)

kwargs = (;
    boundary_conditions = (; u = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    )),
    Re = 180f,
    backend = CUDABackend(),
)

setup = rf_setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
);

psolver = psolver_transform(setup);

# Initial condition
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

ustart = velocityfield(setup, ustartfunc; psolver);
ArrayType = CuArray

@info "Solving DNS"
(; u, t), outputs = solve_unsteady(;
    # Steady driving force, formerly Setup(; bodyforce, issteadybodyforce).
    # Without it the channel is unforced and decays to rest, silently.
    force! = rf_bodyforce_navierstokes!,
    force_cache = rf_steady_force_cache(setup, channel_bodyforce),
    setup,
    Δt = 0.0005f,
    # LMWray3 by Rik's decision of 2026-09-11: stated, never inherited from the library default.
    method = LMWray3(; T = eltype(ustart)),
    start = (; u = ustart),
    params = rf_params(setup),
    docopy = false,
    tlims = (0f, tsim),
    
    processors = (;
        log = timelogger(; nupdate = 1000),
        #fields = fieldsaver(; nupdate = 1000, setup),
        ehist = realtimeplotter(;
                setup,
                plot = energy_history_plot,
                nupdate = 100,
                displayupdates = false,
                displayfig = false,
            ),
    ),
    psolver,
);

outdir = @__DIR__()*"/output/HF"
ispath(outdir) || mkpath(outdir)

filename = "$outdir/u_start_T$(Int(tsim))_$(nx)_$(ny)_$(nz).jld2"
u_start = u |> Array;
jldsave(filename; u_start);

# Plot
save(outdir*"/ehist_spinup_$(nx)_$(ny)_$(nz)_tspin$(tsim).png",outputs.ehist)

