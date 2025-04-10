## Turbulent channel flow

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CairoMakie
using CUDA
#using CUDSS
using RikFlow
using JLD2
using LoggingExtras
#using WGLMakie

jobid = ENV["SLURM_JOB_ID"]
#taskid = ENV["SLURM_ARRAY_TASK_ID"]
logfile = joinpath(@__DIR__, "log_$(jobid).out")
filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
logger = TeeLogger(global_logger(), filelogger)
global_logger(logger)


# Precision
T = Float32
f = one(T)
# f = one(Float64)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = Float32(0.1)
# Grid
nx = 256 #512
ny = 256 #512
nz = 128 #256

@info "Grid size: $(nx) x $(ny) x $(nz)"

kwargs = (;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re = 180f,
    bodyforce = (dim, x, y, z, t) -> 1 * (dim == 1),
    issteadybodyforce = true,
    backend = CUDABackend(),
)

setup = Setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
);

@info "factorize psolver ..."
flush(stdout)
#@time psolver = default_psolver(setup);
psolver = psolver_cg(setup);
@info "factorize psolver done"
flush(stdout)

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

@info "Solving DNS"
# Solve DNS and store filtered quantities
(; u, t), outputs = solve_unsteady(;
    setup,
    ustart,
    docopy = false,
    tlims = (0f, tsim),
    
    processors = (;
        log = timelogger(; nupdate = 1),
        ehist = realtimeplotter(;
                setup,
                plot = energy_history_plot,
                nupdate = 10,
                displayupdates = false,
                displayfig = false,
            ),
    ),
    psolver,
);

outdir = @__DIR__()*"/output"
ispath(outdir) || mkpath(outdir)

filename = "$outdir/u_start_$(nx)_$(ny)_$(nz)_tspin$(tsim).jld2"
u_start = u |> Array;
jldsave(filename; u_start);

# Plot
save(outdir*"/ehist_spinup_$(nx)_$(ny)_$(nz)_tspin$(tsim).png",outputs.ehist)

# a = load(filename, "u_start")
# heatmap(a[10,:,:,3])
