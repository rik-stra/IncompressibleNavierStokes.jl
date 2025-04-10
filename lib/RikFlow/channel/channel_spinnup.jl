## Turbulent channel flow

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CairoMakie
using CUDA
using CUDSS
using RikFlow
using JLD2
#using WGLMakie


# Precision
f = one(Float32)
# f = one(Float64)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 0.1*f
# Grid
# nx = 48
# ny = 24
# nz = 24
# nx = 64
# ny = 32
# nz = 32
nx = 512
ny = 512
nz = 256


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


psolver = default_psolver(setup);

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
        log = timelogger(; nupdate = 100),
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
