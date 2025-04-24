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
#using WGLMakie


# Precision
f = one(Float32)
# f = one(Float64)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 5f
# Grid
# nx = 48
# ny = 24
# nz = 24
# nx = 64
# ny = 32
# nz = 32
nx = 128
ny = 64
nz = 64
Δt = 0.002f

nx_les = 64
ny_les = 32
nz_les = 32

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

les_setup = Setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz_les + 1)
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

qois = [["Z",0,6],["E", 0, 6],["Z",7,16],["E", 7, 16]];
ArrayType = CuArray


# Number of time steps to save
nt = round(Int, tsim / Δt)
Δt = tsim / nt

to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = les_setup, 
    nstep=nt);


@info "Solving DNS"
# Solve DNS and store filtered quantities
(; u, t), outputs = solve_unsteady(;
    setup,
    ustart,
    docopy = false,
    tlims = (0f, tsim),
    Δt,
    processors = (;
        f = RikFlow.filtersaver(
            setup,
            [les_setup,],
            (FaceAverage(),),
            [2,],
            [to_setup_les,];
            nupdate = 10,
            n_plot = 100,
            #checkpoints,
            #checkpoint_name,
        ),
        log = timelogger(; nupdate = 10),
    ),
    psolver,
);

q_data = stack(outputs.f.data[1].qoi_hist)
lines(q_data[1,:])
lines(q_data[2,:])
lines(q_data[3,:])
lines(q_data[4,:])

size(u)
total_kinetic_energy(u, setup)
q_data[2,end]
q_data[4,end]

heatmap(u[:,:, 1,1] |> Array)

u_x = u[:,:,:,1] |> Array
using Statistics
u_x = mean(u_x, dims = [1,3])

fig = Figure()
ax = Axis(fig[1, 1], title = "u_x", xscale = log10)
lines!(180*setup.grid.x[2][1:33]|> Array, u_x[1:33] )
display(fig)

180*setup.grid.x[2][1:33]|> Array