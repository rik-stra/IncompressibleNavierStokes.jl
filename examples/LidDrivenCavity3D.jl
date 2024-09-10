# # Lid-Driven Cavity - 3D
#
# In this example we consider a box with a moving lid. The velocity is initially at rest. The
# solution should reach at steady state equilibrium after a certain time. The same steady
# state should be obtained when solving a steady state problem.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Case name for saving results
outdir = joinpath(@__DIR__, "output", "LidDrivenCavity3D")

# Floating point type
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(1_000)

# A 3D grid is a Cartesian product of three vectors. Here we refine the grid
# near the walls.
x = cosine_grid(T(0), T(1), 25)
y = cosine_grid(T(0), T(1), 25)
z = LinRange(-T(0.2), T(0.2), 11)
plotgrid(x, y, z)

# Boundary conditions: horizontal movement of the top lid
U(dim, x, y, z, t) = dim() == 1 ? one(x) : dim() == 2 ? zero(x) : one(x) / 5
dUdt(dim, x, y, z, t) = zero(x)
boundary_conditions = (
    ## x left, x right
    (DirichletBC(), DirichletBC()),

    ## y rear, y front
    (DirichletBC(), DirichletBC(U, dUdt)),

    ## z bottom, z top
    (PeriodicBC(), PeriodicBC()),
)

# Build setup and assemble operators
setup = Setup(x, y, z; Re, boundary_conditions, ArrayType);

# Initial conditions
ustart = create_initial_conditions(setup, (dim, x, y, z) -> zero(x))

# Solve unsteady problem
(; u, t), outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(0.2)),
    Δt = T(1e-3),
    processors = (
        ## rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 50),
        ehist = realtimeplotter(; setup, plot = energy_history_plot, nupdate = 10),
        ## espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        ## anim = animator(; setup, path = "$outdir/solution.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 100, dir = outdir, filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 20),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields

# Export to VTK
save_vtk(state; setup, filename = joinpath(outdir, "solution"))

# Energy history
outputs.ehist
