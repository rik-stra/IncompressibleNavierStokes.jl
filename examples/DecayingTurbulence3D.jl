# # Decaying Homogeneous Isotropic Turbulence - 3D
#
# In this example we consider decaying homogeneous isotropic turbulence,
# similar to the cases considered in [Kochkov2021](@cite) and
# [Kurz2022](@cite). The initial velocity field is created randomly, but with a
# specific energy spectrum. Due to viscous dissipation, the turbulent features
# eventually group to form larger visible eddies.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes

# Output directory
outdir = joinpath(@__DIR__, "output", "DecayingTurbulence3D")
ispath(outdir) || mkpath(outdir)

# Floating point precision
T = Float64

# Array type
ArrayType = Array
## using CUDA; ArrayType = CuArray
## using AMDGPU; ArrayType = ROCArray
## using oneAPI; ArrayType = oneArray
## using Metal; ArrayType = MtlArray

# Reynolds number
Re = T(6_000)

# A 3D grid is a Cartesian product of three vectors
n = 32
lims = T(0), T(1)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1), LinRange(lims..., n + 1)

# Build setup and assemble operators
setup = Setup(; x, Re, ArrayType);

# Since the grid is uniform and identical for x, y, and z, we may use a
# specialized spectral pressure solver
psolver = psolver_spectral(setup);

# Initial conditions
ustart = random_field(setup; psolver);

# Solve unsteady problem
(; u, t), outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(1)),
    Δt = T(1e-3),
    psolver,
    processors = (
        ## rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 10),
        ehist = realtimeplotter(;
            setup,
            plot = energy_history_plot,
            nupdate = 10,
            displayfig = false,
        ),
        espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        ## anim = animator(; setup, path = "$outdir/solution.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = outdir, filename = "solution"),
        ## field = fieldsaver(; setup, nupdate = 10),
        log = timelogger(; nupdate = 100),
    ),
);

# ## Post-process
#
# We may visualize or export the computed fields `(u, p)`

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec

# Export to VTK
save_vtk(state; setup, filename = joinpath(outdir, "solution"), psolver)
