if false
    include("src/RikFlow.jl")
    include("../../src/IncompressibleNavierStokes.jl")
end
using RikFlow

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
using JLD2

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

# Builed TO_setup
qois = [["Z",0,10],["E", 0, 10]] # energy and enstrophy
qoi_refs_folder = "C:/Users/rik/Documents/julia_code/IncompressibleNavierStokes.jl/lib/RikFlow/output/"
to_mode = "TRACK_REF" # "CREATE_REF" or "TRACK_REF"

to_setup = RikFlow.TO_Setup(; qois, qoi_refs_folder, to_mode, ArrayType, setup)


# Since the grid is uniform and identical for x, y, and z, we may use a
# specialized spectral pressure solver
psolver = psolver_spectral(setup);

# Initial conditions
ustart = random_field(setup; psolver);

# Solve unsteady problem
if to_setup.to_mode == "CREATE_REF"
    method = RKMethods.RK44()  ## selects the standard solver
elseif to_setup.to_mode == "TRACK_REF"
    method = TOMethod(; to_setup) ### selects the special TO solver
end


if to_mode == "TRACK_REF" to_setup.qoi_ref.time_index[] = 1 end #reset counter for QoI trajectory
(; u, t), outputs = solve_unsteady(;
    setup,
    ustart,
    method,
    tlims = (T(0), T(1)),
    Δt = T(1e-3),
    psolver,
    processors = (
        ## rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 10),
        qoihist = RikFlow.qoisaver(; setup, to_setup, nupdate = 1),
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
# ## save QoI trajectories to file
q = stack(outputs.qoihist)
if to_setup.to_mode == "CREATE_REF"
    # put a sinewave in the QoI trajectory
    for i in 1:2 #to_setup.N_qois
        s = -sin.((1:1001) ./ (8*i * π))*0.05*(maximum(q[i,:])-minimum(q[i,:]))
        q[i,:]+=s
    end
    jldsave(to_setup.qoi_refs_folder*"QoIhist.jld2"; q)
end


f = Figure()
axs = [Axis(f[1, i]) for i in 1:size(q, 1)]
p1 = plot!(axs[1],q[1,:])
p1 = plot!(axs[1],to_setup.qoi_ref.qoi_trajectories[1,:])
p2 = plot!(axs[2],q[2,:])
p2 = plot!(axs[2],to_setup.qoi_ref.qoi_trajectories[2,:])

# We may visualize or export the computed fields `(u, p)`

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec