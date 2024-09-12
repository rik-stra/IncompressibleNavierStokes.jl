if false
    include("src/RikFlow.jl")
    include("../../src/IncompressibleNavierStokes.jl")
end

using RikFlow


# # Taylor-Green vortex - 3D
#
# In this example we consider the Taylor-Green vortex.

#md using CairoMakie
using GLMakie #!md
using IncompressibleNavierStokes
using JLD2

# Floating point precision
T = Float64

# ## Array type
#
# Running in 3D is heavier than in 2D.
# If you are running this on a CPU, consider using multiple threads by
# starting Julia with `julia -t auto`, or
# add `-t auto` to # the `julia.additionalArgs` # setting in VSCode.
ArrayType = Array
## using CUDA; ArrayType = CuArray

## TO parameters

qois = [["E", 0, 5],["E",6,10], ["Z", 0, 5],["Z",6,10]]
qoi_refs_folder = "C:/Users/rik/Documents/julia_code/IncompressibleNavierStokes.jl/lib/RikFlow/output/"
to_mode = "CREATE_REF" # "CREATE_REF" or "TRACK_REF"

# ## Setup

n = 32
r = range(T(0), T(1), n + 1)
setup = Setup(; x = (r, r, r), Re = T(1e3), ArrayType);
psolver = psolver_spectral(setup);

# Initial conditions
U(dim, x, y, z) =
    if dim == 1
        sinpi(2x) * cospi(2y) * sinpi(2z) / 2
    elseif dim == 2
        -cospi(2x) * sinpi(2y) * sinpi(2z) / 2
    else
        zero(x)
    end
ustart = velocityfield(setup, U, psolver);

# initialize TO_setup
to_setup = RikFlow.TO_Setup(; qois, qoi_refs_folder, to_mode, ArrayType, setup)

# test some stuff
u_hat = RikFlow.get_u_hat(ustart, setup);
w_hat = RikFlow.get_w_hat(ustart, setup);

Q = RikFlow.compute_QoI(u_hat, w_hat, to_setup, setup)



# ## Solve unsteady problem

state, outputs = solve_unsteady(;
    setup,
    ustart,
    tlims = (T(0), T(1.0)),
    Δt = T(1e-3),
    processors = (
        ## rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 10),
        qoihist = RikFlow.qoisaver(; setup, to_setup, nupdate = 1),
        #espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
        ## anim = animator(; setup, path = "$outdir/solution.mkv", nupdate = 20),
        ## vtk = vtk_writer(; setup, nupdate = 10, dir = outdir, filename = "solution"),
        log = timelogger(; nupdate = 100),
    ),
    psolver,
);

# ## save QoI trajectories to file
q = stack(outputs.qoihist)
if to_setup.to_mode == "CREATE_REF"
    # put a sinewave in the QoI trajectory
    s = sin.((1:1001) ./ (8 * π))*0.2*(maximum(q[1,:])-minimum(q[1,:]))
    q[1,:]+=s
    jldsave(to_setup.qoi_refs_folder*"QoIhist.jld2"; q)
end

lines(s)
lines(q[1,:])


fieldplot(state; fieldname = 1, setup)

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec
