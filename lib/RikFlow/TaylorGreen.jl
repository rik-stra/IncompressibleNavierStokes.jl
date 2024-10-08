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

qois = [["Z", 0, 10]] # energy and enstrophy
qoi_refs_folder = "C:/Users/rik/Documents/julia_code/IncompressibleNavierStokes.jl/lib/RikFlow/output/"
to_mode = "TRACK_REF" # "CREATE_REF" or "TRACK_REF"

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
w_hat2 = RikFlow.get_w_hat_from_u_hat(u_hat, to_setup);
Q = RikFlow.compute_QoI(u_hat, w_hat, to_setup, setup)



# ## Solve unsteady problem
if to_setup.to_mode == "CREATE_REF"
    method = RKMethods.RK44()  ## selects the standard solver
elseif to_setup.to_mode == "TRACK_REF"
    method = TOMethod(; to_setup) ### selects the special TO solver
end

#reset counter for QoI trajectory
if to_mode == "TRACK_REF" to_setup.qoi_ref.time_index[] = 1 end
state, outputs = solve_unsteady(;
    setup,
    ustart,
    method,
    tlims = (T(0), T(1)),
    Δt = T(1e-3),
    processors = (
        ## rtp = realtimeplotter(; setup, plot = fieldplot, nupdate = 10),
        qoihist = RikFlow.qoisaver(; setup, to_setup, nupdate = 1),                          # obtain QoI trajectories
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
    for i in 1:1 #to_setup.N_qois
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


fieldplot(state; fieldname = 1, setup)

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec
