if false
    include("src/RikFlow.jl")
    include("../../src/IncompressibleNavierStokes.jl")
end

# # Kolmogorov flow (3D)
#
# The Kolmogorov flow in a periodic box ``\Omega = [0, 3]x[0, 1]x[0, 1]`` is initiated
# via the force field
#
# ```math
# f(x, y) =
# \begin{pmatrix}
#     A \sin(\pi k y) \\
#     0 \\
#     0
# \end{pmatrix} - \mu \tilde{u}
# ```
#
# where `k` is the wavenumber where energy is injected, and ``\mu \tilde{u}`` is a linear damping term acting only on large scales in u, via a sharp cutoff filter.

# ## Packages
#
# We just need the `IncompressibleNavierStokes` and a Makie plotting package.

using CairoMakie
# using GLMakie #!md
using IncompressibleNavierStokes
using JLD2
using CUDA; ArrayType = CuArray
using RikFlow
using Statistics

T = Float64
outdir = joinpath(@__DIR__, "output", "3D_kolmogorov")
ispath(outdir) || mkpath(outdir)


# ## Setup
#
# Define a uniform grid with a steady body force field.

n = 32
axis_x = range(0.0, 3., 3*n + 1)
axis_yz = range(0.0, 1., n + 1)
setup = Setup(;
    x = (axis_x, axis_yz, axis_yz),
    Re = 5e3,
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y),
    issteadybodyforce = true,
    ArrayType = ArrayType,
);

# Builed TO_setup
qois = [["Z",0,4],["E", 0, 4],["Z",5,10],["E", 5, 10]] # energy and enstrophy
qoi_refs_folder = outdir
to_mode = "TRACK_REF" # "CREATE_REF" or "TRACK_REF"

tlims = (T(0), T(15))
Δt = T(5e-3)
nstep = round(Int, (tlims[2] - tlims[1]) / Δt)

to_setup = RikFlow.TO_Setup(; qois, qoi_refs_folder, to_mode, ArrayType, setup, nstep)

ustart = random_field(setup, 0.0; A = 0.1);

# ## Plot body force
#
# Since the force is steady, it is just stored as a field.
mean(setup.bodyforce[1],dims=3)[:,:] |> Array |> heatmap


vortplot(state; setup) = begin
    state isa Observable || (state = Observable(state))
    ω = lift(state) do state
        vx = mean(state.u[1],dims=3)[1:end-1, :]
        vy = mean(state.u[2],dims=3)[:, 1:end-1]
        ω = -diff(vx; dims = 2) + diff(vy; dims = 1)
        Array(ω)
    end
    heatmap(ω; figure = (; size = (900, 350)), axis = (; aspect = DataAspect()))
end
#vortplot(state; setup)

# ## Solve unsteady problem
if to_setup.to_mode == "CREATE_REF"
    method = RKMethods.RK44()  ## selects the standard solver
elseif to_setup.to_mode == "TRACK_REF"
    method = TOMethod(; to_setup) ### selects the special TO solver
end
if to_mode == "TRACK_REF" to_setup.qoi_ref.time_index[] = 1 end #reset counter for QoI trajectory
state, outputs = solve_unsteady(;
    setup,
    method,
    ustart = ustart,
    tlims = tlims,
    Δt = Δt,
    processors = (
        #rtp = realtimeplotter(; setup, nupdate = 100),
        qoihist = RikFlow.qoisaver(; setup, to_setup, nupdate = 1),
        ehist = realtimeplotter(;
            setup,
            plot = energy_history_plot,
            nupdate = 10,
            displayupdates = false,
            displayfig = false,
        ),
        espec = realtimeplotter(;
            setup,
            plot = energy_spectrum_plot,
            nupdate = 10,
            displayupdates = false,
            displayfig = false,
        ),
        vort = realtimeplotter(;
            setup,
            plot = vortplot,
            nupdate = 10,
            displayupdates = true,
            displayfig = true,
        ),
        log = timelogger(; nupdate = 10),
    ),
);

# ## Post-process
# ## save QoI trajectories to file
q = stack(outputs.qoihist)
if to_setup.to_mode == "CREATE_REF"
    # put a sinewave in the QoI trajectory
    #for i in 1:2 #to_setup.N_qois
    #    s = -sin.((1:1001) ./ (8*i * π))*0.05*(maximum(q[i,:])-minimum(q[i,:]))
    #   q[i,:]+=s
    #end
    jldsave(to_setup.qoi_refs_folder*"/QoIhist.jld2"; q)
end


f = Figure()
axs = [Axis(f[i ÷ 2, i%2]) for i in 0:size(q, 1)-1]
for i in 1:size(q, 1)
    plot!(axs[i],q[i,:])
    if to_mode == "TRACK_REF" plot!(axs[i],to_setup.qoi_ref.qoi_trajectories[i,:]) end
end
display(f)

g = Figure()
axs = [Axis(g[i ÷ 2, i%2], title = "dQ $(i)") for i in 0:size(q, 1)-1]
for i in 1:size(q, 1)
    plot!(axs[i],to_setup.outputs.dQ[10:end,i])
end
display(g)

g = Figure()
axs = [Axis(g[i ÷ 2, i%2], title = "tau $(i)") for i in 0:size(q, 1)-1]
for i in 1:size(q, 1)
    plot!(axs[i],to_setup.outputs.tau[10:end,i])
end
display(g)

heatmap(Array(mean(state.u[3],dims=3)[:,:]))


# Field plot
outputs.vort

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec