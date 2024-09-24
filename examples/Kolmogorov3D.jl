if false
    #include("src/RikFlow.jl")
    include("../src/IncompressibleNavierStokes.jl")
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
using Statistics
using CUDA


# ## Setup
#
# Define a uniform grid with a steady body force field.

n = 64
axis_x = range(0.0, 6*pi, 3*n + 1)
axis_yz = range(0.0, 2*pi, n + 1)
setup = Setup(;
    x = (axis_x, axis_yz, axis_yz),
    Re = 1e2,
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sin(y),
    issteadybodyforce = true,
    ArrayType = CuArray,
);
ustart = random_field(setup, 0.0; A = 1e-1);

# ## Plot body force
#
# Since the force is steady, it is just stored as a field.
mean(setup.bodyforce[1],dims=3)[:,:] |> Array |> heatmap

# ## Solve unsteady problem

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

state, outputs = solve_unsteady(;
    setup,
    ustart = ustart,
    tlims = (0.0, 40),
    # Δt = 1e-2,
    # cfl = 0.4,
    processors = (
        #rtp = realtimeplotter(; setup, nupdate = 100),
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


heatmap(Array(mean(state.u[1],dims=3)[:,:]))


# Field plot
outputs.vort

# Energy history
outputs.ehist

# Energy spectrum
outputs.espec