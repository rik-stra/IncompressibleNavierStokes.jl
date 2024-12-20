if false
    include("src/RikFlow.jl")
    include("../../src/IncompressibleNavierStokes.jl")
end
#using OUForcer
using IncompressibleNavierStokes
using CairoMakie
using CUDA; ArrayType = CuArray
#ArrayType = Array
using Random

T = Float32
outdir = joinpath(@__DIR__, "output", "3D_forced")
ispath(outdir) || mkpath(outdir)


# ## Setup
#
# Define a uniform grid with a steady body force field.

n = 128
axis = range(0.0, 1., n + 1)
setup = Setup(;
    x = (axis, axis, axis),
    Re = 2e3,
    ou_bodyforce = (; T_L = 0.01, e_star = 0.1, k_f = sqrt(2), rng = Xoshiro(25)),
    ArrayType = ArrayType,
);
setup = Setup(;
    x = (axis, axis, axis),
    Re = 2e3,
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y),
    issteadybodyforce = true,
    ArrayType = ArrayType,
);

tlims = (T(0), T(2))
Δt = T(1e-3)

#ustart = random_field(setup, 0.0; A = 0.1);
ustart = vectorfield(setup);

state, outputs = solve_unsteady(;
    setup,
    ustart = ustart,
    tlims = tlims,
    #Δt = Δt,
    processors = (
        #ehist = realtimeplotter(;
        #    setup,
        #    plot = energy_history_plot,
        #    nupdate = 10,
        #    displayupdates = false,
        #    displayfig = false,
        #),
        espec = realtimeplotter(;
            setup,
            plot = energy_spectrum_plot,
            nupdate = 10,
            displayupdates = true,
            displayfig = true,
        ),
        log = timelogger(; nupdate = 10),
    ),
);


# plot a z-slice of the velocity field
heatmap(Array(state.u[1])[ :,30, :])

######
## Test 2D
######

n = 512
axis = range(0.0, 1., n + 1)
setup = Setup(;
    x = (axis, axis),
    Re = 7e3,
    ou_bodyforce = (; T_L = 0.02, e_star = 0.01, k_f = 2*sqrt(2), rng = Xoshiro(25)),
    ArrayType = ArrayType,
);


tlims = (T(0), T(7))
Δt = T(1e-3)

#ustart = random_field(setup, 0.0; A = 0.1);
ustart = vectorfield(setup);

state, outputs = solve_unsteady(;
    setup,
    ustart = ustart,
    tlims = tlims,
    Δt = Δt,
    processors = (
        #ehist = realtimeplotter(;
        #    setup,
        #    plot = energy_history_plot,
        #    nupdate = 10,
        #    displayupdates = false,
        #    displayfig = false,
        #),
        flow = realtimeplotter(;
            setup,
             plot = fieldplot,
             nupdate = 10,
             displayupdates = true,
             displayfig = true,
         ),
        espec = realtimeplotter(;
            setup,
            plot = energy_spectrum_plot,
           nupdate = 100,
           displayupdates = false,
            displayfig = false,
        ),
        log = timelogger(; nupdate = 10),
    ),
);

heatmap(Array(state.u[1]))
outputs.espec