# ⚠️ Scratch/demo scripts, updated mechanically to the IncompressibleNavierStokes >= 5 API at the
# upstream merge. They are NOT verified to run, and they did not run before the merge either:
# `OU_setup` takes `rng_seed`, never an `rng` object, so the `rng = Xoshiro(...)` below has always
# been an unknown-keyword error. The OU forcing itself lives in `src/ouforcer.jl` and is exercised
# by `lib/RikFlow/exp_square_HIT/tools/small_case.jl` and `lib/RikFlow/analysis/ou_replay.jl`;
# those are the checks that matter for it (handoff item 22a, item 27).

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
setup = RikFlow.rf_setup(;
    x = (axis, axis, axis),
    Re = 2e3,
    ArrayType = ArrayType,
);
setup = RikFlow.rf_setup(;
    x = (axis, axis, axis),
    Re = 2e3,
    ArrayType = ArrayType,
);

tlims = (T(0), T(2))
Δt = T(1e-3)

#ustart = random_field(setup, 0.0; A = 0.1);
ustart = vectorfield(setup);

state, outputs = solve_unsteady(;
    setup,
    start = (; u = ustart),
    params = RikFlow.rf_params(setup),
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
setup = RikFlow.rf_setup(;
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
    start = (; u = ustart),
    params = RikFlow.rf_params(setup),
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