# Adaptive-timestep probe of the Float64 / LMWray3 configuration on the production 512^3 HIT case.
#
# Purpose: let IncompressibleNavierStokes choose the step size and see what it picks, so the
# stability limit can be read off the output instead of assumed.
#
# Why. The production fixed step Δt = 2.5e-4 sits at **78% of the CFL = 1 limit** — measured on the
# archived initial condition, convection binding, 1.28x of headroom. RK44 ran there for 100 TU. A
# three-stage scheme has an imaginary-axis stability limit of sqrt(3) against RK44's 2*sqrt(2), a
# ratio of 0.61, and advection-dominated flow sits near the imaginary axis, which is the binding
# direction here. So LMWray3 at the production step may be outside its stability region. Those are
# textbook limits for a scalar model problem, not a bound for this discretization — hence a
# measurement.
#
# How to read the output. `solve_unsteady`'s adaptive branch sets
#
#     Δt = cfl * propose_timestep(force!, state, setup, params)
#
# every step, and `propose_timestep` returns the CFL = 1 step. So the Δt history reports what the
# flow itself demands, and the number that matters is printed at the end:
#
#     the ratio of the mean adaptive Δt to the production 2.5e-4
#
#   ratio >= 1  the production step is at or below what the solver would choose - comfortable
#   ratio <  1  the solver wants a smaller step than production uses - the fixed Δt is optimistic
#
# and, decisively: if the run **completes** with a finite field, LMWray3 is stable at this cfl. If
# it stops early, `solve_unsteady` broke out because the proposed step collapsed, which is what a
# diverging field does to `propose_timestep`.
#
# ⚠️ Two things this run is deliberately not. It is **not** a reference run: the OU chain advances
# by the varying adaptive Δt, so its forcing is not comparable to a fixed-Δt production run, and
# `freeze` must be 1 because a frozen body force cannot span adaptive steps (`solve_unsteady`
# asserts this; production uses freeze = 10). And the QoIs recorded here are physically meaningless
# at varying Δt — they are saved to exercise the filter, the masks, `curl` and `compute_QoI` at
# full 512^3 -> 64^3 scale on the GPU, which nothing else in this configuration does.

if false                                               #src
    include("../src/RikFlow.jl")                       #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
    using .IncompressibleNavierStokes                  #src
end

println("Loading modules...")
t0 = time()
using JLD2
using Observables      # `on`, for the step logger; RikFlow re-exports none of Observables
using Printf
using Random
using RikFlow
using IncompressibleNavierStokes
using CUDA
t1 = time()
println("Modules loaded. Time: $(t1-t0) s")

# ---------------------------------------------------------------------------------------------
# Parameters. One block, environment-overridable, nothing shadowed.
# ---------------------------------------------------------------------------------------------

T = Float64

# Device. Defaults to the GPU, which is what this is for. `CFL_DEVICE=cpu` exists so the script
# can be smoke-tested end to end before it reaches Snellius — gotcha #47 records this class of
# error (an undefined name inside a function body, invisible until the line executes) shipping to
# the cluster three times, and a parse check does not catch it.
const ONCPU = get(ENV, "CFL_DEVICE", "gpu") == "cpu"
ArrayType = ONCPU ? Array : CuArray
backend = ONCPU ? IncompressibleNavierStokes.CPU() : CUDABackend()

n_dns = parse(Int, get(ENV, "CFL_N_DNS", "512"))
n_les = parse(Int, get(ENV, "CFL_N_LES", "64"))
Re = T(2_000)

# Long enough for the adaptive step to settle and for an instability to show, short enough to be
# cheap. At Δt ~ 2.5e-4 this is a few hundred steps.
tsim = parse(T, get(ENV, "CFL_TSIM", "0.05"))

# IncompressibleNavierStokes' own default safety factor. Left at the library value on purpose:
# the point is to see what INS chooses, not to tune it.
cfl = parse(T, get(ENV, "CFL_SAFETY", "0.9"))

# Re-propose every step. The default is also 1; stated explicitly because a stability probe must
# not be averaging over a stale step size.
n_adapt = parse(Int, get(ENV, "CFL_N_ADAPT", "1"))

# The fixed production step, for the comparison printed at the end.
dt_production = T(2.5e-4)

# forcing
T_L = 0.01     # correlation time of the forcing
e_star = 0.1   # energy injection rate
k_f = sqrt(2)  # forcing wavenumber
# 🔴 Must be 1: `solve_unsteady` refuses to freeze a body force across adaptive steps, because the
# freeze interval is counted in solver steps and an adaptive step has no fixed size.
freeze = 1

seeds = (;
    dns = 123,  # DNS initial condition
    ou = 333,   # OU process
    to = 234,   # TO method online sampling
)

qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
lims = ((T(0), T(1)), (T(0), T(1)), (T(0), T(1)))
savefreq = 10   # DNS steps between QoI samples, as in production

outdir = @__DIR__() * "/output"
indir = @__DIR__() * "/output"
ispath(outdir) || mkpath(outdir)

# ---------------------------------------------------------------------------------------------
# Initial condition: the archived spin-up, promoted to Float64.
#
# Reused rather than re-run, for the reasons recorded in `2_HF_ref.jl`: the spin-up is a plain
# forced DNS so `∂` never touched it, and it seeds its OU chain from `ou_spin` rather than `ou`,
# so no forcing state carries over. Float32 -> Float64 promotion is exact.
# ---------------------------------------------------------------------------------------------

# `freeze_10_tsim4.0` is hardcoded on purpose: it names the *archived spin-up*, which was run at
# freeze = 10 for 4 TU. This probe's own `freeze` is 1, which is a property of this run and not of
# the file it starts from.
icfile = indir * "/u_start_spinnup_$(n_dns)_Re$(Re)_freeze_10_tsim4.0.jld2"

# 🔴 A synthetic initial condition is opt-in, never a fallback.
#
# An earlier version substituted one silently when the spin-up was missing. On a workstation that
# is a convenience; on Snellius it is a trap — the run completes, prints a confident stability
# verdict, and the only sign it measured nothing is a warning buried in the SLURM log. max|u| sets
# the CFL limit, so a synthetic field gives a synthetic Δt and a synthetic answer.
const SYNTHETIC = get(ENV, "CFL_SYNTHETIC", "0") == "1"

if isfile(icfile)
    println("Loading initial condition: $icfile")
    ustart = load(icfile, "u_start")
    ustart isa Tuple && (ustart = stack(ustart))
    println("  stored as $(eltype(ustart)) $(size(ustart)); promoting to $T")
    ustart = ArrayType{T}(ustart)
elseif SYNTHETIC
    @warn "SYNTHETIC initial condition (CFL_SYNTHETIC=1): this exercises the machinery and " *
          "measures NOTHING about production stability. max|u| sets the CFL limit, so a " *
          "synthetic field gives a synthetic Δt. Do not read a stability verdict from this run."
    ustart = nothing   # built below, once `dns` exists
else
    error(
        "No initial condition at:\n    $icfile\n\n" *
        "Copy the archived spin-up there, for example:\n" *
        "    scp <archive>/u_start_spinnup_$(n_dns)_Re$(Re)_freeze_10_tsim4.0.jld2 \\\n" *
        "        <host>:<repo>/lib/RikFlow/exp_square_HIT/output/\n\n" *
        "⚠️ `1_spinnup.jl` writes to output_spinnup/, not output/, so a spin-up produced on this " *
        "machine also has to be moved.\n\n" *
        "To exercise the script without it, set CFL_SYNTHETIC=1 — but that run measures nothing " *
        "about stability.",
    )
end

# ---------------------------------------------------------------------------------------------
# Setups and operators
# ---------------------------------------------------------------------------------------------

dns = rf_setup(; x = ntuple(a -> LinRange(lims[a]..., n_dns + 1), 3), Re, ArrayType, backend)
les = rf_setup(; x = ntuple(a -> LinRange(lims[a]..., n_les + 1), 3), Re, ArrayType, backend)
compression = n_dns ÷ n_les
@info "Grid" n_dns n_les compression

# Forcing is the right-hand side and its cache since the upstream merge, not a setup field.
force_cache = ou_force_cache(dns; T_L, e_star, k_f, rng_seed = seeds.ou, freeze)

psolver = psolver_spectral(dns)

if isnothing(ustart)
    ustart = velocityfield(
        dns,
        (dim, x, y, z) -> dim == 1 ? T(1) * sinpi(2 * y) * cospi(2 * z) :
                          dim == 2 ? T(1) * sinpi(2 * z) * cospi(2 * x) :
                                     T(1) * sinpi(2 * x) * cospi(2 * y);
        psolver,
    )
end

# `:CREATE_REF` allocates no output arrays, so the unknown step count of an adaptive run is fine.
to_setup_les = RikFlow.TO_Setup(; qois, to_mode = :CREATE_REF, ArrayType, setup = les, nstep = 1)

# What the adaptive stepper would choose from the initial field, before any stepping. Printed so
# the first proposal is on the record even if the run stops immediately.
dt0 = cfl * IncompressibleNavierStokes.propose_timestep(
    ou_navierstokes!, (; u = ustart), dns, rf_params(dns))
@printf("first proposed Δt = %.4e  (cfl = %.2f; production fixed Δt = %.4e)\n",
    dt0, cfl, dt_production)

# ---------------------------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------------------------

"Per-step record of `(n, t, max|u|)`. Δt is recovered by differencing `t`."
steplogger() = processor() do state
    rec = NamedTuple{(:n, :t, :umax),Tuple{Int,Float64,Float64}}[]
    on(state) do (; u, t, n)
        push!(rec, (; n, t = Float64(t), umax = Float64(maximum(abs, u))))
    end
    state[] = state[]
    rec
end

# ---------------------------------------------------------------------------------------------
# Solve, adaptively
# ---------------------------------------------------------------------------------------------

@info "Solving DNS with adaptive Δt" stepper = "LMWray3" T cfl tsim
t3 = time()
(; u, t), outputs = solve_unsteady(;
    setup = dns,
    start = (; u = ustart),
    force! = ou_navierstokes!,
    force_cache,
    params = rf_params(dns),
    method = LMWray3(; T),
    docopy = false,
    tlims = (T(0), tsim),
    Δt = nothing,            # <- adaptive: INS chooses the step
    cfl,
    n_adapt_Δt = n_adapt,
    processors = (;
        f = RikFlow.filtersaver(dns, [les], (FaceAverage(),), [compression], [to_setup_les];
            nupdate = savefreq, n_plot = 1_000_000,
            checkpoints = nothing, checkpoint_name = nothing),
        steps = steplogger(),
        log = timelogger(; nupdate = 50),
    ),
    psolver,
)
t4 = time()
println("Adaptive DNS done. Time: $(t4-t3) s")

# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

rec = outputs.steps
qoi_hist = stack(outputs.f.data[1].qoi_hist)
ua = Array(u)
finite = all(isfinite, ua)
reached = Float64(t) >= Float64(tsim) * (1 - 1e-9)

ts = [x.t for x in rec]
# 🔴 Drop the final interval. `solve_unsteady` clips the last step with
# `Δt_step = min(Δt, tend - stepper.t)`, so it is bounded by how much time is left rather than by
# stability. Including it reports `tsim` as a step size whenever the proposed step is the larger of
# the two — on a short smoke run that turned "1 step of 0.01" into a confident "40x production".
dts_all = length(ts) > 1 ? diff(ts) : Float64[]
dts = length(dts_all) > 1 ? dts_all[1:end-1] : dts_all

# Below this the Δt statistics measure nothing.
const MIN_STEPS_FOR_VERDICT = 10

println()
@printf("steps taken            %d\n", isempty(rec) ? 0 : rec[end].n)
@printf("t reached              %.6f of %.6f%s\n", Float64(t), Float64(tsim),
    reached ? "" : "   <- STOPPED EARLY")
@printf("field finite           %s\n", finite)
@printf("max|u| final           %.6g\n", finite ? maximum(abs, ua) : NaN)
@printf("qoi_hist               %s  (every %d steps)\n", string(size(qoi_hist)), savefreq)
if isempty(dts)
    println()
    println("🔴 No unclipped steps were taken, so there is no Δt to report. The proposed step")
    println("   exceeded the whole simulated interval — raise CFL_TSIM.")
elseif length(dts) < MIN_STEPS_FOR_VERDICT
    @printf("Δt  min / mean / max   %.4e  %.4e  %.4e   (only %d unclipped steps)\n",
        minimum(dts), sum(dts) / length(dts), maximum(dts), length(dts))
    println()
    @printf("🔴 Only %d unclipped steps — too few for a verdict. Raise CFL_TSIM to at least %.3g\n",
        length(dts), MIN_STEPS_FOR_VERDICT * maximum(dts) * 2)
    println("   so the adaptive step has room to settle.")
else
    @printf("Δt  min / mean / max   %.4e  %.4e  %.4e   (%d unclipped steps)\n",
        minimum(dts), sum(dts) / length(dts), maximum(dts), length(dts))
    ratio = (sum(dts) / length(dts)) / dt_production
    @printf("mean Δt / production   %.3f\n", ratio)
    println()
    if !(reached && finite)
        println("🔴 The run stopped early or went non-finite: LMWray3 is UNSTABLE at cfl = $cfl")
        println("   on this configuration. `solve_unsteady` breaks out when the proposed step")
        println("   collapses, which is what a diverging field does to propose_timestep.")
    elseif ratio >= 1
        println("LMWray3 is stable here, and the adaptive step is at or above the production")
        println("   fixed Δt — so 2.5e-4 is inside what the solver would choose. Comfortable.")
    else
        @printf("⚠️  LMWray3 is stable at cfl = %.2f, but the adaptive step averages %.3fx the\n",
            cfl, ratio)
        @printf("   production fixed Δt. A fixed 2.5e-4 is then above what the solver picks for\n")
        @printf("   itself; running LMWray3 at the production step is not supported by this.\n")
    end
end

filename = "$outdir/cfl_probe_$(n_dns)_f64_lmwray3.jld2"
jldsave(filename;
    t = ts,
    umax = [x.umax for x in rec],
    n = [x.n for x in rec],
    qoi_hist,
    params = (; n_dns, n_les, Re, tsim, cfl, n_adapt, dt_production, dt_first = dt0,
                savefreq, stepper = "LMWray3", precision = string(T),
                forcing = (; T_L, e_star, k_f, freeze, rng_seed = seeds.ou)),
    summary = (; nsteps = isempty(rec) ? 0 : rec[end].n, tfinal = Float64(t),
                 reached, finite,
                 dtmin = isempty(dts) ? NaN : minimum(dts),
                 dtmean = isempty(dts) ? NaN : sum(dts) / length(dts),
                 dtmax = isempty(dts) ? NaN : maximum(dts)))
println("\nWritten: $filename")
