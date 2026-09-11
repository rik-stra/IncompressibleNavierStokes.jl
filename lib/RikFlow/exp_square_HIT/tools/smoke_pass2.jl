#=
    smoke_pass2.jl — smoke test for the channel / taylor-green / time_solvers port.

Checklist item 22, at the validation level Rik set for it: **smoke only**. Tiniest viable grid,
CPU, a handful of steps, assert it runs and the output is finite. No golden data, no bit
comparison — their archives are deferred and nothing downstream reads their numbers right now.
The bit-level treatment in `small_case.jl` is for the HIT case alone, because that is the one
guarding data about to be regenerated.

## Why this rather than "it loads and precompiles"

Item 22 allows falling back to "loads" if a smoke run is not cheap to stand up. It is cheap, and
loading would miss everything that actually broke in this port. These scripts are the only ones
that exercise:

  - **Dirichlet walls.** Every HIT case is periodic in all three directions, so the
    field-keyed `boundary_conditions.u` change is only half-tested by the HIT path.
  - **`psolver_transform`.** The FFT/DCT solver the channel needs because `psolver_spectral`
    requires periodicity everywhere. Ten call sites, none of them in HIT.
  - **The steady driving force.** `Setup(; bodyforce, issteadybodyforce)` is gone; the channel is
    driven by `rf_steady_force_cache` now. 🔴 If that is left unwired the channel simply decays to
    rest — no error, no warning, just a wrong answer. Nine solves needed it wired by hand.
  - **The eddy-viscosity models.** `Smagorinsky` and `WALE` through
    `rf_eddyvisc_force_cache`/`rf_eddyvisc_navierstokes!`, replacing `setup.closure_model` + `θ`.

Each of those is a place where a mistake is silent or near-silent, which is exactly what a smoke
test is for.

    julia --startup-file=no --project=lib/RikFlow \
        lib/RikFlow/exp_square_HIT/tools/smoke_pass2.jl
=#

using IncompressibleNavierStokes
using Printf
using RikFlow

const T = Float32
const NSTEP = 20

"A tiny wall-bounded channel: periodic in x and z, Dirichlet in y."
function channel_setup(; n = 8)
    rf_setup(;
        x = (
            range(T(0), T(4) * T(pi), n + 1),
            range(T(-1), T(1), n + 1),
            range(T(0), T(2) * T(pi), n + 1),
        ),
        Re = T(180),
        boundary_conditions = (;
            u = (
                (PeriodicBC(), PeriodicBC()),
                (DirichletBC(), DirichletBC()),
                (PeriodicBC(), PeriodicBC()),
            )
        ),
    )
end

"A tiny triply periodic box, standing in for Taylor-Green and the HIT LES."
periodic_setup(; n = 8) =
    rf_setup(; x = ntuple(_ -> range(T(0), T(1), n + 1), 3), Re = T(2000))

"The channel's steady streamwise driving force, matching the scripts."
channel_bodyforce(dim, x, y, z) = 1 * (dim == 1)

"""
    smoke(name, setup, force!, force_cache; psolver, nstep)

Run `nstep` steps and assert the result is finite and not identically zero. Returns the final
field on success and `nothing` on failure, so one bad case does not hide the rest and so callers
can check that two configurations actually differ.
"""
function smoke(name, setup, force!, force_cache; psolver, nstep = NSTEP)
    Δt = T(1e-3)
    # 🔴 The initial field must have non-zero velocity gradients, and this is not a detail.
    # A uniform field has zero strain, so every eddy-viscosity model returns exactly zero and
    # Smagorinsky, WALE and no-closure all produce byte-identical output - the test passes while
    # exercising none of the closures it names. The first version of this file did exactly that.
    ustart = velocityfield(
        setup,
        (dim, x, y, z) -> dim == 1 ? T(0.1) * sinpi(2 * y) * cospi(2 * z) :
                          dim == 2 ? T(0.05) * sinpi(2 * z) * cospi(2 * x) :
                                     T(0.05) * sinpi(2 * x) * cospi(2 * y);
        psolver,
        doproject = false,
    )
    try
        (; u, t), _ = solve_unsteady(;
            setup,
            method = RKMethods.RK44(; T = eltype(ustart)),
            start = (; u = ustart),
            force!,
            force_cache,
            params = rf_params(setup),
            tlims = (T(0), Δt * nstep),
            Δt,
            psolver,
        )
        ua = Array(u)
        all(isfinite, ua) || error("non-finite entries in u")
        m = maximum(abs, ua)
        m > 0 || error("u is identically zero after $nstep steps")
        @printf("  %-34s ok    max|u| = %.6e\n", name, m)
        ua
    catch e
        @printf("  %-34s FAIL  %s\n", name, first(sprint(showerror, e), 160))
        nothing
    end
end

"""
    differs(name, a, b)

Assert two smoke results are not the same field.

🔴 This is the positive control, and without it the suite is decorative. A uniform initial field
has zero strain, so every eddy-viscosity model contributes exactly zero and a closure run is
byte-identical to a no-closure run — the cases all "pass" while testing nothing. The structured
initial field in `smoke` is what gives them something to act on, and this is what checks that they
did. Same lesson as gotcha #47: a detector that has never seen an offender is not a test.
"""
function differs(name, a, b)
    if isnothing(a) || isnothing(b)
        @printf("  %-34s SKIP  a prerequisite case failed\n", name)
        return false
    end
    d = maximum(abs, Float64.(a) .- Float64.(b))
    s = maximum(abs, Float64.(b))
    rel = s == 0 ? 0.0 : d / s
    pass = rel > 1e-6
    @printf("  %-34s %s  relative difference %.4e\n", name, pass ? "ok  " : "FAIL", rel)
    pass
end

function main()
    println("pass-2 smoke: $(NSTEP) steps on an 8^3 grid, CPU\n")
    ok = Bool[]

    # --- channel family: Dirichlet walls + psolver_transform + steady force ---
    ch = channel_setup()
    chp = psolver_transform(ch)
    ch_plain = smoke("channel, steady force", ch,
        rf_bodyforce_navierstokes!, rf_steady_force_cache(ch, channel_bodyforce); psolver = chp)
    ch_smag = smoke("channel, Smagorinsky + force", ch,
        rf_eddyvisc_navierstokes!,
        rf_eddyvisc_force_cache(ch; model = Smagorinsky(T(0.071)), bodyforce = channel_bodyforce);
        psolver = chp)
    ch_wale = smoke("channel, WALE + force", ch,
        rf_eddyvisc_navierstokes!,
        rf_eddyvisc_force_cache(ch; model = WALE(T(0.53)), bodyforce = channel_bodyforce);
        psolver = chp)
    append!(ok, .!isnothing.((ch_plain, ch_smag, ch_wale)))

    # --- taylor-green / periodic family ---
    pe = periodic_setup()
    pep = psolver_spectral(pe)
    pe_plain = smoke("periodic, no closure", pe,
        IncompressibleNavierStokes.navierstokes!, nothing; psolver = pep)
    pe_smag = smoke("periodic, Smagorinsky", pe,
        rf_eddyvisc_navierstokes!, rf_eddyvisc_force_cache(pe; model = Smagorinsky(T(0.17)));
        psolver = pep)
    pe_wale = smoke("periodic, WALE", pe,
        rf_eddyvisc_navierstokes!, rf_eddyvisc_force_cache(pe; model = WALE(T(0.53)));
        psolver = pep)
    pe_ou = smoke("periodic, OU forcing", pe,
        ou_navierstokes!,
        ou_force_cache(pe; T_L = 0.01, e_star = 0.1, k_f = sqrt(2), rng_seed = 333, freeze = 1);
        psolver = pep)
    append!(ok, .!isnothing.((pe_plain, pe_smag, pe_wale, pe_ou)))

    # --- the positive controls: each closure must actually change the answer ---
    println("\ndiscriminating checks (a closure that changes nothing is not being applied):")
    push!(ok, differs("channel Smagorinsky vs plain", ch_smag, ch_plain))
    push!(ok, differs("channel WALE vs plain", ch_wale, ch_plain))
    push!(ok, differs("channel WALE vs Smagorinsky", ch_wale, ch_smag))
    push!(ok, differs("periodic Smagorinsky vs plain", pe_smag, pe_plain))
    push!(ok, differs("periodic WALE vs plain", pe_wale, pe_plain))
    push!(ok, differs("periodic WALE vs Smagorinsky", pe_wale, pe_smag))
    push!(ok, differs("periodic OU vs plain", pe_ou, pe_plain))

    n = count(ok)
    @printf("\n%d of %d checks passed\n", n, length(ok))
    n == length(ok)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main() ? 0 : 1)
end
