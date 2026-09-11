# The OU replay, and the one check on it that needs the solver.
#
# 🔴 This is D6's highest-risk item. A forecast launched from `fields[k]` must start with the forcing
# chain the reference had at step `n_k`. `Setup` starts every chain at zero, so launching from
# `fields[k]` without replaying puts every member's forcing `n_k` steps out of phase with the field
# it was handed. That inflates skill without inflating spread and biases the spread-skill ratio
# **downward** -- toward a false "over-confident" verdict, which is the direction that looks like a
# real finding. The existing HIT drivers are correct only because they launch from `fields[1]`,
# where `n_1 = 0` and a zero state is the right state.
#
# The replay itself is `OU_advance!` in `src/ouforcer.jl`, and `online_sgs(; ou_advance = n_k)`
# applies it. Its arithmetic -- the Markov property, determinism, `n = 0` being a no-op, and
# equality with the real `OU_forcing_step!` -- is asserted in `lib/RikFlow/test/test_ou.jl`, which
# is stdlib-only and runs everywhere.
#
# What that suite cannot see is **how many times a solve advances the chain**, because that is a
# property of `solve_unsteady`, not of the OU code. This driver measures it on a CPU mini-solve.
# It is the decisive check: if the count is wrong the misphase is silent and every downstream number
# is untrustworthy.
#
# 🔑 The answer, and it is not the obvious one. `solve_unsteady` advances the chain **`nstep + 1`**
# times for `nstep` steps: once at `solver.jl:61-63`, before the loop, and once per iteration at
# `solver.jl:102-104`. So the forcing seen by step `it` is the state after `it + 1` advances. The
# reference's step `n_k + 1` -- the first step a forecast from `fields[k]` has to match -- therefore
# used the state after `n_k + 2` advances, and a forecast that pre-advances by `n_k` and then takes
# its own priming advance plus its own first iteration arrives at exactly `n_k + 2`. The two extra
# advances cancel **because both runs make them**, which is why `ou_advance = n_k` is right and why
# it would not be right if the priming call existed on only one side.
#
# Usage (the solver environment, not `analysis/`; this is the only analysis file that needs it):
#   julia --startup-file=no --project=. lib/RikFlow/analysis/ou_replay.jl

using IncompressibleNavierStokes
using Random
using Printf

"HIT's forcing parameters, from `params_track.ou_bodyforce` on the 100 TU tracked record."
const HIT_OU = (T_L = 0.01, e_star = 0.1, k_f = sqrt(2), freeze = 1, rng_seed = 333)

"HIT's time step. `100.0f0 / 40000 === Float32(2.5e-3)`, so the reference and a 1308-step forecast
step the chain identically -- which `online_sgs` asserts before it replays anything."
const HIT_DT = Float32(2.5e-3)

"""
    mini_setup(; n = 8, ou = HIT_OU, T = Float32)

A tiny periodic CPU setup carrying HIT's OU forcing. Small enough to solve in a second, and the OU
chain it builds is the same object the 64^3 runs build: `OU_setup` sizes `state` by the forced
wavenumbers, which depend on `k_f` alone, so an 8^3 grid and a 64^3 grid carry the *same* chain.
That is what makes a mini-solve decisive rather than merely suggestive.
"""
function mini_setup(; n::Int = 8, ou = HIT_OU, T = Float32)
    setup = rf_setup(;
        x = ntuple(α -> LinRange(T(0), T(1), n + 1), 3),
        Re = T(2000),
    )
    # Since the upstream merge the chain lives in the force cache, not in the setup, so this
    # returns both. Everything below reads `force_cache.ou_setup` where it used to read
    # `setup.ou_setup`.
    (setup, ou_force_cache(setup; ou...))
end

"""
    replay_state(nsteps; n = 8, ou = HIT_OU, Δt = HIT_DT)

The OU state after `nsteps` advances, computed without the solver. Returns a plain `Array` copy.

Offline counterpart of what `online_sgs(; ou_advance)` does in place: the state is a function of
`(rng_seed, nsteps, Δt)` and nothing else, so this reproduces any run's chain from its parameters.
"""
function replay_state(nsteps::Integer; n::Int = 8, ou = HIT_OU, Δt = HIT_DT)
    _, force_cache = mini_setup(; n, ou)
    OU_advance!(; force_cache.ou_setup, Δt, n = nsteps)
    return Array(force_cache.ou_setup.state)
end

"""
    solver_state(nsteps; n = 8, ou = HIT_OU, Δt = HIT_DT)

The OU state left behind by an actual `solve_unsteady` of `nsteps` steps. Returns a plain `Array`
copy.
"""
function solver_state(nsteps::Integer; n::Int = 8, ou = HIT_OU, Δt = HIT_DT)
    setup, force_cache = mini_setup(; n, ou)
    T = eltype(setup.x[1])
    tsim = T(Δt) * nsteps
    @assert round(Int, tsim / Δt) == nsteps "mini-solve step count is not $nsteps"
    ustart = IncompressibleNavierStokes.vectorfield(setup)
    psolver = psolver_spectral(setup)
    solve_unsteady(;
        # Upstream changed solve_unsteady's default method from RKMethods.RK44 to LMWray3 at the
        # merge; pinned so this keeps the pre-merge integrator.
        method = RKMethods.RK44(; T = eltype(ustart)),
        setup,
        start = (; u = ustart),
        force! = ou_navierstokes!,
        force_cache,
        params = rf_params(setup),
        tlims = (T(0), tsim),
        Δt = T(Δt),
        psolver,
    )
    return Array(force_cache.ou_setup.state)
end

"""
    check_advance_count(; nsteps = 25, probe = 4)

Measure how many `OU_advance!` steps reproduce a `solve_unsteady` of `nsteps` steps, by trying every
count in `0:nsteps+probe` and reporting which ones match bit-for-bit.

Reported, not assumed. The whole `ou_advance` design rests on this number, and reading it off the
source is how an off-by-one survives.
"""
function check_advance_count(; nsteps::Int = 25, probe::Int = 4, freeze::Int = 1)
    ou = (; HIT_OU..., freeze)
    @printf("mini-solve: %d steps on an 8^3 grid, Δt = %g, freeze = %d, rng_seed = %d\n",
            nsteps, HIT_DT, freeze, ou.rng_seed)
    ref = solver_state(nsteps; ou)
    @printf("  solver left state %s, |state| = %.6e\n", string(size(ref)), sqrt(sum(abs2, ref)))

    # 🔑 The replay has to be probed at the step size the solver actually uses, `Δt * freeze`
    # (the OU block in `solve_unsteady`), not `Δt`. With `freeze = 1` the two coincide, which is why a
    # freeze-blind replay looks correct on HIT and is wrong everywhere else.
    matches = Int[]
    for m in 0:(nsteps + probe)
        replay_state(m; ou, Δt = HIT_DT * freeze) == ref && push!(matches, m)
    end
    if freeze != 1
        blind = [m for m in 0:(nsteps + probe) if replay_state(m; ou, Δt = HIT_DT) == ref]
        @printf("  a freeze-blind replay (Δt instead of Δt*freeze) matches at: %s\n",
                isempty(blind) ? "nothing — so getting freeze wrong is detectable, not silent" :
                string(blind))
    end

    if isempty(matches)
        @printf("  🔴 NO advance count in 0:%d reproduces the solver's state.\n", nsteps + probe)
        return (; ok = false, matches, nsteps)
    end
    for m in matches
        @printf("  match at %d advances  (nstep %+d)\n", m, m - nsteps)
    end
    # What the source predicts: one priming call (before the loop) plus one per iteration whose
    # `stepper.n` is divisible by `freeze`.
    expected = count(m -> mod(m, freeze) == 0, 0:(nsteps - 1)) + 1
    @printf("  source predicts %d = 1 priming + %d in-loop\n", expected, expected - 1)
    ok = matches == [expected]
    if ok && freeze == 1
        println("  ✅ exactly one match, at nstep + 1, as `online_sgs`'s ou_advance assumes.")
        println("     A forecast pre-advanced by n_k then takes the same priming advance and the")
        println("     same first-iteration advance the reference did, so the two cancel.")
    elseif ok
        println("  ✅ matches the source's prediction. Note the count is NOT nstep + 1 here, and")
        println("     the step size is Δt*freeze — which is why `online_sgs` refuses to replay")
        println("     unless freeze == 1 rather than guessing the generalisation.")
    else
        println("  🔴 the match is not what the source predicts. `ou_advance = n_k` is then WRONG")
        println("     and the spread-skill ratio it feeds is biased. Fix before running anything.")
    end
    return (; ok, matches, nsteps, freeze, expected)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # `freeze = 1` is HIT's, and the only case `online_sgs(; ou_advance)` accepts. `freeze = 10` is
    # the DNS reference's, measured here so that the refusal in `online_sgs` rests on a measurement
    # rather than on caution.
    results = map((1, 10)) do freeze
        r = check_advance_count(; freeze)
        println()
        r
    end
    all(r -> r.ok, results) || exit(1)
end
