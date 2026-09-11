# V30 -- the OU replay.
#
# 🔴 D6's highest-risk item. A forecast launched from `fields[k]` must start with the forcing chain
# the reference had at step `n_k`; launching from `Setup`'s zero state instead puts every member's
# forcing `n_k` steps out of phase with its own initial condition, which inflates skill without
# inflating spread and biases the spread-skill ratio downward -- toward a false "over-confident"
# verdict. The bias is silent and points in the direction that looks like a finding, so the replay
# is tested against the production forcing step rather than against a transcription of its formula.
#
# What is asserted here is everything that does not need a solver: the Markov property that licenses
# replay at all, determinism, `n = 0` being a provable no-op (so `ou_advance = 0` cannot shift
# anything archived), the state's shape and identity, and -- the one that matters -- that `n`
# replayed steps leave the state **bit-identical** to `n` real `OU_forcing_step!` calls.
#
# What is *not* asserted here is how many times a solve advances the chain. That is a property of
# `solve_unsteady`, not of the OU code, and it is measured by `analysis/ou_replay.jl` on a CPU
# mini-solve. The answer is `nstep + 1`, and the reasoning is in that file's header.
#
# The suite reaches the real `src/ouforcer.jl` without loading IncompressibleNavierStokes: the file
# is stdlib-only at parse time, and `OU_setup` touches its `setup` argument through four fields, all
# of which a plain named tuple can supply. So this stays inside `test/Project.toml`'s stdlib-only
# design while testing production code and not a copy of it.

@testmodule OU begin
    using Random

    # `code_base/src/ouforcer.jl` -- IncompressibleNavierStokes', not RikFlow's.
    const OUSRC = normpath(joinpath(@__DIR__, "..", "..", "..", "src", "ouforcer.jl"))
    include(OUSRC)

    """
        fake_setup(; N = 8, T = Float32)

    The four things `OU_setup` reads out of a `setup`: `Re` (for the element type), `ArrayType`,
    `dimension`, and `Nu[1][1]` (the number of points per direction, used to build the partial
    inverse-transform matrix `E`).

    Nothing else in `OU_setup` or `OU_forcing_step!` touches the flow, which is the point: the OU
    chain is a function of `(rng_seed, nsteps, Δt)` and of the forced wavenumbers only.

    ⚠️ Flat since the upstream merge. `dimension` and `Nu` used to live under `setup.grid`;
    upstream hoisted every grid field to the top level of the setup NamedTuple, so this stub
    follows. `Re` and `ArrayType` are RikFlow's own additions (`rf_setup`) and are unchanged.
    """
    fake_setup(; N::Int = 8, T = Float32) =
        (; Re = T(2000), ArrayType = Array,
         dimension = () -> 3, Nu = [ntuple(_ -> N, 3) for _ in 1:3])

    "HIT's forcing parameters, from `params_track.ou_bodyforce` on the 100 TU tracked record."
    const HIT_OU = (T_L = 0.01, e_star = 0.1, k_f = sqrt(2), freeze = 1, rng_seed = 333)

    "HIT's time step: `100.0f0 / 40000 === Float32(2.5e-3)`."
    const HIT_DT = Float32(2.5e-3)

    "A fresh chain with HIT's parameters."
    chain(; N::Int = 8, seed = HIT_OU.rng_seed) =
        OU_setup(; HIT_OU.T_L, HIT_OU.e_star, HIT_OU.k_f, HIT_OU.freeze,
                 rng_seed = seed, setup = fake_setup(; N))

    "The state after `n` replayed advances, as a plain copy."
    function advanced(n; N::Int = 8, seed = HIT_OU.rng_seed, Δt = HIT_DT)
        ou = chain(; N, seed)
        OU_advance!(; ou_setup = ou, Δt, n)
        return copy(ou.state)
    end

    "The state after `n` **full** forcing steps, as a plain copy."
    function forced(n; N::Int = 8, seed = HIT_OU.rng_seed, Δt = HIT_DT)
        ou = chain(; N, seed)
        for _ in 1:n
            OU_forcing_step!(; ou_setup = ou, Δt)
        end
        return copy(ou.state)
    end
end

@testitem "V30 the replay reproduces the real forcing step bit-for-bit" default_imports = false setup = [OU] begin
    using Test
    # 🔑 The decisive code-level check. `OU_advance!` skips the forcing field, which is the expensive
    # half; it may not skip anything that touches `rng`. If the two ever diverge -- a reordered
    # random draw, a changed formula in one copy -- every D6 member's forcing is wrong and nothing
    # else in this file would notice.
    for n in (1, 2, 7, 100, 523)
        @test OU.advanced(n) == OU.forced(n)
    end
end

@testitem "V30 the Markov property that licenses replay" default_imports = false setup = [OU] begin
    using Test
    # Advancing n then m must equal advancing n+m in one go. Without this, a chain could not be
    # resumed from its own state and `ou_advance` would be meaningless.
    for (n, m) in ((0, 13), (13, 0), (5, 5), (37, 63), (100, 1))
        ou = OU.chain()
        OU.OU_advance!(; ou_setup = ou, Δt = OU.HIT_DT, n = n)
        OU.OU_advance!(; ou_setup = ou, Δt = OU.HIT_DT, n = m)
        @test ou.state == OU.advanced(n + m)
    end
end

@testitem "V30 the state is a function of (rng_seed, n, Δt) and nothing else" default_imports = false setup = [OU] begin
    using Test

    # Deterministic: same arguments, bit-identical state.
    @test OU.advanced(250) == OU.advanced(250)

    # Different n, different state -- so the count is load-bearing and an off-by-one is detectable.
    @test OU.advanced(250) != OU.advanced(251)
    @test OU.advanced(250) != OU.advanced(249)

    # Different seed, different chain.
    @test OU.advanced(250) != OU.advanced(250; seed = 334)

    # Different Δt, different chain. The replay must use the reference's step size, which is why
    # `online_sgs` asserts `tsim / Δt` is integral before it replays.
    @test OU.advanced(250) != OU.advanced(250; Δt = Float32(2.5e-3) * 2)

    # 🔑 Grid-independent. `OU_setup` sizes `state` by the forced wavenumbers, which depend on `k_f`
    # alone, so an 8^3 chain and a 64^3 chain are the same chain. This is what makes the mini-solve
    # in `analysis/ou_replay.jl` decisive for the 64^3 production runs rather than merely suggestive.
    @test OU.advanced(250; N = 8) == OU.advanced(250; N = 16)
    @test OU.advanced(250; N = 8) == OU.advanced(250; N = 64)
end

@testitem "V30 n = 0 is a provable no-op, so ou_advance = 0 shifts nothing archived" default_imports = false setup = [OU] begin
    using Test
    ou = OU.chain()
    zero_state = copy(ou.state)
    @test all(iszero, zero_state)              # `OU_setup` starts every chain at exactly zero

    OU.OU_advance!(; ou_setup = ou, Δt = OU.HIT_DT, n = 0)
    @test ou.state == zero_state

    # And it consumes no randomness: advancing 0 then 5 equals advancing 5. That is what makes
    # `online_sgs(; ou_advance = 0)` -- the default, and every archived run -- provably unchanged.
    OU.OU_advance!(; ou_setup = ou, Δt = OU.HIT_DT, n = 5)
    @test ou.state == OU.advanced(5)

    @test_throws ErrorException OU.OU_advance!(; ou_setup = OU.chain(), Δt = OU.HIT_DT, n = -1)
end

@testitem "V30 the replay writes in place and changes no shape or type" default_imports = false setup = [OU] begin
    using Test
    ou = OU.chain()
    state = ou.state
    N_d, num_dims = size(state)

    @test eltype(state) == ComplexF32
    @test num_dims == 3
    @test N_d > 0

    OU.OU_advance!(; ou_setup = ou, Δt = OU.HIT_DT, n = 64)
    @test ou.state === state                    # same array object; nothing reallocated
    @test size(ou.state) == (N_d, num_dims)
    @test eltype(ou.state) == ComplexF32
    @test all(isfinite, ou.state)

    # The forcing field is deliberately *not* updated by a replay, and that is safe only because
    # `solve_unsteady` calls `OU_forcing_step!` followed by `OU_get_force!` before every use of the
    # body force (`solver.jl:61-63, 87-89, 102-104`). Recorded here so the assumption is visible.
    @test all(d -> all(iszero, ou.f_hat[d]), 1:num_dims)
    OU.OU_forcing_step!(; ou_setup = ou, Δt = OU.HIT_DT)
    @test any(d -> any(!iszero, ou.f_hat[d]), 1:num_dims)
end

@testitem "V30 a replayed chain is stationary at HIT's parameters" default_imports = false setup = [OU] begin
    using Test
    using Statistics
    # A cheap sanity floor on the physics rather than on the plumbing: the OU process has stationary
    # variance `Var * T_L / ...` reached after a few `T_L`. With T_L = 0.01 TU and Δt = 2.5e-3 TU,
    # 4000 steps is 10 TU, a thousand correlation times, so the state must be finite and neither
    # decaying to zero nor growing. A sign error in the drift or diffusion term shows up here.
    early = OU.advanced(400)
    late = OU.advanced(4000)
    later = OU.advanced(8000)
    m(x) = sqrt(sum(abs2, x) / length(x))
    @test all(isfinite, late)
    @test m(early) > 0
    @test 0.2 < m(later) / m(late) < 5.0        # loose: one draw of a stochastic state, not a mean
end
