# V0 -- the harness runs at all -- and the normalization convention (TODO-0).

@testitem "V0 harness runs" default_imports = false begin
    using Test
    @test true
end

@testitem "V0 the ts_* layer loads without RikFlow" default_imports = false setup = [TSLayer] begin
    using Test
    # The point of the assertion is the absence of IncompressibleNavierStokes and CUDA from this
    # process, not the presence of the names. If this ever fails, the `ts_*` files have grown a
    # dependency and the cheap-CI property in test/Project.toml is gone.
    for f in (:_normalise, :fit_scaling, :build_history, :fit_ridge, :autocorr, :rollout)
        @test isdefined(TSLayer, f)
    end
    @test !haskey(Base.loaded_modules, Base.PkgId(Base.UUID("5e318141-6589-402b-868d-77d7df8c442e"),
                                                  "IncompressibleNavierStokes"))
end

@testitem "_normalise arithmetic is unchanged by the move from scale.jl" default_imports = false setup = [TSLayer] begin
    using Test
    using Statistics
    rng_x = Float32[1 2 3 4; 10 20 30 40]

    # :normal -- centred, divided by the uncorrected std, with ϵ added after the std.
    y, s = TSLayer._normalise(rng_x; normalization = :normal)
    mu = mean(rng_x, dims = 2)
    sig = std(rng_x, dims = 2, corrected = false) .+ Float32(1e-6)
    @test s.mu ≈ mu
    @test s.sigma ≈ sig
    @test y ≈ (rng_x .- mu) ./ sig

    # :standardise -- NOT centred. This is the whole reason TODO-0 exists.
    y2, s2 = TSLayer._normalise(rng_x; normalization = :standardise)
    @test all(iszero, s2.mu)
    @test s2.sigma ≈ sig
    @test y2 ≈ rng_x ./ sig

    # :minmax and :Id
    _, s3 = TSLayer._normalise(rng_x; normalization = :minmax)
    @test s3.mu ≈ (minimum(rng_x, dims = 2) .+ maximum(rng_x, dims = 2)) ./ 2
    _, s4 = TSLayer._normalise(rng_x; normalization = :Id)
    @test iszero(s4.mu) && isone(s4.sigma)

    # ϵ is converted to eltype(x) before use, so the Float32 path stays Float32.
    @test eltype(s.sigma) === Float32
    @test eltype(y) === Float32
end

@testitem "scale_input and scale_output round-trip" default_imports = false setup = [TSLayer] begin
    using Test
    x = Float32[1 2 3; 4 5 6]
    _, s = TSLayer.fit_scaling(x)
    @test TSLayer.scale_output(TSLayer.scale_input(x, s), s) ≈ x
    # A Scaling is a drop-in replacement for the legacy (; mu, sigma) named tuple.
    _, nt = TSLayer._normalise(x)
    @test TSLayer.scale_input(x, s) ≈ TSLayer.scale_input(x, nt)
end

@testitem "the convention travels with the fit" default_imports = false setup = [TSLayer] begin
    using Test
    x = Float32[1 2 3 4; 10 20 30 40]

    # paper 4's harmonized convention
    _, s = TSLayer.fit_scaling(x)
    @test s.mode === :normal
    @test s.penalize_intercept == false
    @test s.stats_from === :q
    @test s.shared_in_out
    @test s.version == TSLayer.SCALING_VERSION
    @test TSLayer.is_centred(s)
    @test TSLayer.assert_convention(s; mode = :normal, penalize_intercept = false) === s

    # the paper-2-faithful path, reachable as keyword arguments because G1 needs it
    _, sf = TSLayer.fit_scaling(x; normalization = :standardise, penalize_intercept = true)
    @test sf.mode === :standardise
    @test sf.penalize_intercept
    @test !TSLayer.is_centred(sf)

    # a mismatch throws and names what it found
    @test_throws ErrorException TSLayer.assert_convention(sf; mode = :normal)
    @test_throws ErrorException TSLayer.assert_convention(s; penalize_intercept = true)
    @test_throws ErrorException TSLayer.fit_scaling(x; normalization = :nonsense)

    # a legacy named tuple carries no convention, and saying so is the point
    _, nt = TSLayer._normalise(x)
    @test_throws ErrorException TSLayer.assert_convention(nt; mode = :normal)

    # adopting one requires the convention from outside, and an inconsistent claim is caught:
    # `:standardise` implies mu == 0 and this tuple was built with `:normal`.
    @test_throws ErrorException TSLayer.as_scaling(nt; normalization = :standardise)
    adopted = TSLayer.as_scaling(nt; normalization = :normal)
    @test adopted.penalize_intercept   # the archive's default, not paper 4's
    @test adopted.mu ≈ nt.mu
end

@testitem "scaling_pair reproduces what the deployed model reads" default_imports = false setup = [TSLayer] begin
    using Test
    x = Float32[1 2 3; 4 5 6]
    _, s = TSLayer.fit_scaling(x)
    p = TSLayer.scaling_pair(s)
    # `time_series_methods.jl:75-89` reaches for exactly these two names.
    @test p.in_scaling === s
    @test p.out_scaling === s
    split = TSLayer.Scaling(:normal, s.mu, s.sigma, s.eps, :q, false, false,
                            TSLayer.SCALING_VERSION)
    @test_throws ErrorException TSLayer.scaling_pair(split)
end

@testitem "a centred convention with a shared scaling puts the intercept near zero" default_imports = false setup = [TSLayer] begin
    using Test
    using Random, LinearAlgebra, Statistics
    # The free check that the convention change took effect: under `:normal` with one shared
    # scaling the target is centred too, so an unpenalized intercept fitted in scaled space has
    # essentially no level left to carry.
    #
    # 🔴 What it is NOT, measured here rather than assumed. Both `plan.md` §8a's prose and the
    # handoff say that under `:standardise` "the intercept has to carry the signal level". It does
    # not. On this fixture the scaled target mean is [170, 5170, 28500] while the fitted intercept
    # is at most 65, and the **total block sum is 1.000, 0.998, 0.995 under both conventions** --
    # identical to four digits. The level travels through the coefficient block, because at leading
    # order every lag in the window is ≈ q^{n-1} and the block sum is ≈ I whether or not the data
    # were centred. So metric #24's split is convention-dependent only in the intercept's share of
    # a level the coefficients already carry.
    #
    # This does not weaken TODO-0: the reason to harmonize is the Gram spectrum (#23), where an
    # uncentred design adds a leading eigenvalue of order N·(mean/σ)², and that is independent of
    # where the level ends up. It does mean the "intercept carries the level" sentence in both
    # documents should be corrected.
    rng = Random.MersenneTwister(20260907)
    nq, T = 3, 4000
    level = [50.0, 500.0, 5000.0]                 # QoI bands differ by orders of magnitude
    q = level .+ cumsum(randn(rng, nq, T + 1) .* 0.5; dims = 2) .* 0.02
    q_star = q[:, 1:T] .+ randn(rng, nq, T) .* 0.01

    spec = TSLayer.HistorySpec(; h = 2, n_qoi = nq)
    function fitted(mode)
        _, s = TSLayer.fit_scaling(q[:, 1:T]; normalization = mode)
        X, Y, _ = TSLayer.build_history(spec, TSLayer.scale_input(q_star, s),
                                        TSLayer.scale_input(q, s))
        C = TSLayer.fit_ridge(X, Y; lambda = 0.0)
        return (; b = C[TSLayer.bias_column(spec), :],           # the intercept
                S = vec(sum(C[1:(end - 1), :]; dims = 1)),       # total block sum, metric #24
                targetmean = vec(mean(Y; dims = 1)),
                targetrms = sqrt(mean(Y .^ 2)))
    end

    nrm = fitted(:normal)
    std_ = fitted(:standardise)

    # The scaled target has unit rms under `:normal`, so this is a relative statement too.
    @test nrm.targetrms ≈ 1 atol = 0.05
    @test maximum(abs, nrm.b) < 1e-2
    # The intercept tracks the scaled target mean, which centring has driven to ~0.
    @test maximum(abs, nrm.b) < 10 * maximum(abs, nrm.targetmean) + 1e-6
    # Under `:standardise` it is three orders of magnitude larger in absolute terms.
    @test maximum(abs, std_.b) > 1e3 * maximum(abs, nrm.b)

    # Metric #24: the block sum is ≈ I and is the same under both conventions.
    @test all(abs.(nrm.S .- 1) .< 1e-2)
    @test nrm.S ≈ std_.S rtol = 1e-3
    # ... while the intercept's share of the level is not: large mean, small intercept.
    @test maximum(abs, std_.b) < 0.5 * maximum(abs, std_.targetmean)
end
