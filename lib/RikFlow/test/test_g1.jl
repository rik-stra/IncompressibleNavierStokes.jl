# G1 -- reproduce paper 2's archived HIT fits from the tracked record.
#
# `plan.md` §12: "offline refit of `c` against the archived `LinReg.jld2` to `rtol=1e-6` at λ=0 and
# `1e-3` at λ=0.01 (runs in CI, seconds -- the strongest available reproduction test)".
#
# This is the half of V1 that needs paper 2's archive rather than only its source. The archive is
# gitignored (`*output/`) and lives outside the repository, so the path comes from the environment
# and every item here skips when it is absent. Point `RIKFLOW_ARCHIVE` at the directory holding
# `inputs.jld2` and `TO_LRS/<name>/LinReg.jld2`.
#
# ⚠️ λ > 0 is not reproduced here. Those fits went through `RegularizedLeastSquares`' ADMM, which
# is not available in this stdlib-only environment; the QR-vs-ADMM parity check at λ = 0, 0.01 and
# 0.1 needs an environment carrying that package and belongs to P2's G1 driver. What runs here is
# the λ = 0 acceptance, which is the exact `\` path and therefore the tight one.

@testmodule Archive begin
    using JLD2

    # Two roots, because the archive is split and the two halves are not the same runs.
    #
    #  - `frozen` is paper 2's frozen archive (July 2025). It carries LinReg1, 63 and 64 with
    #    `data_online_tsim100.0_replica{1..5}` and the 100 TU HF reference.
    #  - `dev` is the working repository (February 2025). It carries all five online
    #    configurations -- LinReg1, 63, 64, 73, 74 -- but its online runs are the
    #    `_rand_initial_dQ` variant, so its *trajectories* are not interchangeable with the frozen
    #    ones. Its `LinReg.jld2` fits are, and that is asserted below.
    #
    # Layouts differ: the frozen archive keeps models under `TO_LRS/<name>/`, the working
    # repository under `output/online/<name>/`. Override the pair with RIKFLOW_ARCHIVE and
    # RIKFLOW_DEV_ARCHIVE.
    const FROZEN = raw"C:\Users\rik\Documents\julia_code\INS_paper_summer2025\lib\RikFlow\exp_square_HIT\output\paper_data_HIT"
    const DEV = raw"C:\Users\rik\Documents\julia_code\IncompressibleNavierStokes.jl\lib\RikFlow\exp_square_HIT\paper_runs"

    frozen() = get(ENV, "RIKFLOW_ARCHIVE", FROZEN)
    dev() = get(ENV, "RIKFLOW_DEV_ARCHIVE", DEV)

    model_dir(::Val{:frozen}, name) = joinpath(frozen(), "TO_LRS", name)
    model_dir(::Val{:dev}, name) = joinpath(dev(), "output", "online", name)
    inputs_file(::Val{:frozen}) = joinpath(frozen(), "inputs.jld2")
    inputs_file(::Val{:dev}) = joinpath(dev(), "inputs.jld2")

    has(which::Symbol, name) = isfile(joinpath(model_dir(Val(which), name), "LinReg.jld2"))
    available() = isfile(inputs_file(Val(:frozen))) || isfile(inputs_file(Val(:dev)))

    "Which root holds a named configuration's fit; the frozen archive wins when both do."
    function where_is(name::AbstractString)
        has(:frozen, name) && return :frozen
        has(:dev, name) && return :dev
        return nothing
    end

    "The full input entry for a named configuration -- `parameters.jld2` does not store `lambda`."
    function config(name::AbstractString)
        for which in (:frozen, :dev)
            f = inputs_file(Val(which))
            isfile(f) || continue
            inp = load(f, "inputs")
            i = findfirst(e -> e.name == name, inp)
            i === nothing || return inp[i]
        end
        error("no configuration named $name in any archive inputs.jld2")
    end

    """
        fit(name; which = where_is(name))

    The archived fit. `c` is stored transposed relative to the solve (the script saves `c'`), so it
    comes back as `N_Q × nfeatures` and is returned that way; the caller transposes.

    `stoch_distr` is a `Distributions.MvNormal` and this environment has neither Distributions nor
    PDMats, so JLD2 hands back a reconstructed object. Its fields are still reachable, which is all
    the residual covariance is needed for -- and keeping it that way is what preserves the
    stdlib-only property of the suite.
    """
    function fit(name::AbstractString; which = where_is(name))
        which === nothing && error("no archived fit for $name under either root")
        return jldopen(joinpath(model_dir(Val(which), name), "LinReg.jld2")) do io
            (; c = io["c"], scaling = io["scaling"], hist_var = io["hist_var"],
             hist_len = io["hist_len"], include_predictor = io["include_predictor"],
             fitted_qois = io["fitted_qois"], stoch_distr = io["stoch_distr"], root = which)
        end
    end

    """
        mvg(sd)

    Mean and covariance of an archived residual `MvNormal`, reconstructed or not.

    `getproperty`, not `getfield`: a `JLD2.ReconstructedMutable` stores everything in one `fields`
    tuple and exposes the original names through `getproperty` only.
    """
    function mvg(sd)
        mu = getproperty(sd, :μ)
        S = getproperty(sd, :Σ)
        Sig = S isa AbstractMatrix ? S : getproperty(S, :mat)
        return collect(mu), collect(Sig)
    end
end

@testitem "G1 reproduces the archived HIT coefficients at lambda = 0" default_imports = false setup = [TSLayer, TrackedData, Archive] begin
    using Test, LinearAlgebra, Statistics

    rec = TrackedData.hit10()
    if !Archive.available() || rec === nothing
        @warn "G1 skipped" archive = Archive.available() record = rec !== nothing
        @test_skip false
    else
        names = filter(n -> Archive.where_is(n) !== nothing, ("LinReg1", "LinReg64", "LinReg74"))
        @test !isempty(names)
        for name in names
            cfg = Archive.config(name)
            @test cfg.lambda == 0.0                 # the λ>0 configs need the ADMM environment
            arch = Archive.fit(name)

            # Reproduce the archived convention exactly: `:standardise`, statistics from `q` over
            # the training window only, one scaling for both directions, penalized intercept.
            a, b = cfg.train_range
            _, s = TSLayer.fit_scaling(rec.q[:, a:(b - 1)]; normalization = cfg.normalization,
                                       penalize_intercept = true)
            TSLayer.assert_convention(s; mode = :standardise, penalize_intercept = true)

            # The archive stores its own scaling; it must be the same object we just recomputed.
            @test s.sigma ≈ arch.scaling.in_scaling.sigma
            @test all(iszero, arch.scaling.in_scaling.mu)
            @test arch.scaling.in_scaling === arch.scaling.out_scaling ||
                  arch.scaling.in_scaling == arch.scaling.out_scaling

            spec = TSLayer.HistorySpec(; h = cfg.hist_len, n_qoi = size(rec.q, 1),
                                       hist_var = cfg.hist_var,
                                       include_predictor = cfg.include_predictor)
            @test TSLayer.nfeatures(spec) == size(arch.c, 2)

            X, Y, _ = TSLayer.build_history(spec, TSLayer.scale_input(rec.q_star[:, a:(b - 1)], s),
                                            TSLayer.scale_input(rec.q[:, a:b], s))
            @test size(X, 1) == 3600 - cfg.hist_len       # N = 3600 over the training window

            C = TSLayer.fit_ridge(X, Y; lambda = cfg.lambda, penalize_intercept = true)
            Carch = collect(arch.c')

            rel = norm(C - Carch) / norm(Carch)
            relpred = norm(X * C - X * Carch) / norm(X * Carch)
            @info "G1 $name" root = arch.root h = cfg.hist_len lambda = cfg.lambda rel_c = rel rel_pred = relpred

            # 🔴 `plan.md` §12 sets the λ=0 acceptance at `rtol = 1e-6` on `c`. That is not
            # attainable and the tolerance should be restated. Measured here: the coefficients
            # reproduce to ~6-8e-5 while the *predictions* reproduce to ~6-8e-7. Both are correct
            # and consistent -- the archive is Float32 (eps ≈ 1.2e-7) on a design where the lagged
            # `q` and `q*` streams differ only by the correction, so the coefficient vector is
            # poorly determined along the collinear directions while the fitted subspace is not.
            # The prediction-space number is the one that means "same model"; the coefficient-space
            # number mostly measures cond(X).
            @test rel < 1e-4
            @test relpred < 1e-6

            # The residual MVG follows from the same fit, so it is a second, independent check on
            # the same coefficients.
            _, Sig = Archive.mvg(arch.stoch_distr)
            Res = Y - X * C
            Smine = cov(Float64.(Res); corrected = false)
            @test norm(Smine - Sig) / norm(Sig) < 1e-3
        end
    end
end

@testitem "G1 the archive is five configurations, not a 2x2 grid" default_imports = false setup = [Archive] begin
    using Test
    if !Archive.available()
        @test_skip false
    else
        # 🔴 `plan.md` §7 says "Paper 2's four archived HIT configs (h ∈ {5,20} × λ ∈ {0,0.01})".
        # The archive says otherwise: five configurations were run online, at h ∈ {5, 10, 40}, and
        # h = 5 exists only at λ = 0. It is not a grid, so the anchor count and the "architecture
        # count is met on its own" claim both need restating.
        got = [(c.hist_len, c.lambda) for c in Archive.config.(("LinReg1", "LinReg63", "LinReg64",
                                                                "LinReg73", "LinReg74"))]
        @test got == [(5, 0.0), (10, 0.01), (10, 0.0), (40, 0.01), (40, 0.0)]
        @test !((5, 0.01) in got)
        # S1's baseline τ is "h=5, λ=0" and it is present.
        @test (5, 0.0) in got
        # Every archived configuration used the uncentred convention, so every quoted λ is in it.
        for nm in ("LinReg1", "LinReg63", "LinReg64", "LinReg73", "LinReg74")
            @test Archive.config(nm).normalization === :standardise
            @test Archive.config(nm).train_range == (400, 4000)
        end
        # The frozen archive holds three of the five; the working repository holds all five.
        @test count(n -> Archive.has(:frozen, n),
                    ("LinReg1", "LinReg63", "LinReg64", "LinReg73", "LinReg74")) == 3
    end
end

@testitem "G1 the archived residual model is samplable, not merely readable" default_imports = false setup = [Archive] begin
    using Test, Random, LinearAlgebra, Distributions
    # 🔴 This test exists because 1422 passing tests said nothing about it.
    #
    # `LinReg.jld2` stores its residual model as a serialised
    # `MvNormal{Float64, PDMat{Float64, Matrix{Float64}}}`. PDMats later gained a third type
    # parameter -- `PDMat{T,S,C}` -- **inside the 0.11 patch series**, so on a newer PDMats JLD2
    # cannot map the stored two-parameter type onto the installed three-parameter one and returns a
    # `JLD2.ReconstructedMutable` instead. Everything in this suite kept working, because
    # `Archive.mvg` reads `μ` and `Σ` through `getproperty` precisely to tolerate that. But the
    # **deployed** path calls `rand(rng, stoch_distr)` on every step
    # (`src/time_series_methods.jl:150`), and there is no `rand` method for a reconstruction --
    # so the D6 smoke test died there while the suite stayed green.
    #
    # The pins in `test/Project.toml` and `lib/RikFlow/Project.toml` are what keep this passing.
    # If it fails, the archived model is unloadable by the deployed sampler and no online run is
    # possible, whatever the rest of the suite says.
    if !Archive.available()
        @test_skip false
    else
        names = filter(n -> Archive.where_is(n) !== nothing, ("LinReg1", "LinReg64", "LinReg74"))
        @test !isempty(names)
        for name in names
            sd = Archive.fit(name).stoch_distr
            # The type resolved to the installed one, rather than being reconstructed.
            @test sd isa Distributions.MvNormal
            mu, Sig = Archive.mvg(sd)
            nq = length(mu)
            @test nq == 6
            @test size(Sig) == (nq, nq)
            @test issymmetric(Sig) || Sig ≈ Sig'
            @test isposdef(Sig)

            # What the deployed sampler actually does, and that it is reproducible.
            x = rand(Xoshiro(1), sd)
            @test length(x) == nq
            @test all(isfinite, x)
            @test x == rand(Xoshiro(1), sd)
            @test x != rand(Xoshiro(2), sd)
            # The draw is on the residual's own scale, not wildly off it.
            @test all(abs.(x .- mu) .<= 12 .* sqrt.(diag(Sig)))
        end
    end
end

@testitem "G1 the frozen and working fits agree where both exist" default_imports = false setup = [Archive] begin
    using Test, LinearAlgebra
    # The two roots are different *runs* -- the working repository's online trajectories are the
    # `_rand_initial_dQ` variant -- but they should be the same *fits*. If they are not, "the
    # archived M0" is ambiguous and every reproduction number has to name its root.
    both = filter(n -> Archive.has(:frozen, n) && Archive.has(:dev, n),
                  ("LinReg1", "LinReg63", "LinReg64"))
    if isempty(both)
        @test_skip false
    else
        for nm in both
            a = Archive.fit(nm; which = :frozen)
            b = Archive.fit(nm; which = :dev)
            @test a.c == b.c
            @test a.scaling.in_scaling.sigma == b.scaling.in_scaling.sigma
            @test a.hist_len == b.hist_len && a.hist_var == b.hist_var
        end
    end
end
