# V18 and the rest of the regime-0 mechanism diagnostics.
#
# The companion's block assignment is an off-by-one waiting to happen -- three derivations of it
# disagreed -- so it is built here from **known** blocks planted in a synthetic `C` and read back,
# with the layout taken from `ts_history.jl`'s column helpers rather than from any document.

@testsnippet SpectrumFixture begin
    using Random, LinearAlgebra

    """
        plant(spec, A_star, A, B, b)

    A coefficient matrix `C` with the given blocks written into the column ranges the history spec
    defines. `C` is `nfeatures x N_Q` and a prediction is the row `x' * C`, so each block goes in
    transposed -- which is exactly the transposition `coefficient_blocks` has to undo.
    """
    function plant(spec, A_star, A, B, b)
        C = zeros(Float64, TSLayer.nfeatures(spec), spec.n_qoi)
        cs = TSLayer.qstar_columns(spec)
        cs === nothing || (C[cs, :] .= transpose(A_star))
        for k in 1:spec.h
            C[TSLayer.qlag_columns(spec, k), :] .= transpose(A[k])
            C[TSLayer.qstarlag_columns(spec, k), :] .= transpose(B[k])
        end
        C[TSLayer.bias_column(spec), :] .= b
        return C
    end

    "A design matrix with a prescribed singular spectrum, bias column last."
    function design_with_spectrum(n, svals; seed = 1)
        rng = Random.MersenneTwister(seed)
        m = length(svals)
        U = Matrix(qr(randn(rng, n, m)).Q)
        V = Matrix(qr(randn(rng, m, m)).Q)
        return U * Diagonal(svals) * V'
    end
end

@testitem "the column helpers partition the row exactly once" default_imports = false setup = [TSLayer, SpectrumFixture] begin
    using Test
    for h in (1, 2, 5), nq in (1, 3, 6), ip in (true, false)
        spec = TSLayer.HistorySpec(; h, n_qoi = nq, hist_var = :q_star_q, include_predictor = ip)
        m = TSLayer.nfeatures(spec)
        seen = zeros(Int, m)
        cs = TSLayer.qstar_columns(spec)
        cs === nothing || (seen[cs] .+= 1)
        for k in 1:h
            seen[TSLayer.qlag_columns(spec, k)] .+= 1
            seen[TSLayer.qstarlag_columns(spec, k)] .+= 1
        end
        seen[TSLayer.bias_column(spec)] += 1
        @test all(==(1), seen)          # every column claimed exactly once: no gap, no overlap
        @test m == nq * (2h + 1) + 1 - (ip ? 0 : nq)
    end
    # The starred-lag range sits immediately after that lag's corrected range, which is what makes
    # the pairing same-index.
    spec = TSLayer.HistorySpec(; h = 3, n_qoi = 2)
    for k in 1:3
        @test first(TSLayer.qstarlag_columns(spec, k)) ==
              last(TSLayer.qlag_columns(spec, k)) + 1
    end
    # Out-of-range lags and the streams that carry no such block return nothing.
    @test TSLayer.qstarlag_columns(spec, 4) === nothing
    @test TSLayer.qstarlag_columns(TSLayer.HistorySpec(; h = 2, n_qoi = 2, hist_var = :q), 1) ===
          nothing
    @test TSLayer.qlag_columns(TSLayer.HistorySpec(; h = 2, n_qoi = 2, hist_var = :q_star), 1) ===
          nothing
end

@testitem "V18 coefficient blocks round-trip through a planted C" default_imports = false setup = [TSLayer, SpectrumFixture] begin
    using Test, Random
    rng = Random.MersenneTwister(20260907)
    nq, h = 3, 4
    spec = TSLayer.HistorySpec(; h, n_qoi = nq)
    A_star = randn(rng, nq, nq)
    A = [randn(rng, nq, nq) for _ in 1:h]
    B = [randn(rng, nq, nq) for _ in 1:h]
    b = randn(rng, nq)
    C = plant(spec, A_star, A, B, b)

    bl = TSLayer.coefficient_blocks(C, spec)
    @test bl.A_star ≈ A_star
    @test all(bl.A[k] ≈ A[k] for k in 1:h)
    @test all(bl.B[k] ≈ B[k] for k in 1:h)
    @test bl.b ≈ b
end

@testitem "V18 total block sum equals the constant-input response" default_imports = false setup = [TSLayer, SpectrumFixture] begin
    using Test, Random, LinearAlgebra
    rng = Random.MersenneTwister(5)
    nq, h = 3, 4
    spec = TSLayer.HistorySpec(; h, n_qoi = nq)
    A_star = randn(rng, nq, nq)
    A = [randn(rng, nq, nq) for _ in 1:h]
    B = [randn(rng, nq, nq) for _ in 1:h]
    b = randn(rng, nq)
    C = plant(spec, A_star, A, B, b)

    tbs = TSLayer.total_block_sum(C, spec)
    @test tbs.S ≈ A_star + sum(A) + sum(B)
    @test tbs.b ≈ b

    # The operational meaning: feed a constant unit input through the actual mean map with the bias
    # zeroed, and the answer is S applied to that constant.
    C0 = copy(C)
    C0[TSLayer.bias_column(spec), :] .= 0
    x = ones(TSLayer.nfeatures(spec))
    x[TSLayer.bias_column(spec)] = 0
    @test vec(transpose(x) * C0) ≈ tbs.S * ones(nq)

    # And with the bias restored the response is S*1 + b, which is the split metric #24 reports.
    @test vec(transpose(ones(TSLayer.nfeatures(spec))) * C) ≈ tbs.S * ones(nq) .+ b

    # A mean map that is exactly a one-step copy has S = I and zero deviation.
    Aid = [k == 1 ? Matrix{Float64}(I, nq, nq) : zeros(nq, nq) for k in 1:h]
    Cid = plant(spec, zeros(nq, nq), Aid, [zeros(nq, nq) for _ in 1:h], zeros(nq))
    @test TSLayer.total_block_sum(Cid, spec).dev_from_identity < 1e-14
end

@testitem "V18 companion block assignment and rho" default_imports = false setup = [TSLayer, SpectrumFixture] begin
    using Test, Random, LinearAlgebra
    rng = Random.MersenneTwister(9)
    nq, h = 2, 3
    spec = TSLayer.HistorySpec(; h, n_qoi = nq)
    A_star = randn(rng, nq, nq)
    A = [randn(rng, nq, nq) for _ in 1:h]
    B = [randn(rng, nq, nq) for _ in 1:h]
    C = plant(spec, A_star, A, B, zeros(nq))

    cp = TSLayer.companion(C, spec)
    L = cp.L
    @test size(L) == (h * nq, h * nq)
    @test cp.size == h * nq
    @test cp.closures == (:qstar_is_previous_q, :same_index_pairing)

    # 🔑 The first block row is {A_1 + A_star + B_1, A_2 + B_2, ..., A_h + B_h}. That is the
    # same-index form; the one-step-older alternative would have needed (h+1)*nq states and a
    # different assignment, and the code settled which applies (test_history.jl, SC-48).
    @test L[1:nq, 1:nq] ≈ A[1] + A_star + B[1]
    for k in 2:h
        @test L[1:nq, ((k - 1) * nq + 1):(k * nq)] ≈ A[k] + B[k]
    end
    # The shift rows are exactly identities and nothing else.
    for k in 2:h
        blk = L[((k - 1) * nq + 1):(k * nq), :]
        @test blk[:, ((k - 2) * nq + 1):((k - 1) * nq)] ≈ I(nq)
        @test sum(abs, blk) ≈ nq            # only that identity is nonzero
    end

    # rho is the modulus of the largest eigenvalue of that matrix, and nothing else.
    @test TSLayer.rho(C, spec) ≈ maximum(abs, eigvals(L))

    # A scalar sanity case with an analytic answer: q_n = a q_{n-1}, so rho = |a|.
    for a in (0.5, 0.95, 1.0, 1.3)
        s1 = TSLayer.HistorySpec(; h = 1, n_qoi = 1)
        C1 = plant(s1, zeros(1, 1), [fill(a, 1, 1)], [zeros(1, 1)], zeros(1))
        @test TSLayer.rho(C1, s1) ≈ a
    end
    # The closure folds the predictor into lag 1, so A_star alone also produces growth: with
    # q^{n*} ≈ q^{n-1}, a model that only reads the predictor is still an autonomous recursion.
    s1 = TSLayer.HistorySpec(; h = 1, n_qoi = 1)
    @test TSLayer.rho(plant(s1, fill(0.8, 1, 1), [zeros(1, 1)], [zeros(1, 1)], zeros(1)), s1) ≈ 0.8

    # Only :q_star_q is derived; anything else must refuse rather than guess.
    @test_throws ErrorException TSLayer.companion(C, TSLayer.HistorySpec(; h, n_qoi = nq,
                                                                        hist_var = :q_star))
end

@testitem "V18 starred gain against analytic transfer functions" default_imports = false setup = [TSLayer, SpectrumFixture] begin
    using Test, LinearAlgebra
    nq = 2
    # With no q-lag feedback the transfer function is the static starred block itself.
    s1 = TSLayer.HistorySpec(; h = 1, n_qoi = nq)
    As = [2.0 0.0; 0.0 0.5]
    C = plant(s1, As, [zeros(nq, nq)], [zeros(nq, nq)], zeros(nq))
    g = TSLayer.starred_gain(C, s1)
    @test g.l2 ≈ opnorm(As)
    @test g.hinf ≈ opnorm(As) rtol = 1e-12

    # With q-lag feedback a*I the DC gain is opnorm(A_star)/(1-a), attained at omega = 0.
    for a in (0.0, 0.5, 0.9)
        Cf = plant(s1, As, [a * Matrix{Float64}(I, nq, nq)], [zeros(nq, nq)], zeros(nq))
        gf = TSLayer.starred_gain(Cf, s1)
        @test gf.hinf ≈ opnorm(As) / (1 - a) rtol = 1e-8
        @test gf.hinf_at ≈ 0.0 atol = 1e-9
        @test gf.l2 ≈ opnorm(As)          # the static block is unchanged by the feedback
    end
    # The starred *lag* block contributes to the numerator at nonzero frequency.
    Cb = plant(s1, zeros(nq, nq), [zeros(nq, nq)], [As], zeros(nq))
    @test TSLayer.starred_gain(Cb, s1).hinf ≈ opnorm(As) rtol = 1e-8
    @test TSLayer.starred_gain(Cb, s1).l2 ≈ 0.0
end

@testitem "V23 the Gram spectrum classifies what lambda is doing" default_imports = false setup = [TSLayer, SpectrumFixture] begin
    using Test, LinearAlgebra
    # A prescribed spectrum, so every returned number has a known answer.
    # No singular value sits on the alpha = 0.5 boundary at lambda = 1: a value of exactly 1.0
    # would give alpha = 0.5 to within an SVD rounding error, and `n_erased`'s strict comparison
    # would then flip on the last bit.
    svals = [100.0, 30.0, 10.0, 3.0, 0.3, 0.1, 0.01]
    X = design_with_spectrum(500, svals)
    g0 = TSLayer.gram_diagnostics(X; lambda = 0.0)
    @test g0.sigma ≈ sort(svals; rev = true)
    @test g0.sigma_min2 ≈ minimum(svals)^2
    @test g0.sigma_max2 ≈ maximum(svals)^2
    @test g0.cond ≈ maximum(svals) / minimum(svals)
    @test g0.rank == length(svals)
    @test all(g0.alpha .≈ 1)                       # lambda = 0 touches nothing
    @test g0.effective_dof ≈ length(svals)
    @test g0.branch === :inactive

    # alpha_j = s_j^2/(s_j^2 + lambda), exactly.
    lam = 1.0
    g1 = TSLayer.gram_diagnostics(X; lambda = lam)
    @test g1.alpha ≈ [s^2 / (s^2 + lam) for s in sort(svals; rev = true)]
    @test g1.effective_dof ≈ sum(g1.alpha)
    # lambda = 1 sits inside the small end. Counted from the formula rather than by hand.
    a_want = [s^2 / (s^2 + lam) for s in sort(svals; rev = true)]
    @test g1.n_erased == count(<(0.5), a_want)
    @test g1.n_untouched == count(>(0.99), a_want)
    @test g1.n_erased == 3 && g1.n_untouched == 3      # 0.3, 0.1, 0.01 erased; 100, 30, 10 free
    @test g1.branch === :rank_deficiency_fix

    # A lambda above the median squared singular value reaches well-determined directions.
    g2 = TSLayer.gram_diagnostics(X; lambda = 1e5)
    @test g2.branch === :shrinks_determined_directions
    @test g2.effective_dof < 2
    # And a lambda below the whole spectrum does essentially nothing.
    g3 = TSLayer.gram_diagnostics(X; lambda = 1e-8)
    @test g3.branch === :inactive
    # Not exactly `length(svals)`: at the smallest singular value 0.01 the squared value is 1e-4,
    # so alpha = 1e-4/(1e-4 + 1e-8) = 0.9999 and the effective degrees of freedom fall short by
    # 1e-4. Asserted against the formula so the shortfall is stated rather than tolerated.
    @test g3.effective_dof ≈ sum(s^2 / (s^2 + 1e-8) for s in svals) rtol = 1e-12
    @test length(svals) - g3.effective_dof ≈ 1e-4 rtol = 0.02

    # The three branches are exhaustive and mutually exclusive on this spectrum.
    @test length(unique((g1.branch, g2.branch, g3.branch))) == 3
end

@testitem "V25 the rank deficit is the detector, not the pinv gap" default_imports = false setup = [TSLayer, SpectrumFixture] begin
    using Test, LinearAlgebra, Random
    rng = Random.MersenneTwister(3)
    X = design_with_spectrum(300, [10.0, 5.0, 1.0])
    Y = randn(rng, 300, 2)

    ok = TSLayer.pinv_gap(X, Y)
    @test ok.rank == 3
    @test ok.rank_deficit == 0
    @test ok.gap < 1e-9

    # 🔴 On an **exactly** rank-deficient design the backslash and the pseudoinverse return the
    # same minimum-norm solution, so the gap `plan.md` §0 item 11 specifies stays at machine
    # precision **precisely when** the truncation it was meant to reveal has happened. Asserted in
    # four independent ways so the specification cannot be reinstated by accident.
    for A in (hcat(X, X[:, 1]),                        # appended duplicate
              hcat(X[:, 1], X, X[:, 1]),               # inserted duplicate
              hcat(X, zeros(300)),                     # a dead column
              hcat(X, X[:, 1] + 2X[:, 2]))             # an exact linear combination
        g = TSLayer.pinv_gap(A, Y)
        @test g.rank == 3
        @test g.rank_deficit == size(A, 2) - 3
        @test g.rank_deficit > 0                       # <- the number that actually detects it
        @test g.gap < 1e-9                             # <- the number that does not
        @test norm(A \ Y) ≈ norm(pinv(A) * Y) rtol = 1e-9
    end

    # Where the gap does have power is the **near**-deficient design, which is the regime the real
    # one sits in: a column duplicated to within 1e-12 passes the rank test at full rank while the
    # two solutions differ by a relative 1e-4.
    Xn = hcat(X, X[:, 1] .+ 1e-12 .* randn(rng, 300))
    n = TSLayer.pinv_gap(Xn, Y)
    @test n.rank == 4                                  # the rank test sees nothing
    @test n.rank_deficit == 0
    @test n.cond > 1e10
    @test n.rel > 1e-6                                 # the relative gap does
    @test TSLayer.gram_diagnostics(Xn).cond > 1e10
end

@testitem "V18 the conventions agree on slopes in Float64 and not in Float32" default_imports = false setup = [TSLayer, TrackedData, SpectrumFixture] begin
    using Test, LinearAlgebra, Statistics
    # 🔴 The invariance that separates arithmetic from modelling.
    #
    # At `lambda = 0` the `:normal` and `:standardise` conventions differ only by subtracting a
    # constant from every design column and from the target -- the two share the same `sigma` and
    # differ only in `mu`. A least-squares fit that carries an intercept is invariant to exactly
    # that, so the slope blocks must agree and only the intercept may move.
    #
    # On the real design they agree to 4.7e-11 in Float64 and differ by a **relative 3.3** in
    # Float32, because `cond(X) * eps(Float32)` is about 0.2. The consequences are not cosmetic:
    # `rho(Ctilde)` reads 1.012 in Float32 and 2.174 in Float64, which is the difference between
    # "marginally unstable" and "violently unstable" from the same data.
    #
    # So this test does two things. It checks the invariance, which validates
    # `coefficient_blocks`, `companion` and `starred_gain` against a property they must satisfy.
    # And it asserts the Float32 **failure**, so that no future reader takes a coefficient-level
    # diagnostic from a single-precision fit.
    rec = TrackedData.hit10()
    if rec === nothing
        @test_skip false
    else
        spec = TSLayer.HistorySpec(; h = 5, n_qoi = size(rec.q, 1))
        function fitc(mode, ::Type{T}) where {T}
            q = T.(rec.q)
            qs = T.(rec.q_star)
            _, s = TSLayer.fit_scaling(q[:, 400:3999]; normalization = mode)
            X, Y, _ = TSLayer.build_history(spec, TSLayer.scale_input(qs[:, 400:3999], s),
                                            TSLayer.scale_input(q[:, 400:4000], s))
            return TSLayer.fit_ridge(X, Y; lambda = zero(T)), X
        end
        slopes(C) = C[1:(end - 1), :]

        C64n, X64 = fitc(:normal, Float64)
        C64s, _ = fitc(:standardise, Float64)
        C32n, _ = fitc(:normal, Float32)
        C32s, _ = fitc(:standardise, Float32)

        rel(a, b) = norm(a - b) / norm(b)
        r64 = rel(slopes(C64n), slopes(C64s))
        r32 = rel(slopes(C32n), slopes(C32s))
        kappa = cond(X64)
        @test r64 < 1e-6                       # invariance holds where the arithmetic allows it
        @test r32 > 0.1                        # and fails outright in single precision
        @test r32 > 1e4 * r64
        @test kappa * eps(Float32) > 1e-2      # the reason, not a coincidence

        # The aggregate diagnostic averages the errors away; the coefficient-level ones do not.
        tbs64 = TSLayer.total_block_sum(C64n, spec).dev_from_identity
        tbs32 = TSLayer.total_block_sum(C32n, spec).dev_from_identity
        @test tbs32 ≈ tbs64 rtol = 0.2
        @test !isapprox(TSLayer.rho(C32n, spec), TSLayer.rho(C64n, spec); rtol = 0.2)

        # And the invariance is a property of the blocks, so it must show up in every derived
        # quantity computed in Float64, not only in the raw coefficients.
        @test TSLayer.rho(C64n, spec) ≈ TSLayer.rho(C64s, spec) rtol = 1e-6
        @test TSLayer.starred_gain(C64n, spec).hinf ≈ TSLayer.starred_gain(C64s, spec).hinf rtol = 1e-6
        @test TSLayer.total_block_sum(C64n, spec).S ≈ TSLayer.total_block_sum(C64s, spec).S rtol = 1e-6
    end
end
