# V16 -- ridge parity -- and the intercept-penalty half of TODO-0.
#
# 🔴 The trap these tests exist for: `X'X` squares the condition number, and on this design
# `q^{n-k}` and `q^{n-k*}` differ only by the correction, so the collinearity is severe. The
# normal-equation route was written twice in one session (`fit_ridge`, then `solve_C`) and gave a
# relative error of 5.8 in the coefficients against paper 2's archive. `fit_ridge` now solves
# `[X; √λ·P] \ [Y; 0]` by QR, and these tests are what keep it that way.

@testsnippet FitFixture begin
    using Random, LinearAlgebra

    """
        collinear(; n, m, nq, seed)

    A design with the pathology of the real one: pairs of near-duplicate columns, standing in for
    the lagged `q` and `q_star` streams that differ only by the correction.
    """
    function collinear(; n = 400, m = 21, nq = 3, seed = 20260907)
        rng = Random.MersenneTwister(seed)
        base = randn(rng, n, (m - 1) ÷ 2)
        X = zeros(n, m)
        for j in axes(base, 2)
            X[:, 2j - 1] = base[:, j]
            X[:, 2j] = base[:, j] .+ 1e-7 .* randn(rng, n)   # the near-duplicate partner
        end
        X[:, m] .= 1.0                                        # bias column, last
        Y = X * randn(rng, m, nq) .+ 0.01 .* randn(rng, n, nq)
        return X, Y
    end
end

@testitem "V16 lambda = 0 is the plain least-squares solve" default_imports = false setup = [TSLayer, FitFixture] begin
    using Test
    X, Y = collinear()
    @test TSLayer.fit_ridge(X, Y; lambda = 0.0) == X \ Y
end

@testitem "V16 QR beats the normal equations on this design" default_imports = false setup = [TSLayer, FitFixture] begin
    using Test
    using LinearAlgebra
    X, Y = collinear()
    lambda = 1e-4
    m = size(X, 2)

    P = Matrix{Float64}(I, m, m) .* sqrt(lambda)
    P[m, m] = 0.0                                       # intercept out of the penalty
    qr_sol = TSLayer.fit_ridge(X, Y; lambda, penalize_intercept = false)
    @test qr_sol ≈ [X; P] \ [Y; zeros(m, size(Y, 2))]

    # The reference solution, computed in higher precision through the same augmented system.
    ref = [big.(X); big.(P)] \ [big.(Y); zeros(BigFloat, m, size(Y, 2))]
    ne = (X' * X + lambda * (P' * P)) \ (X' * Y)        # what the bug did
    err_qr = norm(qr_sol .- Float64.(ref)) / norm(Float64.(ref))
    err_ne = norm(ne .- Float64.(ref)) / norm(Float64.(ref))
    @test err_qr < 1e-8
    @test err_ne > 100 * err_qr                          # squaring cond(X) is not a detail
end

@testitem "V16 the intercept is excluded from the penalty when asked" default_imports = false setup = [TSLayer, FitFixture] begin
    using Test
    using LinearAlgebra, Statistics
    X, Y = collinear()
    m = size(X, 2)
    lambda = 1.0                                         # large enough to see the difference

    free = TSLayer.fit_ridge(X, Y; lambda, penalize_intercept = false)
    pen = TSLayer.fit_ridge(X, Y; lambda, penalize_intercept = true)
    @test !(free ≈ pen)

    # An unpenalized intercept is the one that reproduces the augmented system with a zeroed
    # penalty row, and a penalized one the full identity.
    Pf = Matrix{Float64}(I, m, m) .* sqrt(lambda); Pf[m, m] = 0.0
    Pp = Matrix{Float64}(I, m, m) .* sqrt(lambda)
    Z = zeros(m, size(Y, 2))
    @test free ≈ [X; Pf] \ [Y; Z]
    @test pen ≈ [X; Pp] \ [Y; Z]

    # 🔑 The consequence that made this a blocking decision rather than a preference. With a
    # centred design and a centred target, the unpenalized intercept sits at the target mean and
    # the penalized one is pulled toward zero -- so under `:standardise`, where the intercept has
    # to carry the signal level, `λ‖X‖_F²` shrinks the level itself.
    Xc = copy(X); Xc[:, 1:(m - 1)] .-= mean(Xc[:, 1:(m - 1)], dims = 1)
    Yc = Y .+ 100.0                                      # a large level to carry
    bfree = TSLayer.fit_ridge(Xc, Yc; lambda = 1e3, penalize_intercept = false)[m, :]
    bpen = TSLayer.fit_ridge(Xc, Yc; lambda = 1e3, penalize_intercept = true)[m, :]
    @test all(abs.(bfree .- vec(mean(Yc, dims = 1))) .< 1e-6)
    @test all(abs.(bpen) .< abs.(bfree))
end

@testitem "V16 default is unpenalized -- paper 4's convention" default_imports = false setup = [TSLayer, FitFixture] begin
    using Test
    X, Y = collinear()
    @test TSLayer.fit_ridge(X, Y; lambda = 0.1) ==
          TSLayer.fit_ridge(X, Y; lambda = 0.1, penalize_intercept = false)
end

@testitem "V25 rank and pinv check at lambda = 0" default_imports = false setup = [TSLayer, FitFixture] begin
    using Test
    using LinearAlgebra
    # `plan.md` §0 item 11: Julia's `\` returns a minimum-norm solution on a design it judges
    # rank-deficient, so `λ = 0` is not a guaranteed unregularized fit. That underwrites a
    # falsification criterion, hence the metric (#25) rather than a comment.
    X, Y = collinear()
    gap = norm(X \ Y - pinv(X) * Y)
    @test isfinite(gap)
    @test rank(X) <= size(X, 2)

    # Made explicit on a design that is exactly rank-deficient: the two solutions differ, so the
    # check has power.
    Xd = [X[:, 1:(end - 1)] X[:, 1] X[:, end]]
    @test rank(Xd) < size(Xd, 2)
    @test norm(Xd \ Y - pinv(Xd) * Y) > 1e-8
end
