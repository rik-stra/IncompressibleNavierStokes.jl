# Mechanism diagnostics -- regime 0, computed from the fitted parameters and the design matrix.
#
# Metrics #21 (rho of the lifted companion), #22 (starred-block gain), #23 (Gram spectrum) and
# #24 (total block sum and bias). None of them needs a trajectory.
#
# ⚠️ Every number here is convention-dependent, so a `Scaling` should accompany any that is
# reported. #23 in particular: an uncentred design adds a leading eigenvalue of order
# `N*(mean/sigma)^2` to `X'X`, which moves the very spectrum the metric reads.

"""
    gram_diagnostics(X; lambda = 0.0, tol = nothing)

Metric #23. One SVD of the design matrix, yielding every number the ridge question needs.

Returns `(; sigma, sigma2, sigma_min2, sigma_max2, cond, rank, alpha, n_erased, n_untouched,
branch, lambda, effective_dof, pinv_gap)`.

Ridge replaces each squared singular value `s_j^2` by `s_j^2 + lambda`, so direction `j` survives
with weight

    alpha_j = s_j^2 / (s_j^2 + lambda)

Directions with `s_j^2` far above `lambda` are untouched; those far below are erased. Comparing
`lambda` against the spectrum therefore says which directions the penalty is acting on, and that
decides which of three mutually exclusive stories the paper tells:

  * `:rank_deficiency_fix` -- `lambda` sits inside the small end. Its accuracy cost is the
    suppression of exactly the collinear directions a long history exists to supply, and the
    memory-representation lever follows directly.
  * `:shrinks_determined_directions` -- `lambda` reaches well-determined directions. The model is
    being shrunk toward its own marginal, and that carries the mechanism.
  * `:inactive` -- `lambda` is below the whole spectrum. Ridge does essentially nothing and paper
    2's documented `lambda` effect is not the penalty; the remaining candidates are sampling (its
    appendix rests on five replicas) and numerics (an unconverged ADMM solve, or silent rank
    truncation in the `lambda = 0` backslash).

`branch` reports which, using the quartiles of the spectrum: `:inactive` when `lambda` is below
`sigma_min2`, `:shrinks_determined_directions` when it exceeds the median `s_j^2`, and
`:rank_deficiency_fix` in between.

`effective_dof = sum(alpha)` is the usual ridge degrees of freedom, and `pinv_gap` is metric #25 --
`norm(X\\Y - pinv(X)*Y)` cannot be computed without a `Y`, so what is returned here is the
rank-based warning sign: the number of singular values below `tol`, i.e. how many directions the
backslash would silently truncate at `lambda = 0`.
"""
function gram_diagnostics(X::AbstractMatrix; lambda = 0.0, tol = nothing)
    Xf = Matrix{Float64}(X)
    s = svdvals(Xf)
    s2 = s .^ 2
    lam = float(lambda)
    t = tol === nothing ? maximum(s) * max(size(Xf)...) * eps(Float64) : float(tol)
    alpha = s2 ./ (s2 .+ lam)
    med = s2[max(1, cld(length(s2), 2))]
    branch = if lam <= 0 || lam < minimum(s2)
        :inactive
    elseif lam > med
        :shrinks_determined_directions
    else
        :rank_deficiency_fix
    end
    return (; sigma = s, sigma2 = s2, sigma_min2 = minimum(s2), sigma_max2 = maximum(s2),
            cond = maximum(s) / minimum(s), rank = count(>(t), s), alpha,
            n_erased = count(<(0.5), alpha), n_untouched = count(>(0.99), alpha),
            branch, lambda = lam, effective_dof = sum(alpha),
            n_below_tol = count(<=(t), s), tol = t)
end

"""
    pinv_gap(X, Y)

Metric #25: is `lambda = 0` really an unregularized fit?

Returns `(; rank, ncol, rank_deficit, cond, gap, rel)`.

🔴 **The metric as specified in `plan.md` §0 item 11 does not work, and the fix is in which number
you read.** The specification is `norm(X\\Y - pinv(X)*Y)`, on the reasoning that Julia's backslash
returns a minimum-norm solution on a design it judges rank-deficient, so a nonzero gap would reveal
that `lambda = 0` was never unregularized. The premise is right and the inference does not follow:
because the backslash returns *the same* minimum-norm solution that `pinv` does, the gap is about
`1e-15` **exactly when** truncation happens. Measured on Julia 1.12.7 over an appended duplicate
column, an inserted duplicate, a zero column and an exact linear combination: rank drops from 4 to
3 in every case and the gap stays at `1e-15`, with identical solution norms. The metric is blind to
the thing it was written to catch.

**What is load-bearing instead:**

  * `rank_deficit = ncol - rank`. Any positive value means the backslash *did* select a
    minimum-norm solution out of an affine space of exact minimisers, so `lambda = 0` carried an
    implicit regularization. This is the number that guards the falsification criterion.
  * `cond`, and `gram_diagnostics`' `n_below_tol`, for how close to that the design is.
  * `rel`, the *relative* gap, which does have power on a **near**-deficient design: on a column
    duplicated to within `1e-12` the rank test reports full rank while the two solutions differ by
    `5.6e6` on a solution norm of `5.7e10`, a relative gap of about `1e-4`. That is the regime the
    real design sits in -- the lagged `q` and `q*` streams differ only by the correction -- so the
    relative gap is worth reporting even though the absolute one is not.

Costs one line either way, and is reported at every `(h, lambda)`.
"""
function pinv_gap(X::AbstractMatrix, Y::AbstractMatrix)
    Xf = Matrix{Float64}(X)
    Yf = Matrix{Float64}(Y)
    direct = Xf \ Yf
    mn = pinv(Xf) * Yf
    r = rank(Xf)
    return (; rank = r, ncol = size(Xf, 2), rank_deficit = size(Xf, 2) - r, cond = cond(Xf),
            gap = norm(direct - mn), rel = norm(direct - mn) / max(norm(mn), eps()))
end

"""
    coefficient_blocks(C, spec)

Split a fitted mean map into the blocks the diagnostics talk about.

`C` is `nfeatures x N_Q`, the orientation `fit_ridge` returns, so a prediction is the row
`x' * C`. As operators on column vectors the blocks are therefore transposed:

  * `A_star` -- response to the current predictor `q^{n*}`, `N_Q x N_Q`, or `nothing`.
  * `A[k]` -- response to the corrected QoI at lag `k`, `q^{n-k}`.
  * `B[k]` -- response to the predictor at lag `k`, `q^{n-k*}`.
  * `b` -- the intercept, length `N_Q`.
"""
function coefficient_blocks(C::AbstractMatrix, spec::HistorySpec)
    nq, h = spec.n_qoi, spec.h
    cs = qstar_columns(spec)
    A_star = cs === nothing ? nothing : Matrix(transpose(C[cs, :]))
    A = Matrix{eltype(C)}[]
    B = Matrix{eltype(C)}[]
    for k in 1:h
        cq = qlag_columns(spec, k)
        cq === nothing || push!(A, Matrix(transpose(C[cq, :])))
        cb = qstarlag_columns(spec, k)
        cb === nothing || push!(B, Matrix(transpose(C[cb, :])))
    end
    return (; A_star, A, B, b = collect(C[bias_column(spec), :]))
end

"""
    total_block_sum(C, spec)

Metric #24: `S = A_star + sum_k A_k + sum_k B_k`, the mean map's response to a constant unit input,
plus the intercept `b`.

Says how a model splits the signal level between coefficients and intercept.

!!! note
    Printed and reported, never predicted. It replaced a withdrawn prediction that `S` would be
    close to the identity for a specific reason -- "at leading order every lag in the window is
    approximately `q^{n-1}`, so only the total is implied" -- and the measurement is now available:
    on HIT `S` is close to the identity under **both** normalization conventions, agreeing to four
    digits, which means the coefficient block carries the level in both cases and the intercept
    carries only the residual mean. The claim that an uncentred convention forces the *intercept*
    to carry the level is wrong; what the convention changes is the Gram spectrum (#23).
"""
function total_block_sum(C::AbstractMatrix, spec::HistorySpec)
    bl = coefficient_blocks(C, spec)
    S = zeros(Float64, spec.n_qoi, spec.n_qoi)
    bl.A_star === nothing || (S .+= bl.A_star)
    for M in bl.A
        S .+= M
    end
    for M in bl.B
        S .+= M
    end
    return (; S, b = bl.b, dev_from_identity = norm(S - I) / sqrt(spec.n_qoi))
end

"""
    companion(C, spec)

The lifted companion propagator `C_tilde` of metric #21, and the closures it required.

`C` is rectangular, so no spectral radius exists until the **exogenous** starred inputs are closed.
Two closures are needed, not one:

  * **Closure A**: `q^{n*}` is approximately `q^{n-1}`, exact to `O(dt)`.
  * **Closure B**: `q^{n-k*}` is approximately `q^{n-k}` -- the **same-index** pairing.

Closure B's form depended on an index convention that three derivations disagreed about, and it is
now settled from the code rather than by argument: `build_history` and the deployed ring buffer
both place the corrected and starred entries of lag `k` at the **same physical step**
(`test/test_history.jl`, the SC-48 item). So the companion is `h*N_Q` square with first block row

    {A_1 + A_star + B_1, A_2 + B_2, ..., A_h + B_h}

and the one-step-older alternative, which would have given `(h+1)*N_Q`, does not apply here.

Returns `(; L, closures, size)`.
"""
function companion(C::AbstractMatrix, spec::HistorySpec)
    spec.hist_var === :q_star_q ||
        error("companion is derived for hist_var = :q_star_q; got $(spec.hist_var)")
    nq, h = spec.n_qoi, spec.h
    bl = coefficient_blocks(C, spec)
    first_row = Matrix{Float64}[]
    for k in 1:h
        M = Matrix{Float64}(bl.A[k]) .+ Matrix{Float64}(bl.B[k])
        k == 1 && bl.A_star !== nothing && (M = M .+ Matrix{Float64}(bl.A_star))
        push!(first_row, M)
    end
    L = zeros(Float64, h * nq, h * nq)
    for k in 1:h
        L[1:nq, ((k - 1) * nq + 1):(k * nq)] .= first_row[k]
    end
    for k in 2:h
        L[((k - 1) * nq + 1):(k * nq), ((k - 2) * nq + 1):((k - 1) * nq)] .= I(nq)
    end
    return (; L, closures = (:qstar_is_previous_q, :same_index_pairing), size = h * nq)
end

"""
    rho(C, spec)

Metric #21: the modulus of the largest eigenvalue of [`companion`](@ref) -- the growth rate of the
fastest-growing mode of the model's **standalone** QoI process.

Its use here is narrow: the free end of the sweep-adequacy test must reach `rho > 1`, or the
unstable end was never sampled and S2''s falsification clause fires for the wrong reason.

!!! warning
    Not a stability guarantee. In deployment the low-fidelity solver supplies `q^{n*}` and most of
    `dq/dt`, so `rho < 1` is neither necessary nor sufficient for coupled stability. What it rules
    out is an exponentially growing injected signal. Undefined for a nonlinear mean.
"""
rho(C::AbstractMatrix, spec::HistorySpec) = maximum(abs, eigvals(companion(C, spec).L))

"""
    starred_gain(C, spec; nfreq = 512)

Metric #22: the gain from the starred input block to `q`.

Two numbers: the `l2` gain `opnorm(G)` of the static starred block, and the `Hinf` gain
`sup over omega of opnorm(G(exp(i*omega)))` of the transfer function from the starred input to `q`
through the closed `q`-lag loop.

**The more honest feedback diagnostic than #21**, because deployment couples solver and surrogate
through the *exogenous* input rather than through an autonomous mode. Unlike `rho` it is also
estimable empirically, so it survives for a model whose companion is undefined.
"""
function starred_gain(C::AbstractMatrix, spec::HistorySpec; nfreq::Int = 512)
    bl = coefficient_blocks(C, spec)
    nq, h = spec.n_qoi, spec.h
    G0 = bl.A_star === nothing ? zeros(Float64, nq, nq) : Matrix{Float64}(bl.A_star)
    l2 = opnorm(G0)
    # q(z) = [I - sum_k A_k z^{-k}]^{-1} ( A_star + sum_k B_k z^{-k} ) qstar(z)
    hinf = 0.0
    hinf_at = 0.0
    for j in 0:(nfreq - 1)
        w = pi * j / (nfreq - 1)
        z = cis(w)
        Den = Matrix{ComplexF64}(I(nq))
        Num = Matrix{ComplexF64}(G0)
        for k in 1:h
            zk = z^(-k)
            Den .-= Matrix{ComplexF64}(bl.A[k]) .* zk
            Num .+= Matrix{ComplexF64}(bl.B[k]) .* zk
        end
        g = opnorm(Den \ Num)
        if g > hinf
            hinf = g
            hinf_at = w
        end
    end
    return (; l2, hinf, hinf_at)
end
