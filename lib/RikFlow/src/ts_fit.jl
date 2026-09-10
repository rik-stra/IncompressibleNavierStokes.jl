# Joint fit of mean, log-scale head and residual autoregression under one Gaussian likelihood.
#
#   L(theta) = sum_n [ sum_i u_in + 0.5 logdet R + 0.5 z_n' Rinv z_n ] + lambda ||theta||^2
#   mu_eff^n = (x_n C)' + sum_k A_k E^{n-k},    E^n = y^n - (x_n C)'
#   u_in     = x_n w_i  (clipped),  d_in = exp(u_in),  z_n = D_n^{-1} (y_n - mu_eff^n)
#
# Block coordinate descent, because three of the four blocks have closed forms:
#   C   exact generalized-least-squares solve on the quasi-differenced design
#   R   concentrated out: the sample correlation of the standardised residuals
#   W   closed form when Sigma is constant; gradient with backtracking otherwise
#   psi finite-difference gradient with backtracking (p*N_Q <= 24 parameters)
#
# A first-order optimiser is the wrong tool here: C starts at its exact ridge optimum, and a
# fixed-step method moves it away faster than the other blocks can compensate.

"""
    shift_rows(E, k, block_id)

Shift `E` down by `k` rows, writing zeros wherever the shift would cross a block boundary. Blocks
are maximal runs of equal `block_id`. This is the block masking the joint estimator needs: rows
`1..p` of every contiguous block carry no autoregressive term, and a forgotten mask silently
produces a different estimator across cross-validation folds.
"""
function shift_rows(E::AbstractMatrix{T}, k::Int, block_id::AbstractVector) where {T}
    S = zeros(T, size(E))
    @inbounds for n in (k + 1):size(E, 1)
        if block_id[n] == block_id[n - k]
            S[n, :] .= @view E[n - k, :]
        end
    end
    return S
end

"""
    valid_rows(block_id, p)

Rows that carry a full set of `p` autoregressive lags within their own block. The likelihood is
conditional on the first `p` rows of each block, which are excluded rather than fitted with a
truncated mean.
"""
function valid_rows(block_id::AbstractVector, p::Int)
    n = length(block_id)
    v = trues(n)
    for i in 1:n, k in 1:p
        if i - k < 1 || block_id[i] != block_id[i - k]
            v[i] = false
            break
        end
    end
    return findall(v)
end

"""
    fit_ridge(X, Y; lambda = 0.0, penalize_intercept = false)

Multivariate ridge with a shared design. The intercept is the last column of `X`.

At `lambda = 0` all `N_Q` outputs share one regressor matrix, so generalized least squares collapses
to equation-by-equation least squares and this is the joint Gaussian maximum likelihood estimate.

Solved by QR on the augmented system `[X; sqrt(lambda) P] \\ [Y; 0]`, **not** by the normal
equations. The lag basis is strongly collinear -- `q^{n-k}` and `q^{n-k*}` differ only by the
correction -- and forming `X'X` squares its condition number. Measured against paper 2's archived
`LinReg1` fit, the normal-equation route is wrong by a relative 5.8 in the coefficients and 5.1e-3
in the predictions at `lambda = 0`, while this one reproduces the archive to 8e-7 relative.
"""
function fit_ridge(X::AbstractMatrix{T}, Y::AbstractMatrix{T}; lambda = zero(T),
                   penalize_intercept = false) where {T}
    n, m = size(X)
    lambda == 0 && return X \ Y
    P = Matrix{T}(I, m, m) * T(sqrt(lambda))
    penalize_intercept || (P[m, m] = zero(T))
    return [X; P] \ [Y; zeros(T, m, size(Y, 2))]
end

"""
    quasi_design(X, a_i, block_id)

The quasi-differenced regressor `X - sum_k a_k * shift(X, k)` for one QoI. Regressing the
quasi-differenced target on this is algebraically the same model as carrying the shifted residual in
the conditional mean; both forms are implemented and their agreement is a standing check (V3).
"""
function quasi_design(X::AbstractMatrix{T}, a_i::AbstractVector, block_id) where {T}
    Xt = copy(X)
    for k in 1:length(a_i)
        Xt .-= T(a_i[k]) .* shift_rows(X, k, block_id)
    end
    return Xt
end

"""
    nll_parts(X, Y, C, W, a, block_id; uclip)

Everything the objective and its gradients need, in one pass.
"""
function nll_parts(X, Y, C, W, a, block_id; uclip = nothing)
    Mlin = X * C
    E = Y .- Mlin
    M = copy(Mlin)
    for k in 1:size(a, 1)
        M .+= shift_rows(E, k, block_id) .* transpose(@view a[k, :])
    end
    Res = Y .- M
    U = X * W
    if uclip !== nothing
        U = clamp.(U, uclip[1], uclip[2])
    end
    D = exp.(U)
    Z = Res ./ D
    return (; Mlin, E, M, Res, U, D, Z)
end

"""
    nll(X, Y, C, W, R, a, block_id, rows; uclip)

Negative log-likelihood, summed over `rows`, dropping the `N/2 * N_Q * log(2pi)` constant.
"""
function nll(X, Y, C, W, R, a, block_id, rows; uclip = nothing)
    P = nll_parts(X, Y, C, W, a, block_id; uclip)
    Zv = @view P.Z[rows, :]
    quad = sum((Zv / R) .* Zv)
    return sum(@view P.U[rows, :]) + 0.5 * length(rows) * logdet(R) + 0.5 * quad
end

"""
    concentrated_R(Z, rows)

Closed-form minimiser of the likelihood over the constant correlation matrix: the sample
correlation of the standardised residuals. This removes `N_Q(N_Q-1)/2` parameters and the
positive-definite-with-unit-diagonal constraint from the optimisation.

Bollerslev's information matrix is not block diagonal between `D` and `R`, so a confidence interval
on the covariance head still has to come from the full likelihood, not from this correlation.
"""
function concentrated_R(Z::AbstractMatrix{T}, rows) where {T}
    Zv = @view Z[rows, :]
    S = (Zv' * Zv) ./ length(rows)
    d = sqrt.(diag(S))
    R = S ./ (d * d')
    R .= 0.5 .* (R .+ R')
    R[diagind(R)] .= one(T)
    return R
end

"""
    solve_C(X, Y, W, R, a, block_id, rows; lambda, uclip)

Exact generalized-least-squares solve for the mean block, given the scale head, the correlation and
the autoregression. Each QoI has its own quasi-differenced design once the `a_i` differ, so the
equations are coupled through `Sigma^{-1}` and one `m*N_Q` system has to be solved.

That system is solved by QR on a **whitened stacked design**, not by assembling `X' W X`. Writing
`Sigma_n^{-1} = D_n^{-1} L' L D_n^{-1}` with `L` the Cholesky factor of `R^{-1}`, the objective is
an ordinary least-squares problem in `vec(C)` whose row block for sample `n` is
`(L D_n^{-1})[:, i] * xtilde_{n,i}'`. Forming the normal equations instead squares the condition
number of an already collinear lag basis, which is what produced spurious spectral radii of 2.3 and
3.7 at isolated `lambda` in the coloured rungs while the white rung stayed smooth.
"""
function solve_C(X::AbstractMatrix{T}, Y, W, R, a, block_id, rows; lambda = zero(T),
                 uclip = nothing) where {T}
    m, nq = size(X, 2), size(Y, 2)
    p = size(a, 1)

    Yt = copy(Y)
    for k in 1:p
        Yt .-= shift_rows(Y, k, block_id) .* transpose(@view a[k, :])
    end

    # Shared-design shortcut. With p = 0, or whenever every QoI carries the same autoregression,
    # all N_Q equations have the same regressor, and at lambda = 0 generalized least squares
    # collapses to equation-by-equation least squares whatever Sigma is.
    #
    # ⚠️ At lambda > 0 that collapse no longer holds: the penalty breaks the equivalence, so this
    # branch returns the *unweighted* ridge (penalty lambda) rather than the Sigma-weighted optimum
    # (penalty 2*lambda, since `nll` carries the quadratic form with a factor 1/2). Measured on a
    # well-conditioned synthetic problem the gap is a relative 4e-6 in the objective. It is left in
    # place because `fit_joint` guards every block with `keep_better` and so cannot accept a step
    # that raises the objective, and because routing this branch through the whitened QR below
    # would allocate an (n_rows*N_Q) x (m*N_Q) design -- ~290 MB on the 100 TU record.
    shared = p == 0 || all(isapprox(a[:, i], a[:, 1]) for i in 1:nq)
    if shared
        Xt = p == 0 ? X : quasi_design(X, view(a, :, 1), block_id)
        return fit_ridge(Xt[rows, :], Yt[rows, :]; lambda)
    end

    U = X * W
    uclip === nothing || (U = clamp.(U, uclip[1], uclip[2]))
    D = exp.(U)
    Xt = [quasi_design(X, view(a, :, i), block_id) for i in 1:nq]

    # L' L = R^{-1}, so that the whitened residual is L * D_n^{-1} * (y_n - mu_n).
    L = Matrix(cholesky(Symmetric(inv(Symmetric(Float64.(R))))).U)

    # Sequential (updating) QR. Materialising the whole whitened design would need an
    # (n_rows*N_Q) x (m*N_Q) matrix -- 180000 x 402, about 290 MB, on the 100 TU record, which is
    # enough to run the machine out of memory. Instead the rows are folded in a chunk at a time:
    # `acc` holds [R z] for everything seen so far, and QR of [acc; next chunk] is again [R z]
    # because the orthogonal factor never changes the least-squares solution.
    nblk = m * nq
    acc = zeros(T, 0, nblk + 1)
    chunk = max(1, cld(2_000_000, nblk))       # rows per chunk, ~8 MB of Float32 at a time
    buf = zeros(T, chunk * nq, nblk + 1)

    @inbounds for lo in 1:chunk:length(rows)
        hi = min(lo + chunk - 1, length(rows))
        nb = hi - lo + 1
        fill!(buf, zero(T))
        for (r, n) in enumerate(view(rows, lo:hi))
            base = (r - 1) * nq
            for c in 1:nq                      # c indexes the whitened equation
                gv = zero(T)
                for i in 1:nq                  # i indexes the QoI, and so the block of vec(C)
                    w = T(L[c, i]) / D[n, i]
                    w == 0 && continue
                    @views buf[base + c, ((i - 1) * m + 1):(i * m)] .+= w .* Xt[i][n, :]
                    gv += w * Yt[n, i]
                end
                buf[base + c, nblk + 1] = gv
            end
        end
        S = vcat(acc, view(buf, 1:(nb * nq), :))
        acc = Matrix(qr(S).R)
        size(acc, 1) > nblk + 1 && (acc = acc[1:(nblk + 1), :])
    end

    if lambda != 0
        # `nll` carries the quadratic form with a factor 1/2, so a penalty `lambda*norm(C)^2` on
        # that objective is `2*lambda` on the plain least-squares problem solved here. This keeps
        # the effective shrinkage identical to the normal-equation version this replaces.
        P = zeros(T, nblk, nblk + 1)
        s = T(sqrt(2 * lambda))
        for j in 1:nblk
            P[j, j] = s
        end
        for i in 1:nq                          # the bias row of each block stays unpenalised
            P[i * m, i * m] = zero(T)
        end
        acc = Matrix(qr(vcat(acc, P)).R)
        size(acc, 1) > nblk + 1 && (acc = acc[1:(nblk + 1), :])
    end

    return reshape(UpperTriangular(view(acc, 1:nblk, 1:nblk)) \ view(acc, 1:nblk, nblk + 1), m, nq)
end

"""
    solve_W_constant(Res, rows, m, nq)

Closed form for the constant-`Sigma` case: the scale is the residual standard deviation, carried in
the bias row of `W`, and every other row is zero. This is the `W`-zero-outside-its-bias-row
restriction of the ladder, solved rather than iterated.
"""
function solve_W_constant(Res::AbstractMatrix{T}, rows, m, nq) where {T}
    W = zeros(T, m, nq)
    W[m, :] .= log.(vec(std(view(Res, rows, :); dims = 1, corrected = false)))
    return W
end

"""
    fit_joint(X, Y, spec; ar_order, state_dependent, lambda, iters, ...)

Fit `(C, W, psi)` jointly under one likelihood, with `R` concentrated out.

Returns `(model, history)` with `history.nll` the objective after each sweep. It must be
non-increasing; the driver asserts this.
"""
function fit_joint(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, spec::HistorySpec;
                   ar_order::Int = 0, state_dependent::Bool = false, lambda = zero(T),
                   iters::Int = 30, uclip_margin = 1.0, tol = 1e-9,
                   block_id = ones(Int, size(X, 1)), scaling = NamedTuple(),
                   fixed_C = nothing, verbose = false) where {T}
    m, nq = size(X, 2), size(Y, 2)
    p = ar_order
    rows = valid_rows(block_id, p)

    # `fixed_C` freezes the mean map instead of fitting it. That is what turns the rungs whose
    # mean is not a regression -- paper 1's data-driven noise model, and the "noise only" ablations
    # -- into the same estimator as the rest, so the residual model is fitted against a mean that
    # is prescribed rather than estimated.
    C = fixed_C === nothing ? fit_ridge(X, Y; lambda) : copy(fixed_C)
    psi = zeros(T, p, nq)
    P = nll_parts(X, Y, C, zeros(T, m, nq), ar_from_psi(psi), block_id)
    W = solve_W_constant(P.Res, rows, m, nq)
    if p > 0
        for i in 1:nq
            r1 = lag1_autocorrelation(view(P.Res, rows, i))
            psi[1, i] = atanh(clamp(r1, T(-0.95), T(0.95)))
        end
    end
    P = nll_parts(X, Y, C, W, ar_from_psi(psi), block_id)
    R = concentrated_R(P.Z, rows)

    obj(C, W, psi, R) = nll(X, Y, C, W, R, ar_from_psi(psi), block_id, rows)
    hist = Float64[obj(C, W, psi, R)]

    # Every block step is guarded: a step that does not lower the objective is rejected. The
    # closed forms are exact for the block they solve, but they are exact only for the *current*
    # values of the others, and a numerically degenerate solve must not be allowed through.
    keep_better(new, old, f) = (isfinite(f(new)) && f(new) <= f(old)) ? new : old

    for it in 1:iters
        a = ar_from_psi(psi)

        # --- C: exact GLS -----------------------------------------------------------------
        if fixed_C === nothing
            C = keep_better(solve_C(X, Y, W, R, a, block_id, rows; lambda), C,
                            Cn -> obj(Cn, W, psi, R))
        end

        # --- W ----------------------------------------------------------------------------
        P = nll_parts(X, Y, C, W, a, block_id)
        if state_dependent
            Rinv = inv(R)
            ZR = P.Z * Rinv
            G = zeros(T, size(X, 1), nq)
            @views G[rows, :] .= one(T) .- P.Z[rows, :] .* ZR[rows, :]
            g = X[rows, :]' * G[rows, :] .+ 2 * T(lambda) .* W
            W = backtrack(Wn -> obj(C, Wn, psi, R), W, g)
        else
            W = keep_better(solve_W_constant(P.Res, rows, m, nq), W, Wn -> obj(C, Wn, psi, R))
        end

        # --- psi --------------------------------------------------------------------------
        if p > 0
            f0 = obj(C, W, psi, R)
            g = zeros(T, p, nq)
            h = T(1e-5)
            for i in 1:nq, k in 1:p
                psi[k, i] += h
                g[k, i] = (obj(C, W, psi, R) - f0) / h
                psi[k, i] -= h
            end
            psi = backtrack(pn -> obj(C, W, pn, R), psi, g)
        end

        # --- R: concentrated ---------------------------------------------------------------
        P = nll_parts(X, Y, C, W, ar_from_psi(psi), block_id)
        R = keep_better(concentrated_R(P.Z, rows), R, Rn -> obj(C, W, psi, Rn))

        push!(hist, obj(C, W, psi, R))
        verbose && println("  iter $it  nll = $(hist[end])")
        if abs(hist[end - 1] - hist[end]) <= tol * max(1.0, abs(hist[end]))
            break
        end
    end

    uclip = nothing
    if uclip_margin !== nothing && state_dependent
        U = X * W
        uclip = (T(minimum(U) - uclip_margin), T(maximum(U) + uclip_margin))
    end
    return JointModel(spec, C, W, R, psi, uclip, scaling), (; nll = hist)
end

"""
    backtrack(f, x, g; s0, maxhalve)

One backtracking gradient step: halve the step until the objective decreases, or give up and return
`x` unchanged. The step is scaled by `norm(x)/norm(g)` so it is invariant to the gradient's units.
"""
function backtrack(f, x::AbstractArray{T}, g::AbstractArray{T}; s0 = T(1.0),
                   maxhalve::Int = 30) where {T}
    ng = sqrt(sum(abs2, g))
    ng == 0 && return x
    scale = max(sqrt(sum(abs2, x)), one(T)) / ng
    f0 = f(x)
    s = s0 * scale
    for _ in 1:maxhalve
        xn = x .- s .* g
        if isfinite(f(xn)) && f(xn) < f0
            return xn
        end
        s /= 2
    end
    return x
end

ar_from_psi(psi::AbstractMatrix{T}) where {T} =
    size(psi, 1) == 0 ? zeros(T, 0, size(psi, 2)) :
    reduce(hcat, [pacf_to_ar(tanh.(psi[:, i])) for i in 1:size(psi, 2)])

function lag1_autocorrelation(x)
    n = length(x)
    mu = sum(x) / n
    num = sum((x[i] - mu) * (x[i - 1] - mu) for i in 2:n)
    den = sum((x[i] - mu)^2 for i in 1:n)
    return den == 0 ? 0.0 : num / den
end
