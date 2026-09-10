# Free-running (uncoupled) rollout of a fitted model.
#
# This is the offline half of deployment: the model advances its own corrected-QoI history, drawing
# a new innovation every step, with no solver in the loop.
#
# The predictor stream q^{n*} normally comes from the LF solver. Uncoupled it has to be supplied,
# and `replay_qstar` does so from the tracked run. That makes the starred stream open-loop while the
# corrected stream is closed-loop, which is a weaker test than online: the trajectory cannot drift
# the way it would when the solver also responds. A model whose history is `:q` only is autonomous
# and needs no such replay.

"""
    rollout(model, q0, q_star0; nsteps, q_star_series = nothing, rng, spinup)

Advance a fitted [`JointModel`](@ref) for `nsteps` without a solver.

# Arguments
- `q0`: `N_Q x h` seed of corrected QoIs, column 1 most recent, in **physical** units.
- `q_star0`: `N_Q x h` seed of predictor QoIs, same convention.
- `q_star_series`: `N_Q x nsteps` replayed predictor stream. Required whenever the history spec
  uses the starred stream or includes the predictor.
- `spinup`: number of leading steps to discard from the returned trajectory.

# Returns
`(q, eta)` with `q` of size `N_Q x nsteps` in physical units and `eta` the innovation-plus-memory
residual actually realised, in standardised units.

The residual is seeded from the unconditional covariance rather than from zero, which removes the
start-up transient instead of waiting it out.
"""
function rollout(model::JointModel{T}, q0, q_star0; nsteps::Int,
                 q_star_series = nothing, rng = Random.default_rng(), spinup::Int = 0,
                 noise::Bool = true) where {T}
    spec = model.spec
    nq, h = spec.n_qoi, spec.h
    p = ar_order(model)
    a = ar_coefficients(model)
    m = size(model.C, 1)

    needs_star = spec.include_predictor || spec.hist_var in (:q_star, :q_star_q)
    if needs_star && q_star_series === nothing
        error("history spec uses the predictor stream; supply q_star_series")
    end

    buf = HistoryBuffer(spec, T)
    for k in h:-1:1
        push!(buf, scale_in(model, view(q0, :, k)), scale_in(model, view(q_star0, :, k)))
    end

    Rchol = cholesky(Symmetric(model.R)).L
    etahist = zeros(T, nq, max(p, 1))
    if p > 0 && noise
        # seed from the unconditional covariance implied by the fitted innovation scale
        d0 = exp.(model.W[m, :])
        for k in 1:p
            etahist[:, k] .= d0 .* (Rchol * randn(rng, T, nq)) ./ sqrt.(max.(1 .- a[1, :] .^ 2, T(1e-3)))
        end
    end

    qout = zeros(T, nq, nsteps)
    etaout = zeros(T, nq, nsteps)

    for n in 1:nsteps
        qs = needs_star ? scale_in(model, view(q_star_series, :, n)) : zeros(T, nq)
        x = inputvec(buf, qs)

        mu = vec(x' * model.C)
        for k in 1:p
            mu .+= a[k, :] .* view(etahist, :, k)
        end

        u = vec(x' * model.W)
        if model.uclip !== nothing
            u = clamp.(u, model.uclip[1], model.uclip[2])
        end
        xi = noise ? exp.(u) .* (Rchol * randn(rng, T, nq)) : zeros(T, nq)

        eta = copy(xi)
        for k in 1:p
            eta .+= a[k, :] .* view(etahist, :, k)
        end

        y = mu .+ xi                      # y = mu_eff + xi, and mu_eff already carries the memory
        if p > 0
            for k in p:-1:2
                etahist[:, k] .= etahist[:, k - 1]
            end
            etahist[:, 1] .= eta
        end

        qout[:, n] .= scale_out(model, y)
        etaout[:, n] .= eta
        push!(buf, y, qs)
    end

    keep = (spinup + 1):nsteps
    return qout[:, keep], etaout[:, keep]
end

scale_in(model::JointModel, x) = (x .- model.scaling.mu) ./ model.scaling.sigma
scale_out(model::JointModel, x) = x .* model.scaling.sigma .+ model.scaling.mu

"""
    trajectory_stats(q; dt, maxlag = 200)

Summary statistics used to compare a rollout against the tracked reference: per-QoI mean, standard
deviation, lag-one autocorrelation, and the integral correlation time.
"""
function trajectory_stats(q::AbstractMatrix; dt = 1.0, maxlag::Int = 200)
    nq = size(q, 1)
    return (;
        mean = [sum(view(q, i, :)) / size(q, 2) for i in 1:nq],
        std = [std(view(q, i, :)) for i in 1:nq],
        rho1 = [lag1_autocorrelation(vec(view(q, i, :))) for i in 1:nq],
        tint = [correlation_time(vec(view(q, i, :)), dt).T_int for i in 1:nq],
    )
end

"""
    closed_loop_matrix(model)

Companion matrix of the free-running recursion, in standardised units.

During a rollout the predictor stream is exogenous, so the only feedback is through the lagged
corrected QoIs and through the residual memory. Writing `B_k` for the block of `C` that multiplies
`q^{n-k}` and `a_k` for the residual autoregression, the state
`[q^n, ..., q^{n-h+1}, eta^n, ..., eta^{n-p+1}]` advances by

    q^{n+1}   = sum_k B_k' q^{n+1-k} + sum_k diag(a_k) eta^{n+1-k} + (exogenous)
    eta^{n+1} = sum_k diag(a_k) eta^{n+1-k} + xi^{n+1}

The rollout is stable exactly when the spectral radius of this matrix is below one. The PACF
parametrisation already puts the `eta` block inside the unit circle, so any instability comes from
`B_k`, i.e. from how the ridge distributed weight between the collinear `q` and `q^*` lags.
"""
function closed_loop_matrix(model::JointModel{T}) where {T}
    spec = model.spec
    nq, h = spec.n_qoi, spec.h
    p = ar_order(model)
    a = ar_coefficients(model)
    nb = nq * h                       # q-lag block
    n = nb + nq * p
    M = zeros(T, n, n)

    for k in 1:h
        cols = qlag_columns(spec, k)
        cols === nothing && continue
        M[1:nq, ((k - 1) * nq + 1):(k * nq)] .= transpose(@view model.C[cols, :])
    end
    for j in 2:h                       # shift register on the q lags
        M[((j - 1) * nq + 1):(j * nq), ((j - 2) * nq + 1):((j - 1) * nq)] .= I(nq)
    end
    for k in 1:p                       # residual memory feeds the mean and itself
        cols = (nb + (k - 1) * nq + 1):(nb + k * nq)
        M[1:nq, cols] .= Diagonal(view(a, k, :))
        M[(nb + 1):(nb + nq), cols] .= Diagonal(view(a, k, :))
    end
    for j in 2:p
        M[(nb + (j - 1) * nq + 1):(nb + j * nq), (nb + (j - 2) * nq + 1):(nb + (j - 1) * nq)] .= I(nq)
    end
    return M
end

"""
    spectral_radius(model; part = :all)

Largest `|eigenvalue|` of [`closed_loop_matrix`](@ref). Below one the free-running map contracts;
above one the rollout diverges regardless of the noise model.

The matrix is block triangular, so its spectrum is the union of two independent ones and `part`
selects between them:

- `:mean` -- the lagged-QoI feedback carried by `C`. This is what shrinkage acts on directly.
- `:ar`   -- the residual autoregression, which equals `max_i |root of a_i|`.

Keeping them apart matters: a rollout that destabilises as `lambda` grows is not a failure of
shrinkage on the mean map, it is memory being pushed out of the mean and into the residual, where a
near-unit `a_i` makes the noise a random walk.
"""
function spectral_radius(model::JointModel; part::Symbol = :all)
    M = closed_loop_matrix(model)
    nb = n_qoi(model) * model.spec.h
    rng = part === :mean ? (1:nb) :
          part === :ar   ? ((nb + 1):size(M, 1)) :
          (1:size(M, 1))
    isempty(rng) && return 0.0
    return maximum(abs, eigvals(M[rng, rng]))
end
