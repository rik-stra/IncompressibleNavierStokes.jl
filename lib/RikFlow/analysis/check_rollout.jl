# Diagnostics for the free-running rollout.
#
#   V2  batch-vs-online history identity: build_history and HistoryBuffer must agree exactly.
#       Without this, a divergent rollout cannot be distinguished from a wiring mistake.
#   S1  noise-free rollout: if the deterministic mean map diverges, the instability is in C and
#       not in the noise model.
#   S2  where it diverges, in steps and in units of the reference standard deviation.
#
# 🔴 **Partition: `train_frac = 0.75` of the record passed, NOT results.md's.** The held-out NLL
# and the blow-up steps below come off that exploratory split; `analysis/results.md` and
# `score_m0_ddn.jl` use paper 2's own `train_range = (400, 4000)` -- 1-10 TU train, 10-100 TU held
# out -- and fit the scaling on a different span as well. Numbers from here are **not** comparable
# with the same quantity there. The three-way comparison of the splits in this directory is in
# `ladder_setup.jl`'s header.
#
# ⚠️ **Largely superseded.** V2 is now covered properly by `test/test_history.jl`, and the blow-up
# sweep duplicates `fig_stability` in `plot_models.jl` plus `spectral_radius`/`companion` in
# `ts_spectrum.jl`. What is still unique is the *empirical* blow step, which plan section 14's
# **V24** (still TODO) should own as a test rather than as a printout.

using LinearAlgebra, Statistics, Random, Printf, JLD2

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "ts_history.jl"))
include(joinpath(SRC, "ts_models.jl"))
include(joinpath(SRC, "ts_fit.jl"))
include(joinpath(SRC, "ts_score.jl"))
include(joinpath(SRC, "ts_rollout.jl"))

include(joinpath(@__DIR__, "extract_qois.jl"))

# The partition, the standardisation and the rollout handoff, shared with `fit_offline.jl` and
# `plot_models.jl`. Included after the `ts_*` sources because it uses them.
include(joinpath(@__DIR__, "ladder_setup.jl"))

"""
    rollout_trace(model, q0, qs0, qstar; nsteps, noise, rng)

Rollout that reports the first step at which the standardised state leaves `bound`, instead of
running on into NaN.
"""
function rollout_trace(model, q0, qs0, qstar; nsteps, noise = true, rng = MersenneTwister(1),
                       bound = 50.0)
    spec = model.spec
    nq, h = spec.n_qoi, spec.h
    p = ar_order(model)
    a = ar_coefficients(model)
    m = size(model.C, 1)
    buf = HistoryBuffer(spec, Float32)
    for k in h:-1:1
        push!(buf, scale_in(model, view(q0, :, k)), scale_in(model, view(qs0, :, k)))
    end
    L = cholesky(Symmetric(Float64.(model.R))).L
    etah = zeros(Float32, nq, max(p, 1))
    ys = zeros(Float32, nq, nsteps)
    blowstep = 0
    for n in 1:nsteps
        qs = scale_in(model, view(qstar, :, n))
        x = inputvec(buf, qs)
        mu = vec(x' * model.C)
        for k in 1:p
            mu .+= a[k, :] .* view(etah, :, k)
        end
        u = vec(x' * model.W)
        model.uclip === nothing || (u = clamp.(u, model.uclip[1], model.uclip[2]))
        xi = noise ? Float32.(exp.(u) .* (L * randn(rng, nq))) : zeros(Float32, nq)
        eta = copy(xi)
        for k in 1:p
            eta .+= a[k, :] .* view(etah, :, k)
        end
        y = mu .+ xi
        if p > 0
            for k in p:-1:2
                etah[:, k] .= etah[:, k - 1]
            end
            etah[:, 1] .= eta
        end
        ys[:, n] .= y
        if blowstep == 0 && (any(!isfinite, y) || maximum(abs, y) > bound)
            blowstep = n
        end
        push!(buf, y, qs)
    end
    return ys, blowstep
end

function main(file; h = 5, dt = 0.0025, train_frac = 0.75, lambdas = [0.0, 1e-4, 1e-2, 1e-1, 1.0])
    ld = ladder_setup(file; h, train_frac, dt)
    (; nq, scaling, spec, X, Y, qs, qss, steps, ntrain, block) = ld

    # ---- V2: batch vs online history identity ------------------------------------------------
    buf = HistoryBuffer(spec, eltype(X))
    n0 = steps[1]
    for k in h:-1:1
        push!(buf, view(qs, :, n0 - k + 1), view(qss, :, n0 - k))
    end
    maxerr = 0.0
    for (r, n) in enumerate(steps[1:min(500, end)])
        x = inputvec(buf, view(qss, :, n))
        maxerr = max(maxerr, maximum(abs, x .- X[r, :]))
        push!(buf, view(qs, :, n + 1), view(qss, :, n))
    end
    @printf("V2 batch-vs-online history identity: max abs diff = %.3e  (%s)\n",
            maxerr, maxerr < 1e-6 ? "PASS" : "FAIL")

    # ---- rollout stability against lambda ----------------------------------------------------
    (; q0, qs0, qstar_series, nroll) = rollout_handoff(ld)
    qstar = qstar_series

    println("\nrollout stability against lambda, h = $h  (blow step: first |q_std| > 50)")
    for (name, ar, sd) in (("M0   white, constant Sigma", 0, false),
                           ("M0c1 AR(1), constant Sigma", 1, false),
                           ("M0c2 AR(2), constant Sigma", 2, false))
        println("\n  ", name)
        @printf("  %10s %14s %14s %14s\n", "lambda", "heldout NLL", "blow (det)", "blow (stoch)")
        for lam in lambdas
            model, _ = fit_joint(X[1:ntrain, :], Y[1:ntrain, :], spec; ar_order = ar,
                                 state_dependent = sd, lambda = lam, iters = 25,
                                 block_id = block[1:ntrain], scaling)
            rows_ho = valid_rows(block[(ntrain + 1):end], ar)
            nl = nll(X[(ntrain + 1):end, :], Y[(ntrain + 1):end, :], model.C, model.W, model.R,
                     ar_coefficients(model), block[(ntrain + 1):end], rows_ho)
            per = gaussian_nll_per_sample(nl, length(rows_ho), nq)
            _, bd = rollout_trace(model, q0, qs0, qstar; nsteps = nroll, noise = false)
            _, bs = rollout_trace(model, q0, qs0, qstar; nsteps = nroll, noise = true)
            @printf("  %10.4g %14.5f %14s %14s\n", lam, per,
                    bd == 0 ? "stable" : string(bd), bs == 0 ? "stable" : string(bs))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end
