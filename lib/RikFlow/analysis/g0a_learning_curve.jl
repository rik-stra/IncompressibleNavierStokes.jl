# G0-a --- out-of-sample learning curve on blocked forward-chaining splits.
#
# Pre-registered in `methodology.md` §6 and `implementation_plan.md` Phase 1b:
#   M0 refitted at >= 7 log-spaced training-window lengths, scored on a held-out block that follows
#   the training window after an embargo, with block length >= 10*tau_int, and block-bootstrap and
#   HAC (Newey-West) confidence intervals on the curve. tau_int and N_eff are descriptive.
#
# What it gates:
#   - the highest fittable rung (saturation below M3's capacity caps the ladder at M2);
#   - whether channel model selection is allowed at all (SC-49: N_eff < 20 on the selection window
#     means configurations are declared, not selected, and the channel gate is under-powered);
#   - the blocked-CV driver that S1 and the calibration gate both reuse.
#
# Usage:
#   julia --project=<env> g0a_learning_curve.jl <tracking.jld2> <hit|channel>

using LinearAlgebra, Statistics, Random, Printf, JLD2

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "ts_history.jl"))
include(joinpath(SRC, "ts_models.jl"))
include(joinpath(SRC, "ts_fit.jl"))
include(joinpath(SRC, "ts_score.jl"))

include(joinpath(@__DIR__, "extract_qois.jl"))

load_tracked(file) = load_qois(file)

# ---------------------------------------------------------------------------------------------
# uncertainty on the mean of a serially correlated score series
# ---------------------------------------------------------------------------------------------

"""
    hac_se(x; bandwidth)

Newey-West standard error of `mean(x)` with a Bartlett kernel. The i.i.d. standard error is wrong
here by construction: consecutive one-step scores inherit the QoI autocorrelation.
"""
function hac_se(x::AbstractVector; bandwidth::Int = max(1, floor(Int, 4 * (length(x) / 100)^(2 / 9))))
    n = length(x)
    n < 3 && return NaN
    e = x .- mean(x)
    g0 = sum(abs2, e) / n
    s = g0
    for k in 1:min(bandwidth, n - 1)
        gk = sum(e[i] * e[i + k] for i in 1:(n - k)) / n
        s += 2 * (1 - k / (bandwidth + 1)) * gk
    end
    s = max(s, 0.0)
    return sqrt(s / n)
end

"""
    block_bootstrap_ci(x, blocklen; nboot, alpha)

Moving-block bootstrap percentile interval for `mean(x)`. Blocks preserve the serial dependence
that an i.i.d. bootstrap would destroy.
"""
function block_bootstrap_ci(x::AbstractVector, blocklen::Int; nboot::Int = 2000, alpha = 0.05,
                            rng = MersenneTwister(7))
    n = length(x)
    b = clamp(blocklen, 1, n)
    nb = cld(n, b)
    n <= b && return (NaN, NaN)
    ms = Vector{Float64}(undef, nboot)
    buf = Vector{Float64}(undef, nb * b)
    for t in 1:nboot
        for j in 1:nb
            s = rand(rng, 1:(n - b + 1))
            @views buf[((j - 1) * b + 1):(j * b)] .= x[s:(s + b - 1)]
        end
        ms[t] = mean(view(buf, 1:n))
    end
    sort!(ms)
    return (ms[max(1, floor(Int, alpha / 2 * nboot))], ms[min(nboot, ceil(Int, (1 - alpha / 2) * nboot))])
end

# ---------------------------------------------------------------------------------------------
# one fold
# ---------------------------------------------------------------------------------------------

"""
    fold_scores(X, Y, tr, te; lambda)

Fit M0 (linear mean, white residual, constant `Sigma`) on the training rows and return the
per-sample negative log-likelihood and squared error on the test block. Returning per-sample series
rather than their means is what lets the HAC and block-bootstrap intervals be formed.
"""
function fold_scores(X, Y, tr, te; lambda)
    C = fit_ridge(X[tr, :], Y[tr, :]; lambda)
    Rtr = Y[tr, :] .- X[tr, :] * C
    nq = size(Y, 2)
    S = (Rtr' * Rtr) ./ length(tr)
    S += 1e-12 * tr_scale(S) * I(nq)
    Rte = Y[te, :] .- X[te, :] * C
    Sinv = inv(S)
    ld = logdet(S)
    nll_i = [0.5 * (nq * log(2pi) + ld + dot(view(Rte, k, :), Sinv, view(Rte, k, :))) / nq
             for k in 1:length(te)]
    rmse_i = [sqrt(sum(abs2, view(Rte, k, :)) / nq) for k in 1:length(te)]
    return nll_i, rmse_i
end

tr_scale(S) = tr(S) / size(S, 1)

# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

function g0a(file; testbed = :hit, h = 5, dt = 0.0025, lambdas = [0.0, 0.01], nwindows = 7,
             labels = nothing, outdir = joinpath(@__DIR__, "output"))
    raw = load_tracked(file)
    nq, T = size(raw.q_star)
    labs = labels === nothing ? ["q$i" for i in 1:nq] : labels

    println("="^86)
    @printf("G0-a  %s   file %s (key %s)\n", uppercase(string(testbed)), basename(file), raw.key)
    @printf("      %d steps, dt = %g, %.1f TU\n", T, dt, T * dt)
    println("="^86)

    # --- descriptive: tau_int and N_eff on the whole record ---------------------------------
    tau_int = [correlation_time(vec(raw.q[i, 1:T]), dt).T_int for i in 1:nq]
    println("\ndescriptive (whole record)")
    @printf("  %-10s %10s %10s %12s\n", "QoI", "tau_int", "tau_int/dt", "N_eff")
    for i in 1:nq
        @printf("  %-10s %10.4f %10.1f %12.1f\n", labs[i], tau_int[i], tau_int[i] / dt,
                T * dt / (2 * tau_int[i]))
    end
    tmax = maximum(tau_int)
    B = ceil(Int, 10 * tmax / dt)         # test-block length, >= 10 tau_int
    emb = ceil(Int, 2 * tmax / dt)        # embargo between train and test
    @printf("\n  block length B = %d steps (10 tau_int_max = %.3f TU), embargo = %d steps\n",
            B, 10 * tmax, emb)

    # --- regressor --------------------------------------------------------------------------
    mu = mean(raw.q; dims = 2)
    sg = std(raw.q; dims = 2)
    spec = HistorySpec(; h, n_qoi = nq, hist_var = :q_star_q, include_predictor = true)
    X, Y, _ = build_history(spec, (raw.q_star .- mu) ./ sg, (raw.q .- mu) ./ sg)
    N = size(X, 1)
    m = size(X, 2)
    @printf("  regressor %d rows x %d cols; usable folds need L + %d + %d <= %d\n", N, m, emb, B, N)

    # Test blocks are FIXED across the whole curve, and each training window is the L rows ending
    # one embargo before its test block. Anchoring the test set this way is what separates the
    # effect of training-set size from the effect of where in the record the test block sits; a
    # forward-chaining design that lets the test position slide with L confounds the two.
    Lmax = max(m + 10, floor(Int, 0.6 * N) - emb)
    first_test = Lmax + emb + 1
    test_starts = collect(first_test:B:(N - B + 1))
    if isempty(test_starts) || Lmax < m + 10
        println("\n  ABORT: no training window long enough for a fold. Record too short.")
        return nothing
    end
    Lmin = max(m + 10, min(B, Lmax))
    Ls = unique(round.(Int, exp.(range(log(Lmin), log(Lmax); length = nwindows))))
    @printf("  %d fixed test blocks starting at %s\n", length(test_starts),
            length(test_starts) <= 6 ? string(test_starts) : "$(test_starts[1]) … $(test_starts[end])")
    @printf("  training windows: %s\n", join(Ls, ", "))

    # The penalty is a fixed additive term while the data term grows with L, so a constant lambda
    # shrinks progressively less as the training window lengthens. `:fixed` is what the pipeline
    # actually does; `:scaled` holds the relative shrinkage constant by taking lambda proportional
    # to L. Reporting both separates a genuine data effect from a moving effective prior.
    results = Dict{Any,Any}()
    for lam in lambdas, mode in (lam == 0 ? (:fixed,) : (:fixed, :scaled))
        println("\n", "-"^86)
        @printf("learning curve, M0, lambda = %g (%s)\n", lam, mode)
        @printf("%8s %6s %12s %10s %22s %10s\n",
                "L", "folds", "NLL", "HAC se", "block-boot 95% CI", "RMSE")
        rows = NamedTuple[]
        for L in Ls
            nll_all = Float64[]
            rmse_all = Float64[]
            nfold = 0
            for ts in test_starts
                trend = ts - emb - 1
                trstart = trend - L + 1
                trstart < 1 && continue
                lam_eff = mode === :scaled ? lam * L / Ls[1] : lam
                nl, rm = fold_scores(X, Y, trstart:trend, ts:(ts + B - 1); lambda = lam_eff)
                append!(nll_all, nl)
                append!(rmse_all, rm)
                nfold += 1
            end
            nfold == 0 && continue
            bl = max(1, ceil(Int, tmax / dt))
            se = hac_se(nll_all; bandwidth = bl)
            lo, hi = block_bootstrap_ci(nll_all, bl)
            @printf("%8d %6d %12.5f %10.5f   [%8.5f, %8.5f] %10.5f\n",
                    L, nfold, mean(nll_all), se, lo, hi, mean(rmse_all))
            push!(rows, (; L, nfold, nll = mean(nll_all), se, lo, hi, rmse = mean(rmse_all),
                         nsamp = length(nll_all)))
        end
        results[(lam, mode)] = rows
        saturation_verdict(rows)
    end

    mkpath(outdir)
    out = joinpath(outdir, "g0a_$(testbed).jld2")
    jldsave(out; tau_int, block = B, embargo = emb, windows = Ls,
            curves = Dict(string(k) => v for (k, v) in results), dt, testbed = string(testbed))
    println("\nwrote ", out)
    return results
end

"""
    saturation_verdict(rows)

Has the curve flattened? Compare the improvement over the last doubling of the training window
against the HAC standard error there. This is the number §10 step 1 branches on.
"""
function saturation_verdict(rows)
    length(rows) < 2 && return
    a, b = rows[end - 1], rows[end]
    d = a.nll - b.nll
    println()
    @printf("  last window step: L %d -> %d, NLL %.5f -> %.5f, improvement %+.5f, HAC se %.5f\n",
            a.L, b.L, a.nll, b.nll, d, b.se)
    if abs(d) <= b.se
        println("  VERDICT: improvement is inside one HAC standard error -- the curve has SATURATED",
                "\n           on this record. Capacity above M0 is not supported by these data.")
    elseif d > 0
        println("  VERDICT: still improving with more data -- NOT saturated.")
    else
        println("  VERDICT: score worsens with more data; check for non-stationarity across the record.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    file = ARGS[1]
    tb = length(ARGS) > 1 ? Symbol(ARGS[2]) : :hit
    if tb == :channel
        g0a(file; testbed = tb, dt = 0.005,
            labels = ["Z[0,3]", "E[0,3]", "Z[4,10]", "E[4,10]", "Z[11,17]", "E[11,17]"])
    else
        g0a(file; testbed = tb, dt = 0.0025,
            labels = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"])
    end
end
