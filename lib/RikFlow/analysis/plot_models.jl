# Trajectory plots for the fitted ladder, on one testbed, offline.
#
# Two questions, two kinds of figure.
#
#   (1) One-step prediction on held-out data. The regressor rows are built from the tracked `q` and
#       `q_star`, so every model is teacher-forced: it sees the true history and predicts one step.
#       All the models should sit on top of the reference here, and the informative quantity is the
#       error, not the trajectory. Figures: `onestep_traj`, `onestep_error`, `onestep_rmse`.
#
#   (2) Free-running rollout, no solver. The model advances its own corrected-QoI history; only the
#       predictor stream `q_star` is replayed. This is where a model that predicts well one step at
#       a time can still be useless, and it is the closest offline proxy for deployment.
#       Figures: `rollout_grid`, `rollout_pdf`, `stability`.
#
# Stability is not a mystery to be discovered by running: the free-running mean map is linear, so
# its companion matrix has a spectral radius that says in advance whether the rollout diverges.
# `stability` sweeps that against lambda; `spectral_radius` is printed with every fit.
#
# 🔴 **Partition: `train_frac = 0.75` of the record passed, NOT results.md's.** Every number and
# every figure here comes off that exploratory split; `analysis/results.md` and `score_m0_ddn.jl`
# use paper 2's own `train_range = (400, 4000)` -- 1-10 TU train, 10-100 TU held out -- and fit the
# scaling on a different span as well. So an RMSE, KS, spectral radius or rollout statistic from
# here is **not** comparable with the same quantity there. The three-way comparison of the splits
# in this directory is in `ladder_setup.jl`'s header.
#
# 🔴 **Scope: the nine configurations below include the L2 and L7 cells**, which are gated on G0-d
# and have never been run on real data. Parked, not retired.
#
# Usage:
#   julia --project=<env> lib/RikFlow/analysis/plot_models.jl <tracking.jld2> [lambda] [hit|channel]

using LinearAlgebra, Statistics, Random, Printf, JLD2, CairoMakie

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "ts_history.jl"))
include(joinpath(SRC, "ts_models.jl"))
include(joinpath(SRC, "ts_fit.jl"))
include(joinpath(SRC, "ts_score.jl"))
include(joinpath(SRC, "ts_rollout.jl"))

# QoI arrays come from the extracted cache, never from the tracking file directly. See
# `extract_qois.jl` for why: `data_track` is one compound dataset whose `fields` member is ~1.3 GB
# of velocity history that nothing here reads.
include(joinpath(@__DIR__, "extract_qois.jl"))

# `ladder_setup.jl` owns the partition, the standardisation and the rollout handoff that this file,
# `fit_offline.jl` and `check_rollout.jl` used to keep three copies of. It uses `load_qois`,
# `HistorySpec` and `build_history`, so it is included after them and includes nothing itself.
include(joinpath(@__DIR__, "ladder_setup.jl"))

"""
    thin(n, target)

Stride that keeps at most `target` points out of `n`. Rasterising 54 panels of 4000 points each is
several hundred thousand line segments per figure, well past what the output resolution can show.
"""
thin(n::Int, target::Int = 1500) = max(1, cld(n, target))


# ---------------------------------------------------------------------------------------------
# the ladder, as nine configurations of one estimator
# ---------------------------------------------------------------------------------------------
#
# `mean = :qstar` freezes the mean map at paper 1's: `q^n = q^{n*} + const`, the identity on the
# predictor block plus a bias, nothing fitted. That is what makes "coloured noise only" and
# "state-dependent noise only" ablations of the *noise* rather than of the mean.

const CONFIGS = [
    (key = "DD",      label = "data-driven noise",         mean = :qstar,  ar = 0, sd = false, noise = true),
    (key = "LR",      label = "LinReg (deterministic)",    mean = :linreg, ar = 0, sd = false, noise = false),
    (key = "LR+MVG",  label = "LinReg + MVG noise",        mean = :linreg, ar = 0, sd = false, noise = true),
    (key = "LR+C",    label = "LinReg + coloured",         mean = :linreg, ar = 1, sd = false, noise = true),
    (key = "LR+V",    label = "LinReg + state-dep",        mean = :linreg, ar = 0, sd = true,  noise = true),
    (key = "LR+VC",   label = "LinReg + state-dep col.",   mean = :linreg, ar = 1, sd = true,  noise = true),
    (key = "DD+C",    label = "coloured only",             mean = :qstar,  ar = 1, sd = false, noise = true),
    (key = "DD+V",    label = "state-dep only",            mean = :qstar,  ar = 0, sd = true,  noise = true),
    (key = "DD+VC",   label = "state-dep + coloured only", mean = :qstar,  ar = 1, sd = true,  noise = true),
]

# Nine distinguishable hues, none of them grey: grey is reserved for the tracked reference, which
# every rollout panel draws underneath the model.
const COLORS = ["#1f77b4", "#8c564b", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#17becf", "#e377c2", "#bcbd22"]

# Which family a configuration belongs to. The two families differ in one-step error by an order of
# magnitude, so plotting them on one axis hides everything inside the smaller one.
family(cfg) = cfg.mean === :linreg ? 1 : 2
const FAMILY_NAME = ["LinReg mean", "data-driven mean (q* + const)"]

"""
    qstar_mean_C(spec, X, Y)

The frozen mean map of paper 1's data-driven noise model, written in the regressor's basis:
identity on the current predictor block, a bias equal to the mean correction, zero everywhere else.
Because `q` and `q_star` are standardised with the same `(mu, sigma)`, the identity survives
standardisation unchanged and the bias is `mean(dQ)/sigma`.
"""
function qstar_mean_C(spec::HistorySpec, X::AbstractMatrix{T}, Y) where {T}
    cols = qstar_columns(spec)
    cols === nothing && error("mean = :qstar needs include_predictor = true")
    C = zeros(T, nfeatures(spec), spec.n_qoi)
    C[cols, :] .= Matrix{T}(I, spec.n_qoi, spec.n_qoi)
    C[bias_column(spec), :] .= vec(mean(Y .- view(X, :, cols); dims = 1))
    return C
end

function fit_all(X, Y, spec, tr, block; lambda, scaling, iters = 40)
    fits = NamedTuple[]
    Cq = qstar_mean_C(spec, X[tr, :], Y[tr, :])
    println("\n", "="^92)
    @printf("%-24s %8s %10s %10s %9s %9s\n", "model", "params", "train NLL", "held NLL",
            "rho mean", "rho AR")
    println("="^92)
    for cfg in CONFIGS
        fC = cfg.mean === :qstar ? Cq : nothing
        model, hist = fit_joint(X[tr, :], Y[tr, :], spec; ar_order = cfg.ar,
                                state_dependent = cfg.sd, lambda, iters,
                                block_id = block[tr], scaling, fixed_C = fC)
        @printf("  fitted %-8s (%d sweeps)\n", cfg.key, length(hist.nll) - 1); flush(stdout)
        push!(fits, (; cfg, model, hist))
    end
    return fits
end

"""
    onestep(model, X, Y, block)

Teacher-forced one-step prediction on the given rows: the conditional mean `mu_eff`, which for a
coloured model uses the *true* past residuals. Returns `(M, Res, rows)`.
"""
function onestep(model, X, Y, block)
    p = ar_order(model)
    a = ar_coefficients(model)
    P = nll_parts(X, Y, model.C, model.W, a, block; uclip = model.uclip)
    return P.M, P.Res, valid_rows(block, p)
end

# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

function fig_onestep(fits, X, Y, block, labs, outdir; window = 1000, show = ["LR+MVG", "LR+C", "DD"])
    nq = size(Y, 2)
    preds = Dict(f.cfg.key => onestep(f.model, X, Y, block) for f in fits)
    w = 1:thin(min(window, size(Y, 1))):min(window, size(Y, 1))

    # (a) reference against a few predictions -- the point is that they are indistinguishable
    fig = Figure(size = (1100, 160 * nq + 60))
    for i in 1:nq
        ax = Axis(fig[i + 1, 1]; ylabel = labs[i], xlabel = i == nq ? "held-out step" : "")
        lines!(ax, w, Y[w, i]; color = :black, linewidth = 2.2, label = "reference")
        for k in show
            j = findfirst(c -> c.key == k, CONFIGS)
            M, _, _ = preds[k]
            lines!(ax, w, M[w, i]; color = COLORS[j], linewidth = 0.9, label = k)
        end
        i < nq && hidexdecorations!(ax; grid = false)
        i == 1 && axislegend(ax; position = :rt, nbanks = 4, framevisible = false, labelsize = 9)
    end
    Label(fig[1, 1], "one-step prediction on held-out data (teacher forced)";
          fontsize = 14, tellwidth = false)
    save(joinpath(outdir, "onestep_traj.png"), fig; px_per_unit = 2)

    # (b) the errors. Split by family: the data-driven mean is an order of magnitude worse, and on
    # one shared axis it would flatten the LinReg family into the zero line.
    fig = Figure(size = (1500, 150 * nq + 90))
    for i in 1:nq, fam in 1:2
        ax = Axis(fig[i + 1, fam]; ylabel = fam == 1 ? labs[i] : "",
                  xlabel = i == nq ? "held-out step" : "",
                  title = i == 1 ? FAMILY_NAME[fam] : "", titlesize = 12)
        for (j, f) in enumerate(fits)
            family(f.cfg) == fam || continue
            _, Res, _ = preds[f.cfg.key]
            lines!(ax, w, Res[w, i]; color = (COLORS[j], 0.85), linewidth = 0.7, label = f.cfg.key)
        end
        hlines!(ax, [0.0]; color = :black, linewidth = 0.5)
        i < nq && hidexdecorations!(ax; grid = false)
        i == 1 && axislegend(ax; position = :rt, nbanks = 3, framevisible = false, labelsize = 9)
    end
    Label(fig[1, 1:2], "one-step error, standardised units"; fontsize = 14, tellwidth = false)
    save(joinpath(outdir, "onestep_error.png"), fig; px_per_unit = 2)

    # (c) the same information as one number per model per QoI
    rmse = zeros(length(fits), nq)
    xs = Int[]; ys = Float64[]; dg = Int[]
    for (j, f) in enumerate(fits), i in 1:nq
        _, Res, rows = preds[f.cfg.key]
        rmse[j, i] = sqrt(mean(abs2, view(Res, rows, i)))
        push!(xs, i); push!(ys, rmse[j, i]); push!(dg, j)
    end
    fig = Figure(size = (1200, 440))
    ax = Axis(fig[1, 1]; xticks = (1:nq, labs), yscale = log10,
              ylabel = "held-out one-step RMSE (standardised)",
              title = "one-step error by model (log scale)")
    barplot!(ax, xs, ys; dodge = dg, color = [COLORS[d] for d in dg])
    ylims!(ax, 0.5 * minimum(ys), 2 * maximum(ys))
    Legend(fig[1, 2], [PolyElement(color = COLORS[j]) for j in 1:length(fits)],
           [f.cfg.key for f in fits]; framevisible = false)
    save(joinpath(outdir, "onestep_rmse.png"), fig; px_per_unit = 2)
    return rmse
end

function fig_rollout(fits, trajs, ref, labs, dt, outdir; window = 4000)
    nq = size(ref, 1)
    nf = length(fits)
    w = 1:thin(min(window, size(ref, 2))):min(window, size(ref, 2))
    t = (w .- 1) .* dt

    fig = Figure(size = (240 * nf, 150 * nq))
    for (j, f) in enumerate(fits)
        tr = trajs[f.cfg.key]
        bad = any(!isfinite, tr) || maximum(abs, tr) > 1e6
        for i in 1:nq
            ax = Axis(fig[i, j];
                      title = i == 1 ? f.cfg.key * (bad ? "  DIVERGED" : "") : "",
                      titlecolor = bad ? :red : :black, titlesize = 11,
                      ylabel = j == 1 ? labs[i] : "", xlabel = i == nq ? "t [TU]" : "")
            lines!(ax, t, ref[i, w]; color = (:black, 0.7), linewidth = 1.2)
            y = tr[i, w]
            if bad     # keep the panel readable; the divergence is reported in the title
                lo, hi = extrema(ref[i, :])
                pad = 5 * (hi - lo)
                y = clamp.(y, lo - pad, hi + pad)
            end
            lines!(ax, t, y; color = COLORS[j], linewidth = 0.8)
            j > 1 && hideydecorations!(ax; grid = false)
            i < nq && hidexdecorations!(ax; grid = false)
        end
    end
    Label(fig[0, 1:nf], "free-running rollout, predictor stream replayed, no solver (grey = tracked)";
          fontsize = 14, tellwidth = false)
    save(joinpath(outdir, "rollout_grid.png"), fig; px_per_unit = 2)
end

"""
    density_line(x, edges)

Histogram density on fixed edges, returned as `(centres, density)`. A hand-rolled histogram keeps
this script free of a KernelDensity dependency and makes the bin width explicit.
"""
function density_line(x, edges)
    n = length(edges) - 1
    c = zeros(Float64, n)
    w = edges[2] - edges[1]
    for v in x
        k = floor(Int, (v - edges[1]) / w) + 1
        1 <= k <= n && (c[k] += 1)
    end
    return (edges[1:n] .+ w / 2), c ./ (sum(c) * w + eps())
end

function fig_pdf(fits, trajs, ref, labs, outdir)
    nq = size(ref, 1)
    fig = Figure(size = (1200, 230 * cld(nq, 2)))
    for i in 1:nq
        r, c = fldmod1(i, 2)
        ax = Axis(fig[r, c]; title = labs[i], xlabel = "value", ylabel = "density")
        lo, hi = extrema(ref[i, :])
        pad = 0.5 * (hi - lo)
        edges = range(lo - pad, hi + pad; length = 81)
        x, d = density_line(ref[i, :], edges)
        lines!(ax, x, d; color = :black, linewidth = 2.5, label = "reference")
        for (j, f) in enumerate(fits)
            tr = trajs[f.cfg.key]
            all(isfinite, view(tr, i, :)) || continue
            x, d = density_line(view(tr, i, :), edges)
            lines!(ax, x, d; color = COLORS[j], linewidth = 1.2, label = f.cfg.key)
        end
        i == 1 && axislegend(ax; position = :rt, framevisible = false, labelsize = 9, nbanks = 2)
    end
    Label(fig[0, 1:2], "rollout marginal distributions (values outside the reference range are cut)";
          fontsize = 14, tellwidth = false)
    save(joinpath(outdir, "rollout_pdf.png"), fig; px_per_unit = 2)
end

"""
    fig_correction(fits, trajs, ref, qstar, labs, dt, outdir)

The rollout in terms of the correction `dQ = q - q*` rather than the level `q`.

This is the discriminating view. The predictor stream is replayed, and `q` is `q*` plus a small
correction, so every model's *level* trajectory is pinned to the reference whether the model is any
good or not. The correction is the part the model is actually responsible for, and it is where the
nine configurations separate.
"""
function fig_correction(fits, trajs, ref, qstar, labs, dt, outdir; window = 4000)
    nq = size(ref, 1)
    nf = length(fits)
    w = 1:thin(min(window, size(ref, 2))):min(window, size(ref, 2))
    t = (w .- 1) .* dt
    dref = ref .- qstar

    fig = Figure(size = (240 * nf, 150 * nq + 40))
    for (j, f) in enumerate(fits)
        d = trajs[f.cfg.key] .- qstar
        ok = all(isfinite, d) && maximum(abs, d) < 1e6
        for i in 1:nq
            ax = Axis(fig[i + 1, j];
                      title = i == 1 ? f.cfg.key * (ok ? "" : "  DIVERGED") : "",
                      titlecolor = ok ? :black : :red, titlesize = 11,
                      ylabel = j == 1 ? labs[i] : "", xlabel = i == nq ? "t [TU]" : "")
            lines!(ax, t, dref[i, w]; color = (:black, 0.7), linewidth = 1.2)
            ok && lines!(ax, t, d[i, w]; color = (COLORS[j], 0.9), linewidth = 0.8)
            ylims!(ax, minimum(dref[i, :]) - 0.5 * std(dref[i, :]),
                   maximum(dref[i, :]) + 0.5 * std(dref[i, :]))
            j > 1 && hideydecorations!(ax; grid = false)
            i < nq && hidexdecorations!(ax; grid = false)
        end
    end
    Label(fig[1, 1:nf], "rollout correction dQ = q - q* (black = tracked); y-limits set by the reference";
          fontsize = 14, tellwidth = false)
    save(joinpath(outdir, "rollout_correction.png"), fig; px_per_unit = 2)
end

"""
    fig_rollout_error(fits, trajs, ref, labs, dt, outdir)

Rollout minus reference, in units of each QoI's reference standard deviation. A free run is not
expected to track the reference path -- it is a different realisation of the noise -- so this is
read for drift and for growth, not for smallness.
"""
function fig_rollout_error(fits, trajs, ref, labs, dt, outdir; window = 4000)
    nq = size(ref, 1)
    w = 1:thin(min(window, size(ref, 2))):min(window, size(ref, 2))
    t = (w .- 1) .* dt
    sd = [std(view(ref, i, :)) for i in 1:nq]

    fig = Figure(size = (1500, 150 * nq + 90))
    for i in 1:nq, fam in 1:2
        ax = Axis(fig[i + 1, fam]; ylabel = fam == 1 ? labs[i] : "",
                  xlabel = i == nq ? "t [TU]" : "",
                  title = i == 1 ? FAMILY_NAME[fam] : "", titlesize = 12)
        drawn = 0
        for (j, f) in enumerate(fits)
            family(f.cfg) == fam || continue
            tr = trajs[f.cfg.key]
            (all(isfinite, tr) && maximum(abs, tr) < 1e6) || continue
            lines!(ax, t, (tr[i, w] .- ref[i, w]) ./ sd[i]; color = (COLORS[j], 0.85),
                   linewidth = 0.8, label = f.cfg.key)
            drawn += 1
        end
        hlines!(ax, [0.0]; color = :black, linewidth = 0.5)
        i < nq && hidexdecorations!(ax; grid = false)
        # A family can be empty when every one of its models diverged; asking for a legend then is
        # an error, not an empty legend.
        i == 1 && drawn > 0 &&
            axislegend(ax; position = :rt, nbanks = 3, framevisible = false, labelsize = 9)
    end
    Label(fig[1, 1:2], "rollout minus reference, in reference standard deviations";
          fontsize = 14, tellwidth = false)
    save(joinpath(outdir, "rollout_error.png"), fig; px_per_unit = 2)
end

"""
    fig_stability(X, Y, spec, tr, block, scaling, outdir; lambdas)

Spectral radius of the free-running companion matrix against `lambda`, for each mean model and AR
order. This is the diagnostic the trial-and-error rollouts were standing in for: below one the
rollout cannot diverge, above one it must, and the shape of the curve says whether the instability
is a shrinkage problem at all.
"""
function fig_stability(X, Y, spec, tr, block, scaling, outdir;
                       lambdas = 10.0 .^ range(-5, 2; length = 9), iters = 10)
    variants = [(k = "LinReg mean, white", mean = :linreg, ar = 0),
                (k = "LinReg mean, AR(1)", mean = :linreg, ar = 1),
                (k = "LinReg mean, AR(2)", mean = :linreg, ar = 2),
                (k = "data-driven mean, AR(1)", mean = :qstar, ar = 1)]
    Cq = qstar_mean_C(spec, X[tr, :], Y[tr, :])
    fig = Figure(size = (1250, 480))
    axm = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "lambda",
               ylabel = "spectral radius", title = "mean map (lagged-QoI feedback in C)")
    axa = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = "lambda",
               title = "residual autoregression (max |root of a_i|)")
    println("\nspectral radius against lambda   (mean block / AR block)")
    for (j, v) in enumerate(variants)
        rm, ra = Float64[], Float64[]
        for lam in lambdas
            # `iters` is deliberately small here. This figure reads only the spectral radius,
            # which is a property of `C`, and `C` is at its exact GLS optimum after the first
            # sweep; the later sweeps refine `psi` and `W`, which move rho very little. Each sweep
            # costs one coupled GLS solve -- 4.6 s at 30000 rows -- so the full iteration count
            # would put this one figure at over an hour on the 100 TU record.
            model, _ = fit_joint(X[tr, :], Y[tr, :], spec; ar_order = v.ar, state_dependent = false,
                                 lambda = lam, iters, block_id = block[tr], scaling,
                                 fixed_C = v.mean === :qstar ? Cq : nothing)
            push!(rm, spectral_radius(model; part = :mean))
            push!(ra, max(spectral_radius(model; part = :ar), 1e-3))
        end
        for (ax, r) in ((axm, rm), (axa, ra))
            lines!(ax, lambdas, r; color = COLORS[j], linewidth = 2, label = v.k)
            scatter!(ax, lambdas, r; color = COLORS[j], markersize = 6)
        end
        @printf("  %-24s mean %s\n  %-24s AR   %s\n", v.k,
                join((@sprintf("%.3f", r) for r in rm), " "), "",
                join((@sprintf("%.3f", r) for r in ra), " "))
    end
    for ax in (axm, axa)
        hlines!(ax, [1.0]; color = :red, linestyle = :dash, linewidth = 1.5)
    end
    Legend(fig[2, 1:2], axm; orientation = :horizontal, framevisible = false)
    Label(fig[0, 1:2],
          "the rollout diverges iff either block leaves the unit circle (dashed red)";
          fontsize = 13, tellwidth = false)
    save(joinpath(outdir, "stability.png"), fig; px_per_unit = 2)
end

# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

function main(file; h = 5, lambda = 0.01, dt = 0.0025, train_frac = 0.75, seed = 20260905,
              bands = [(0, 6), (7, 15), (16, 32)], testbed = "hit",
              outdir = joinpath(@__DIR__, "output", "figs_" * testbed))
    Random.seed!(seed)
    mkpath(outdir)
    ld = ladder_setup(file; h, train_frac, dt, bands)
    (; nq, scaling, spec, X, Y, tr, ho, block) = ld
    labs = ld.labels

    fits = fit_all(X, Y, spec, tr, block; lambda, scaling)
    flush(stdout)

    # scores and the stability number, in one table
    for f in fits
        rows_ho = valid_rows(block[ho], f.cfg.ar)
        nl = nll(X[ho, :], Y[ho, :], f.model.C, f.model.W, f.model.R, ar_coefficients(f.model),
                 block[ho], rows_ho; uclip = f.model.uclip)
        per = gaussian_nll_per_sample(nl, length(rows_ho), nq)
        @printf("%-24s %8d %10.4f %10.4f %9.4f %9.4f\n", f.cfg.label, nparams(f.model),
                f.hist.nll[end] / length(tr) / nq, per,
                spectral_radius(f.model; part = :mean), spectral_radius(f.model; part = :ar))
    end

    println("\none-step figures ...")
    rmse = fig_onestep(fits, X[ho, :], Y[ho, :], block[ho], labs, outdir)
    @printf("\nheld-out one-step RMSE (standardised)\n%-24s", "model")
    for l in labs; @printf(" %9s", l); end
    println()
    for (j, f) in enumerate(fits)
        @printf("%-24s", f.cfg.key)
        for i in 1:nq; @printf(" %9.4f", rmse[j, i]); end
        println()
    end

    println("\nfree-running rollouts ...")
    (; q0, qs0, qstar_series, ref, nroll) = rollout_handoff(ld)

    trajs = Dict{String,Matrix{Float64}}()
    for f in fits
        t, _ = rollout(f.model, q0, qs0; nsteps = nroll, q_star_series = qstar_series,
                       rng = MersenneTwister(seed), spinup = 0, noise = f.cfg.noise)
        trajs[f.cfg.key] = Float64.(t)
    end

    # Statistics are reported twice: on the level `q`, and on the correction `dQ = q - q*`. The
    # level is pinned by the replayed predictor stream and so barely separates the models; the
    # correction is the part the model is responsible for and is where the ladder shows up.
    dref = ref .- qstar_series
    rs = trajectory_stats(ref; dt)
    rd = trajectory_stats(dref; dt)
    @printf("\n%-24s | %-26s | %-34s\n", "", "level q  (QoI 1)", "correction dQ  (mean over QoIs)")
    @printf("%-24s %8s %8s %8s | %8s %8s %8s %8s\n", "model", "std", "rho1", "sumKS",
            "std", "rho1", "T_int", "sumKS")
    @printf("%-24s %8.3f %8.4f %8s | %8.4g %8.4f %8.4f %8s\n", "reference (tracked)",
            rs.std[1], rs.rho1[1], "-", mean(rd.std), mean(rd.rho1), mean(rd.tint), "-")
    for f in fits
        t = trajs[f.cfg.key]
        if !(all(isfinite, t) && maximum(abs, t) < 1e6)
            @printf("%-24s %8s\n", f.cfg.key, "DIVERGED")
            continue
        end
        d = t .- qstar_series
        st, sd = trajectory_stats(t; dt), trajectory_stats(d; dt)
        ks = sum(ks_distance(vec(t[i, :]), vec(ref[i, :])) for i in 1:nq)
        ksd = sum(ks_distance(vec(d[i, :]), vec(dref[i, :])) for i in 1:nq)
        @printf("%-24s %8.3f %8.4f %8.3f | %8.4g %8.4f %8.4f %8.3f\n", f.cfg.key,
                st.std[1], st.rho1[1], ks, mean(sd.std), mean(sd.rho1), mean(sd.tint), ksd)
    end

    fig_rollout(fits, trajs, ref, labs, dt, outdir)
    fig_correction(fits, trajs, ref, qstar_series, labs, dt, outdir)
    fig_rollout_error(fits, trajs, ref, labs, dt, outdir)
    fig_pdf(fits, trajs, ref, labs, outdir)
    println("stability sweep ..."); flush(stdout)
    fig_stability(X, Y, spec, tr, block, scaling, outdir)

    jldsave(joinpath(outdir, "..", "plot_models_$(testbed).jld2");
            keys = [f.cfg.key for f in fits], rmse, dt, lambda, h,
            rho_mean = [spectral_radius(f.model; part = :mean) for f in fits],
            rho_ar = [spectral_radius(f.model; part = :ar) for f in fits])
    println("\nfigures written to ", outdir)
    return fits, trajs, ref
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: plot_models.jl <tracking.jld2> [lambda] [hit|channel]")
    lam = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 0.01
    tb = length(ARGS) > 2 ? ARGS[3] : "hit"
    tag = length(ARGS) > 3 ? ARGS[4] : tb          # separates records of the same testbed
    if startswith(tb, "channel")
        main(ARGS[1]; lambda = lam, dt = 0.005, bands = [(0, 3), (4, 10), (11, 17)], testbed = tag)
    else
        main(ARGS[1]; lambda = lam, dt = 0.0025, bands = [(0, 6), (7, 15), (16, 32)], testbed = tag)
    end
end
