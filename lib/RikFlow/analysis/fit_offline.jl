# Offline driver: fit the first rungs of the ladder to a tracked run, inspect the residual, then
# free-run each model without the solver and compare trajectory statistics.
#
# Usage:
#   julia --project=<env> lib/RikFlow/analysis/fit_offline.jl <tracking_file.jld2>
#
# The modules below are deliberately dependency-light (stdlib only) so they can be developed and
# run without loading IncompressibleNavierStokes, CUDA or Lux.
#
# 🔴 **Partition: `train_frac = 0.75` of the record passed, NOT results.md's.** Every number this
# script prints comes off that exploratory split. `analysis/results.md` and `score_m0_ddn.jl` use
# paper 2's own `train_range = (400, 4000)` -- 1-10 TU train, 10-100 TU held out -- and fit the
# scaling on a different span as well. So a held-out NLL, KS or trajectory statistic from here is
# **not** comparable with the same quantity there and must never be pasted into a table beside
# one. The three-way comparison of the splits in this directory is in `ladder_setup.jl`'s header.
#
# 🔴 **Scope: this is the only driver that fits the L2 and L7 cells** -- M0v (state-dependent
# Sigma), M0c1/M0c2 (AR residual) and M2c1 (both). Those cells are gated on G0-d, which has never
# been run on real data, so this script is parked rather than retired.

using LinearAlgebra
using Statistics
using Random
using Printf
using JLD2

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "ts_history.jl"))
include(joinpath(SRC, "ts_models.jl"))
include(joinpath(SRC, "ts_fit.jl"))
include(joinpath(SRC, "ts_score.jl"))
include(joinpath(SRC, "ts_rollout.jl"))

# `load_qois`, then the shared setup that owns the partition and the rollout handoff. Order
# matters: `ladder_setup.jl` uses `load_qois`, `HistorySpec` and `build_history`, and includes
# nothing itself.
include(joinpath(@__DIR__, "extract_qois.jl"))
include(joinpath(@__DIR__, "ladder_setup.jl"))

# QoI band labels, set from the setup in `main` so the reporting helpers below can stay
# argument-free. `band_labels` itself lives in `ladder_setup.jl`.
const LABELS = Ref(["Z1", "E1", "Z2", "E2", "Z3", "E3"])
qoi_labels() = LABELS[]

# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

function main(file; h = 5, lambda = 0.01, dt = 0.0025, train_frac = 0.75, seed = 20260904,
              bands = [(0, 6), (7, 15), (16, 32)])
    Random.seed!(seed)
    ld = ladder_setup(file; h, train_frac, dt, bands)
    LABELS[] = ld.labels
    (; nq, scaling, spec, X, Y, tr, ho, block) = ld

    configs = [
        (name = "M0   linear mean, white, constant Sigma", ar = 0, sd = false),
        (name = "M0v  linear mean, white, state-dep Sigma", ar = 0, sd = true),
        (name = "M0c1 linear mean, AR(1), constant Sigma", ar = 1, sd = false),
        (name = "M0c2 linear mean, AR(2), constant Sigma", ar = 2, sd = false),
        (name = "M2c1 linear mean, AR(1), state-dep Sigma", ar = 1, sd = true),
    ]

    results = []
    for cfg in configs
        println("\n", "="^78, "\n", cfg.name, "\n", "="^78)
        model, hist = fit_joint(X[tr, :], Y[tr, :], spec; ar_order = cfg.ar,
                                state_dependent = cfg.sd, lambda, iters = 40,
                                block_id = block[tr], scaling, verbose = false)

        mono = all(diff(hist.nll) .<= 1e-6 * abs.(hist.nll[1:(end - 1)]) .+ 1e-9)
        @printf("  fit: nll %.4f -> %.4f over %d iters (monotone: %s)\n",
                hist.nll[1], hist.nll[end], length(hist.nll) - 1, mono ? "yes" : "NO")

        # held-out score
        a = ar_coefficients(model)
        rows_ho = valid_rows(block[ho], cfg.ar)
        nll_ho = nll(X[ho, :], Y[ho, :], model.C, model.W, model.R, a, block[ho], rows_ho;
                     uclip = model.uclip)
        per = gaussian_nll_per_sample(nll_ho, length(rows_ho), nq)
        @printf("  held-out NLL per sample per QoI: %.5f\n", per)

        # residual diagnostics on the training window
        rows_tr = valid_rows(block[tr], cfg.ar)
        P = nll_parts(X[tr, :], Y[tr, :], model.C, model.W, a, block[tr]; uclip = model.uclip)
        bat = residual_battery(P.Res[rows_tr, :], P.Z[rows_tr, :]; dt, nlags = 60)
        report_residuals(bat, a, dt, cfg.ar)

        push!(results, (; cfg, model, per, bat, a))
    end

    # ------------------------------------------------------------------------------------------
    # free-running rollout, no solver
    # ------------------------------------------------------------------------------------------
    println("\n", "="^78, "\n free-running rollout (predictor stream replayed, no solver)\n", "="^78)
    (; q0, qs0, qstar_series, ref, nroll) = rollout_handoff(ld)

    refstats = trajectory_stats(ref; dt)
    @printf("\n%-38s %8s %8s %8s %8s\n", "", "mean", "std", "rho1", "T_int")
    print_stats("reference (tracked)", refstats, 1:nq)

    for r in results
        traj, _ = rollout(r.model, q0, qs0; nsteps = nroll, q_star_series = qstar_series,
                          rng = MersenneTwister(seed), spinup = 0)
        st = trajectory_stats(traj; dt)
        blew = any(!isfinite, traj) || maximum(abs, traj) > 1e6
        println()
        print_stats(r.cfg.name[1:min(end, 38)] * (blew ? "  [DIVERGED]" : ""), st, 1:nq)
        ks = [ks_distance(vec(traj[i, :]), vec(ref[i, :])) for i in 1:nq]
        @printf("%-38s KS per QoI: %s   sum = %.3f\n", "", join((@sprintf("%.3f", k) for k in ks), " "), sum(ks))
    end
end

function print_stats(label, st, idx)
    for i in idx
        @printf("%-38s %8.3f %8.3f %8.4f %8.4f   %s\n",
                i == first(idx) ? label : "", st.mean[i], st.std[i], st.rho1[i], st.tint[i],
                qoi_labels()[i])
    end
end

function report_residuals(bat, a, dt, p)
    lab = qoi_labels()
    @printf("  residual battery (95%% band = %.4f)\n", bat.band)
    @printf("    %-10s %9s %9s %9s %9s %8s %8s %9s\n",
            "QoI", "phi_ee(1)", "phi_e2(1)", "phi_zz(1)", "phi_z2(1)", "skew", "exkurt", "T_exp/dt")
    for i in 1:length(bat.phi_ee)
        @printf("    %-10s %9.4f %9.4f %9.4f %9.4f %8.3f %8.3f %9.2f\n",
                lab[i], bat.phi_ee[i][2], bat.phi_e2e2[i][2], bat.phi_zz[i][2], bat.phi_z2z2[i][2],
                bat.skew[i], bat.kurt[i], bat.tcorr[i].T_exp / dt)
    end
    if p > 0
        @printf("    fitted AR coefficients and implied T^eta:\n")
        for i in 1:size(a, 2)
            Te, om = decorrelation_time(view(a, :, i), dt)
            @printf("      %-10s a = %s   T^eta = %.4g  (%.1f dt)   omega = %.4g\n",
                    lab[i], join((@sprintf("%+.4f", a[k, i]) for k in 1:size(a, 1)), " "),
                    Te, Te / dt, om)
        end
    end
    qmax = maximum(c.lb.Q for c in bat.cross)
    @printf("    constant-R check: max Ljung-Box on standardised cross-products = %.1f (dof %d)\n",
            qmax, bat.cross[1].lb.dof)
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: fit_offline.jl <tracking_file.jld2> [lambda] [testbed]")
    lam = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 0.01
    tb = length(ARGS) > 2 ? Symbol(ARGS[3]) : :hit
    if tb == :channel
        main(ARGS[1]; lambda = lam, dt = 0.005, bands = [(0, 3), (4, 10), (11, 17)])
    else
        main(ARGS[1]; lambda = lam, dt = 0.0025, bands = [(0, 6), (7, 15), (16, 32)])
    end
end
