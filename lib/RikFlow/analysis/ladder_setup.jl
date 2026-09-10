# Shared setup for the three offline ladder drivers.
#
# `fit_offline.jl`, `plot_models.jl` and `check_rollout.jl` each built the same object out of the
# same forty lines: load the QoI cache, standardise on the training window, build the regressor,
# split it in time, and hand a rollout its initial history. Three copies of an index convention is
# three chances to get one of them wrong, and the rollout handoff in particular is four
# reverse-ordered slices whose alignment is not obvious by inspection.
#
# 🔴 **The partition, which is the reason this file carries a warning and not just a function.**
#
# There are **three** different train/test splits in this directory and they are not comparable:
#
#  1. `train_frac = 0.75` of whatever record is passed -- this file, and therefore the three ladder
#     drivers. A convenience split for *exploring* the ladder offline.
#  2. Paper 2's own `train_range = (400, 4000)`, i.e. **1.00-10.00 TU** train and 10.00-100.00 TU
#     held out -- `score_m0_ddn.jl`, and every number in `analysis/results.md`. This is the
#     project's partition (`meta_files/plan.md` section 7: no time unit ever serves two roles).
#  3. Fixed test blocks with an embargo and a sweep of training-window lengths --
#     `g0a_learning_curve.jl`, which is measuring the effect of training-set *size* and needs the
#     test position held still while `L` varies.
#
# ⚠️ So a held-out NLL, RMSE, KS or spectral radius produced through this file is **not** comparable
# with the same quantity in `results.md`: different training window, different verification window,
# and a scaling fitted on a different span. Nothing was ever wrong with either split; what was
# missing was a place that says which one is in force. Report numbers from here as exploratory, and
# never paste one into a table beside a `results.md` figure.
#
# Include this file **after** the `ts_*` sources and `extract_qois.jl`; it uses `load_qois`,
# `HistorySpec` and `build_history` and deliberately includes nothing itself, so a caller keeps
# control of which layer it loaded.

"The exploratory split these three drivers use. See the warning above before quoting a number."
const LADDER_TRAIN_FRAC = 0.75

"""
    band_labels(bands)

QoI labels from a list of shell bands, alternating enstrophy and energy per band, which is the
order `RikFlow.jl:185-193` builds them in. HIT/Taylor-Green use `[(0,6),(7,15),(16,32)]` and the
channel `[(0,3),(4,10),(11,17)]`.
"""
band_labels(bands) = collect(Iterators.flatten([("Z[$a,$b]", "E[$a,$b]") for (a, b) in bands]))

"""
    ladder_setup(file; h, train_frac, dt, bands, hist_var, include_predictor, verbose)

Everything the offline ladder drivers need from a tracked record, built once.

`file` may be a raw tracking file or an already-extracted `*_qois.jld2`; `load_qois` accepts both
and caches the first into the second, so nothing here ever materialises the ~1.3 GB of velocity
history that `data_track` carries alongside its QoIs.

Returns a named tuple with

| field | what |
|---|---|
| `raw` | the QoI cache: `q` (`N_Q x T+1`), `q_star`, `dQ`, `key` |
| `qs`, `qss` | `q` and `q_star` standardised on the **training window only** |
| `scaling` | the `(; mu, sigma)` those were standardised with, as `fit_joint` wants it |
| `spec`, `X`, `Y`, `steps` | the regressor and the record step each of its rows predicts |
| `tr`, `ho` | training and held-out **row** ranges of `X`, contiguous in time |
| `ntr`, `ntrain` | the split in record steps, and in rows |
| `block`, `labels` | the single-block id vector, and the QoI labels |

⚠️ `scaling` is fitted on the training window alone, on purpose: fitting it on the whole record
leaks the held-out mean and variance into the model's input transform, which flatters every cell in
the ladder equally and so silently shrinks the differences the ladder exists to measure.
"""
function ladder_setup(file; h::Int = 5, train_frac = LADDER_TRAIN_FRAC, dt = 0.0025,
                      bands = [(0, 6), (7, 15), (16, 32)], hist_var::Symbol = :q_star_q,
                      include_predictor::Bool = true, verbose::Bool = true)
    raw = load_qois(file)
    nq, T = size(raw.q_star)
    labels = band_labels(bands)
    length(labels) == nq ||
        error("ladder_setup: $(length(labels)) labels from $bands against $nq QoIs in the record")

    ntr = floor(Int, train_frac * T)
    ntr > 1 || error("ladder_setup: training window is $ntr steps")
    mu = mean(raw.q[:, 1:ntr]; dims = 2)
    sigma = std(raw.q[:, 1:ntr]; dims = 2)
    scaling = (; mu = vec(mu), sigma = vec(sigma))
    qs = (raw.q .- mu) ./ sigma
    qss = (raw.q_star .- mu) ./ sigma

    spec = HistorySpec(; h, n_qoi = nq, hist_var, include_predictor)
    X, Y, steps = build_history(spec, qss, qs)
    ntrain = count(<=(ntr), steps)
    tr = 1:ntrain
    ho = (ntrain + 1):length(steps)
    block = ones(Int, size(X, 1))

    ld = (; raw, nq, T, dt, file, bands, labels, train_frac, ntr, scaling, qs, qss,
          spec, X, Y, steps, ntrain, tr, ho, block)
    verbose && report_ladder_setup(ld)
    return ld
end

"""
    report_ladder_setup(ld; io = stdout)

Print the record, the regressor and -- named, not implied -- which partition is in force.
"""
function report_ladder_setup(ld; io = stdout)
    @printf(io, "data %s (key %s): N_Q = %d, %d steps, dt = %g, %.1f TU\n",
            basename(ld.file), ld.raw.key, ld.nq, ld.T, ld.dt, ld.T * ld.dt)
    @printf(io, "  q columns = %d (initial-state offset: %s)\n", size(ld.raw.q, 2),
            size(ld.raw.q, 2) == ld.T + 1 ? "ok" : "MISMATCH")
    @printf(io, "  regressor %d rows x %d cols (nfeatures = %d), h = %d, hist_var = %s\n",
            size(ld.X, 1), size(ld.X, 2), nfeatures(ld.spec), ld.spec.h, ld.spec.hist_var)
    # One literal format string: `@printf` will not take a concatenation.
    @printf(io, "  partition: train_frac = %.2f -> steps 1:%d (%.2f-%.2f TU), held out %d:%d (%.2f-%.2f TU)\n",
            ld.train_frac, ld.ntr, ld.dt, ld.ntr * ld.dt,
            ld.ntr + 1, ld.T, (ld.ntr + 1) * ld.dt, ld.T * ld.dt)
    @printf(io, "  rows: train %d, held out %d\n", length(ld.tr), length(ld.ho))
    println(io, "  ⚠️  exploratory split, NOT results.md's. results.md uses paper 2's " *
                "train_range = (400, 4000),\n      i.e. 1-10 TU train and 10-100 TU held out; " *
                "numbers from here are not comparable to it.")
    return nothing
end

"""
    rollout_handoff(ld)

The four slices a free-running rollout needs at the train/held-out boundary, plus the reference it
is scored against. Returns `(; q0, qs0, qstar_series, ref, nroll, first_ho)`.

🔑 The reverse-ordered slices are the part worth having in one place. `rollout` and `HistoryBuffer`
want the history **most recent first**, and the two streams sit one step apart: at the first
held-out step `n`, the corrected history is `q[:, n], q[:, n-1], ...` while the predictor history is
`q_star[:, n-1], q_star[:, n-2], ...` -- because `q_star[:, m]` is the one-step prediction *made
from* step `m`, so the predictor contemporaneous with `q[:, n]` is `q_star[:, n-1]`
(`claude_memory.md` gotcha #13). Getting that offset wrong shifts every model's input by one step
and shows up as a plausible-looking loss of skill rather than as an error.

`ref` is the level over the same window, one step ahead of `first_ho` because a rollout's first
output is the *prediction for* `first_ho + 1`.
"""
function rollout_handoff(ld)
    h = ld.spec.h
    nroll = length(ld.ho)
    nroll > 0 || error("rollout_handoff: the held-out window is empty")
    first_ho = ld.steps[ld.ntrain + 1]
    first_ho - h >= 1 ||
        error("rollout_handoff: need $h steps of history before step $first_ho")
    first_ho + nroll <= size(ld.raw.q, 2) ||
        error("rollout_handoff: $nroll steps from $first_ho run past the record")
    q0 = ld.raw.q[:, first_ho:-1:(first_ho - h + 1)]
    qs0 = ld.raw.q_star[:, (first_ho - 1):-1:(first_ho - h)]
    qstar_series = ld.raw.q_star[:, first_ho:(first_ho + nroll - 1)]
    ref = ld.raw.q[:, (first_ho + 1):(first_ho + nroll)]
    return (; q0, qs0, qstar_series, ref, nroll, first_ho)
end
