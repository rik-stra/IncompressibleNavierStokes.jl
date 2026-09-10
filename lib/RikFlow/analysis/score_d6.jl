# Score D6: metric #17 lead-resolved spread-skill, and RH-3 lead-resolved rank histograms.
#
# The last two baseline metrics, and the only two D6 exists for. Both are expectations over
# **initial conditions** at a fixed lead, and every archived online run is one trajectory from one
# initial condition (`6_online_TO_LRS.jl:56-59`), so with the archive alone each lead has exactly one
# verification instance and an RMSE from one sample is not an RMSE.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/score_d6.jl            # score what is there
#   julia --startup-file=no --project=analysis analysis/score_d6.jl --preview  # design only, no runs
#
# Writes `analysis/output/d6_scores.jld2` and prints the tables that go into
# `analysis/results.md`. ⚠️ The report belongs in `results.md`, beside the script that made it --
# not in `meta_files/`, and not left in `analysis/output/`, which `.gitignore:12` excludes.
#
# ---------------------------------------------------------------------------------------------
# Three decisions that determine whether the numbers mean anything
# ---------------------------------------------------------------------------------------------
#
# **Scored on the QoI level `q`, with the correction `dQ` reported beside it.** "Score `dQ`, never
# `q`" is regime-B-scoped and applying it here would be an error (`claude_memory.md` gotcha #27).
# D6 is regime C -- free-running, no replayed predictor stream -- and free-running the level is what
# the claims are about: the long-term QoI distribution, the QoI decorrelation time, the QoI spread.
# The choice changes verdicts: LinReg73 reads 0.997 on the correction and **0.809** on the level.
# The correction is reported as secondary because the level's own temporal statistic is null
# (gotcha #28), so a level-only temporal claim would make every configuration look equally good.
#
# **Leads are per QoI and in physical time.** `T_int` spans 0.0082-0.3017 TU across the six bands, a
# factor 36.8 (gotcha #30), so one grid in units of `t_int` cannot serve them all. The grid is
# `{0.25, 0.5, 1, 2, 5, 10} x T_int(i)`; the largest entry, 1207 steps, is what set the 1208-step
# forecast, and short leads are free within a run.
#
# **Truth is the high-fidelity reference, not the tracked record.** The ICs are cut from the tracked
# record's velocity fields, but the tracked run is an LF simulation nudged onto the reference, and
# what a forecast should be scored against is the reference itself. The two agree to 3e-5-2.3e-3 of
# a standard deviation (`results.md` section 1), so this is a small choice -- but it is a choice, and
# `TRUTH_SOURCE` names it.

using LinearAlgebra
using Statistics
using Random
using Printf
using JLD2
using Dates

const HERE = @__DIR__
const SRC = normpath(joinpath(HERE, "..", "src"))
include(joinpath(SRC, "ts_score.jl"))
include(joinpath(HERE, "extract_qois.jl"))
include(joinpath(HERE, "build_d6_ics.jl"))
# `load_ensemble`, for the ordinal-0 validation comparison against the archived LinReg1 runs.
include(joinpath(HERE, "extract_archive.jl"))

const OUT = joinpath(HERE, "output")
const DT = 2.5e-3                      # HIT LES time step, TU
const SEED = 20260909

"QoI band labels. HIT shells are [0,6] [7,15] [16,32]; rows alternate enstrophy / energy."
const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

"""
Index of `Z[16,32]`, named because it is excluded from the validation verdict.

Commit `09954be1` (2025-06-04, *"exclude derivative of nyqist freq"*) changed how that QoI is
computed and **every archived record predates it**, so it is a different quantity today
(`claude_memory.md` gotcha #45). Derived with `findfirst` rather than hard-coded, so a change to the
QoI set cannot leave a stale `5` behind pointing at the wrong band.
"""
const IZ1632 = something(findfirst(==("Z[16,32]"), LABELS))

"""
Integral timescale per QoI, in TU, measured on the reference `dQ` (`analysis/results.md` section 1).

🔴 Not one number. These span a factor 36.8, and `T_exp` disagrees with `T_int` by up to 4x within a
single QoI. Every document in the project said 0.04 TU; neither estimator gives that
(`claude_memory.md` gotcha #30).

⚠️ They are timescales of the **correction**. The level decorrelates far more slowly, so the level's
saturation lead may lie beyond `10 x T_int`. That is reported, never extrapolated.
"""
const T_INT = [0.1118, 0.0082, 0.0923, 0.0669, 0.2926, 0.3017]

"Which record supplies the verification truth. See the header."
const TRUTH_SOURCE = get(ENV, "D6_TRUTH", "hf_reference")

const D6_DIR = get(ENV, "D6_OUT",
                   normpath(joinpath(HERE, "..", "exp_square_HIT", "output", "D6")))

# ---------------------------------------------------------------------------------------------
# index alignment
# ---------------------------------------------------------------------------------------------
#
# 🔑 Asserted, not assumed, and tested against planted step indices in `test/test_d6_score.jl`.
# Run step `s` is tracked step `n_k + s`; the forecast's `t = 0` is at run step `nwarm`, because the
# first `nwarm` steps replay the record's correction rather than predicting it.

"""
    forecast_column(lead; nwarm = N_WARM)

Column of a run's own `q` holding lead `lead`. Run step `nwarm + lead`, and `q` carries the initial
state, so column `nwarm + lead + 1` (`RikFlow.jl:294`).
"""
forecast_column(lead::Integer; nwarm::Integer = N_WARM) = nwarm + lead + 1

"""
    truth_column(n_k, lead; nwarm = N_WARM)

Column of the reference `q` verifying lead `lead` of a run launched from step `n_k`: tracked step
`n_k + nwarm + lead`, plus the same initial-state offset.
"""
truth_column(n_k::Integer, lead::Integer; nwarm::Integer = N_WARM) = n_k + nwarm + lead + 1

"""
    forecast_column_dq(lead; nwarm = N_WARM)
    truth_column_dq(n_k, lead; nwarm = N_WARM)

The same two, for `dQ`. `dQ` has **no** initial-state column -- it is `nstep` long against `q`'s
`nstep + 1` -- so both offsets drop by one. Keeping the two pairs separate is deliberate: a single
"column" helper shared between `q` and `dQ` is precisely the off-by-one that would misalign the
secondary metric while the primary one looked right.
"""
forecast_column_dq(lead::Integer; nwarm::Integer = N_WARM) = nwarm + lead
truth_column_dq(n_k::Integer, lead::Integer; nwarm::Integer = N_WARM) = n_k + nwarm + lead

# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------

"""
    load_members(dir = D6_DIR)

Every `d6_online_ic<k>_m<member>.jld2` in `dir`, grouped by IC and sorted by member.

Refuses a ragged ensemble: `spread_skill`'s finite-`M` correction is a function of `M`, so an IC
with fewer members than the rest would be silently down-weighted and its correction wrong.
"""
function load_members(dir = D6_DIR)
    isdir(dir) || return nothing
    pat = r"^d6_online_ic(\d+)_m(\d+)\.jld2$"
    byic = Dict{Int,Vector{Tuple{Int,String}}}()
    for f in readdir(dir)
        m = match(pat, f)
        m === nothing && continue
        push!(get!(byic, parse(Int, m[1]), Tuple{Int,String}[]),
              (parse(Int, m[2]), joinpath(dir, f)))
    end
    isempty(byic) && return nothing
    ks = sort(collect(keys(byic)))
    Ms = [length(byic[k]) for k in ks]
    allequal(Ms) || error("ragged ensemble: members per IC are $(sort(unique(Ms))). " *
                          "The finite-M correction is a function of M; fix the runs, do not average.")
    for k in ks
        sort!(byic[k], by = first)
        first.(byic[k]) == collect(1:Ms[1]) || error("IC $k has member ids $(first.(byic[k]))")
    end
    return (; ks, M = Ms[1], files = byic)
end

"""
    compare_validation(; dir = D6_DIR, io = stdout)

Check the validation run (ordinal 0) against the archived online ensemble it should reproduce.

🔑 **What makes this a correctness check and not a plausibility argument.** The validation IC is
`fields[1]` of the 10 TU tracked record -- the archived runs' own initial condition, so `n_k = 0`,
`ou_advance = 0` and the OU chain starts at zero, exactly as the archived driver leaves it -- and
`run_d6.jl` gives its members the archive's own model seeds, `Xoshiro(236 + member)`. Every input is
therefore the archive's, so the output must be too: member `i`'s `q` against archived replica `i`'s
first `size(q, 2)` columns. Agreement there exercises the whole D6 path at once: IC packaging, the
warm-up slice, `ou_advance` at its identity point, the driver, and the output format -- against a
trajectory produced years earlier by different code.

Reported per replica **and per QoI** as a relative rms in units of each QoI's own standard
deviation, plus whether the columns are bit-identical.

⚠️ Bit-identity is the ideal but not the acceptance criterion: the archive was produced by an older
RikFlow, and Float32 differences of a few ulp in the first steps amplify. What would indicate a real
defect is disagreement that is *large from step 1* -- a wrong warm-up slice, a misphased chain, or a
seed mismatch -- rather than growth from round-off.

🔴 **`Z[16,32]` is expected to disagree, and by a known amount.** Commit `09954be1` (2025-06-04,
*"exclude derivative of nyqist freq"*) changed how it is computed, and every archived record
predates it: the masks keep the Nyquist shell in the `[16,32]` band while `∂` no longer does, so
today's `Z[16,32]` is a **different quantity**, off by ~1.06e-3 on an identical velocity field
(`claude_memory.md` gotcha #45, reproduced both ways). `E[16,32]` is unaffected because it never
goes through `curl`. **So read the other five QoIs as the reproduction test and this one as
expected-to-differ** -- it is not evidence of a D6 defect, and a `Z[16,32]` offset of that size is
the sign that everything is working as understood.
"""
function compare_validation(; dir = D6_DIR, io = stdout)
    isdir(dir) || (println(io, "no run directory at $dir"); return nothing)
    pat = r"^d6_valid_ic(\d+)_m(\d+)\.jld2$"
    files = sort([(parse(Int, m[2]), joinpath(dir, f))
                  for f in readdir(dir) for m in (match(pat, f),) if m !== nothing])
    if isempty(files)
        println(io, "no validation runs in $dir (run `tools/run_d6.jl 0`)")
        return nothing
    end
    arch = try
        load_ensemble("LinReg1")
    catch err
        println(io, "no archived LinReg1 ensemble to compare against: ", err)
        return nothing
    end

    println(io, "\nValidation: ordinal 0 against the archived LinReg1 ensemble")
    @printf(io, "  archive: %s root, %d replicas\n", string(arch.root), length(arch.q))
    @printf(io, "  %8s %10s %12s   %s\n", "member", "identical", "max ex Z16", "per-QoI rel rms")
    rows = NamedTuple[]
    for (member, path) in files
        d = load(path)
        get(d, "validation", false) ||
            error("$path is not marked as a validation run; the glob picked up a scored file")
        member <= length(arch.q) ||
            (println(io, "  member $member has no archived replica"); continue)
        a = Float64.(d["q"])
        b = Float64.(arch.q[member])
        n = min(size(a, 2), size(b, 2))
        sd = vec(std(view(b, :, 1:n); dims = 2))
        e = vec(sqrt.(mean(abs2, view(a, :, 1:n) .- view(b, :, 1:n); dims = 2))) ./ sd
        ident = view(d["q"], :, 1:n) == view(arch.q[member], :, 1:n)
        # 🔴 The verdict excludes Z[16,32] (index 5): it is a *different quantity* today than when
        # the archive was written (gotcha #45), so taking the max over all six QoIs would report a
        # known convention change as a reproduction failure. It is still printed, and flagged.
        worst5 = maximum(e[i] for i in eachindex(e) if i != IZ1632)
        @printf(io, "  %8d %10s %12.3e   %s%s\n", member, ident, worst5,
                join((@sprintf("%8.1e", x) for x in e), " "),
                e[IZ1632] > 1e-4 ?
                    @sprintf("   [Z16-32 %.1e — gotcha #45, expected]", e[IZ1632]) : "")
        push!(rows, (; member, identical = ident, rel = e, nsteps = n, seed = d["seed"]))
    end
    if !isempty(rows)
        worst = maximum(maximum(r.rel[i] for i in eachindex(r.rel) if i != IZ1632)
                        for r in rows)
        nid = count(r -> r.identical, rows)
        @printf(io, "  => %d of %d bit-identical; worst relative rms %.3e over %d columns\n",
                nid, length(rows), worst, rows[1].nsteps)
        println(io, worst < 1e-2 ?
                "  ✅ the D6 path reproduces the archived trajectory from the archived inputs." :
                "  🔴 disagreement is large. Check the warm-up slice, ou_advance and the seed " *
                "before trusting\n     anything scored -- see `compare_validation`'s docstring.")
    end
    return rows
end

"""
    load_truth(source = TRUTH_SOURCE)

The verification truth as `(; q, dQ)`. `q` is `N_Q x 40001`; `dQ` is `N_Q x 40000` and comes from
the tracked record in either case, because the high-fidelity reference has no correction of its own
-- `dQ` is a property of the tracking run.
"""
function load_truth(source = TRUTH_SOURCE)
    dd = joinpath(HERE, "data")
    trk = joinpath(dd, "data_track2_dns512_les64_Re2000.0_tsim100.0_qois.jld2")
    isfile(trk) || error("no extracted tracked record at $trk; run analysis/extract_qois.jl")
    t = load(trk)
    if source == "hf_reference"
        hf = joinpath(dd, "hf_reference_tsim100.0_qois.jld2")
        isfile(hf) || error("no extracted HF reference at $hf; run analysis/extract_archive.jl")
        return (; q = load(hf, "q_ref"), dQ = t["dQ"], source)
    elseif source == "tracked"
        return (; q = t["q"], dQ = t["dQ"], source)
    else
        error("D6_TRUTH must be \"hf_reference\" or \"tracked\", got $(repr(source))")
    end
end

"""
    assemble(ens, truth, grid; level = true)

Build `(fc, truth)` as `K x N_Q x M x L` and `K x N_Q x L` on the sorted `grid` of step leads.

`level = true` takes the QoI level `q` from both sides, `false` the correction `dQ`; the two use
different column offsets and that is the point of them being separate functions above.

Every column index is bounds-checked against the record. A lead that runs past the reference throws
rather than being clipped, because a clipped lead reads as a saturated one.
"""
function assemble(ens, truth, grid::AbstractVector{<:Integer}; level::Bool = true)
    K = length(ens.ks)
    M = ens.M
    L = length(grid)
    ref = level ? truth.q : truth.dQ
    nq = size(ref, 1)
    fc = Array{Float64}(undef, K, nq, M, L)
    tr = Array{Float64}(undef, K, nq, L)
    fcol = level ? forecast_column : forecast_column_dq
    tcol = level ? truth_column : truth_column_dq

    for (ik, k) in pairs(ens.ks)
        n_k = nothing
        warm = nothing
        for (im, (mid, path)) in pairs(ens.files[k])
            d = load(path)
            n_k === nothing && (n_k = d["n_k"])
            warm === nothing && (warm = d["nwarm"])
            d["n_k"] == n_k || error("members of IC $k disagree on n_k")
            # V28: the burn-in offset must be identical across all members of one IC. If it were
            # not, lead 0 would mean a different forecast time for different members and the
            # ensemble at a "lead" would not be an ensemble at a lead at all.
            d["nwarm"] == warm || error("members of IC $k disagree on nwarm ($warm vs $(d["nwarm"]))")
            d["k"] == k || error("$path says k = $(d["k"])")
            x = level ? d["q"] : d["dQ"]
            nwarm = d["nwarm"]
            for (j, ℓ) in pairs(grid)
                c = fcol(ℓ; nwarm)
                c <= size(x, 2) || error("lead $ℓ needs column $c of a $(size(x, 2))-column run")
                fc[ik, :, im, j] = @view x[:, c]
            end
            if im == 1
                for (j, ℓ) in pairs(grid)
                    c = tcol(n_k, ℓ; nwarm)
                    c <= size(ref, 2) ||
                        error("lead $ℓ from IC k = $k needs reference column $c of " *
                              "$(size(ref, 2)); the IC pool cap was violated")
                    tr[ik, :, j] = @view ref[:, c]
                end
            end
        end
    end
    return fc, tr
end

# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------

fmt_lead(ℓ) = @sprintf("%d (%.4f TU)", ℓ, ℓ * DT)

"""
    tie_count(fc, tr, i, j)

How often the truth exactly equals one of the members, for QoI `i` at grid column `j`.

Trap 7 of the handoff: `metrics.md` #4 says the inherited stabiliser "produces exact duplicates by
construction", but the clamp has been measured never to fire on HIT -- min `|q*|` is 2.19e-2 against
a 1e-2 threshold -- so ties should be **absent**. The seeded tie-breaking stays as insurance and the
count is reported, because a non-zero count here would mean the census is wrong.
"""
function tie_count(fc::AbstractArray{<:Real,4}, tr::AbstractArray{<:Real,3}, i::Integer, j::Integer)
    K, _, M, _ = size(fc)
    n = 0
    for k in 1:K
        any(m -> fc[k, i, m, j] == tr[k, i, j], 1:M) && (n += 1)
    end
    return n
end

"""
    clamp_report(ens)

How often the inherited stabiliser fired during the forecast, counted **exactly**.

🔑 The clamp is `any(abs.(q_star) .< 1e-2) && (dQ .= 0)` in the `LinReg` path
(`time_series_methods.jl:162,165,190,193`), so a step on which it fired has a `dQ` column that is
**identically zero**. That is a direct, exact indicator and needs no `q_star`.

⚠️ It needs one, because `online_sgs` does not return `q_star`: `allocate_arrays_outputs` stores it
only in `:TRACK_REF` mode (`RikFlow.jl:110-119`), and `to_sgs_term` is out of scope to change.
`clamp_census` -- which wants `q_star` -- can still be run on a reconstruction
`q_star ~ q[:, 2:end] - dQ`, accurate to the O(||sgs||^2) gap of 5.9e-6-5.7e-4 relative (phase-0
check 0.4), which against a 1e-2 threshold and a measured minimum `|q*|` of 2.19e-2 cannot flip the
census. The exact zero-`dQ` count below is preferred anyway, and reported per IC.

The warm-up columns are excluded: there `dQ` is replayed from the record, not predicted.
"""
function clamp_report(ens)
    fired = Int[]
    total = 0
    for k in ens.ks, (_, path) in ens.files[k]
        dQ = load(path, "dQ")
        nwarm = load(path, "nwarm")
        n = 0
        for c in (nwarm + 1):size(dQ, 2)
            all(iszero, @view dQ[:, c]) && (n += 1)
        end
        push!(fired, n)
        total += size(dQ, 2) - nwarm
    end
    return (; nfired = sum(fired), nsteps = total, per_run = fired,
            rate = total == 0 ? NaN : sum(fired) / total)
end

"""
    report_grids(leads; io = stdout)

The lead grids themselves, which are a result: they say what the runs can and cannot resolve.
"""
function report_grids(leads; io = stdout)
    println(io, "\nLead grids -- {0.25, 0.5, 1, 2, 5, 10} x T_int(i), per QoI, in physical time")
    @printf(io, "  %-10s %8s   %s\n", "QoI", "T_int", "leads [steps]")
    for i in eachindex(leads)
        @printf(io, "  %-10s %8.4f   %s\n", LABELS[i], T_INT[i], join(leads[i], ", "))
    end
    @printf(io, "  union: %d distinct leads, longest %s of %d available\n",
            length(union_grid(leads)), fmt_lead(maximum(union_grid(leads))), N_LEAD)
end

"""
    report_scores(ss, rh, sat; label, io = stdout)

Metric #17 and RH-3 side by side, per QoI and lead.

The two belong in one table because neither is readable alone: a flat histogram is reliability, not
skill, and a ratio near 1 says nothing about whether the shape is right.
"""
function report_scores(ss, rh, sat; label, io = stdout)
    println(io, "\n", label, "  (K = ", ss.K, ", M = ", ss.M,
            ", finite-M correction ", @sprintf("%.4f", ss.correction), ")")
    for i in eachindex(ss.leads)
        s = sat[i]
        @printf(io, "\n  %-10s  saturation lead: %s\n", LABELS[i],
                s === nothing ? "NOT REACHED within the grid -- reported, not extrapolated" :
                fmt_lead(s))
        @printf(io, "    %10s %10s %10s %8s %8s %18s %18s\n",
                "lead", "spread", "skill", "ratio", "chi2_eff", "slope [95% CI]",
                "convexity [95% CI]")
        for (t, ℓ) in pairs(ss.leads[i])
            h = rh.hist[i][t]
            @printf(io, "    %10d %10.4g %10.4g %8.3f %8.1f  %6.2f [%6.2f,%6.2f]  %6.2f [%6.2f,%6.2f]\n",
                    ℓ, ss.spread[i][t], ss.skill[i][t], ss.ratio[i][t], h.chi2_eff,
                    h.slope, h.slope_ci[1], h.slope_ci[2],
                    h.convexity, h.convexity_ci[1], h.convexity_ci[2])
        end
    end
end

"""
    saturation(ss, truth_q, M)

The saturation lead per QoI, and the climatological level it is measured against.
"""
function saturation(ss, ref::AbstractMatrix, M::Integer)
    lev = [climatological_skill(collect(float.(view(ref, i, :))), M) for i in 1:size(ref, 1)]
    sat = [saturation_lead(ss.leads[i], ss.skill[i]; sat_level = lev[i]) for i in eachindex(lev)]
    return sat, lev
end

# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------

"""
    preview(; io = stdout)

What the scorer will do, without any runs: the lead grids, and the index alignment written out for
one IC so the arithmetic can be read rather than trusted.
"""
function preview(; io = stdout)
    leads = lead_grid(T_INT; dt = DT, nlead = N_LEAD)
    report_grids(leads; io)
    sel = select_ics(; K = 180)
    k, n_k = sel.k[1], sel.n[1]
    println(io, "\nIndex alignment, worked for the first IC (ordinal 1, k = $k, n_k = $n_k)")
    @printf(io, "  %8s %14s %16s %14s %16s\n",
            "lead", "run q col", "reference q col", "run dQ col", "reference dQ col")
    for ℓ in (0, 1, leads[2][1], leads[6][end])
        @printf(io, "  %8d %14d %16d %14d %16d\n", ℓ,
                forecast_column(ℓ), truth_column(n_k, ℓ),
                forecast_column_dq(ℓ), truth_column_dq(n_k, ℓ))
    end
    @printf(io, "  last reference column used: %d of %d\n",
            truth_column(last(sel.n), maximum(union_grid(leads))), N_REF + 1)
    println(io, "\nNo scored runs found. Submit exp_square_HIT/batch_scripts/run_d6.sh, pull the " *
                "output back into\n$(D6_DIR), then re-run this script.")
    # The validation run is independent of the scored set and worth reporting on its own, because
    # it is the check that has to pass before the scored numbers mean anything.
    compare_validation(; io)
    return leads
end

"""
    main(; dir = D6_DIR, preview_only = false, outdir = OUT, io = stdout)

Score whatever runs are in `dir`, print the tables to `io`, and write `d6_scores.jld2` under
`outdir`.

`outdir` and `io` are parameters so `test/test_d6_score.jl` can drive this whole path on synthetic
members. That matters more than it looks: without it, the first execution of `main` would be on
pilot data, which is to say after GPU time had already been spent.
"""
function main(; dir = D6_DIR, preview_only::Bool = false, outdir = OUT, io = stdout)
    leads = lead_grid(T_INT; dt = DT, nlead = N_LEAD)
    ens = preview_only ? nothing : load_members(dir)
    if ens === nothing
        preview(; io)
        return nothing
    end

    grid = union_grid(leads)
    truth = load_truth()
    report_grids(leads; io)
    @printf(io, "\nD6: %d initial conditions x %d members from %s\n", length(ens.ks), ens.M, dir)
    @printf(io, "truth: %s\n", truth.source)

    # The validation run first: if the D6 path does not reproduce the archived trajectory from the
    # archived inputs, nothing below is worth reading.
    compare_validation(; dir, io)

    cl = clamp_report(ens)
    @printf(io, "clamp: fired on %d of %d forecast steps (%.3g%%)%s\n",
            cl.nfired, cl.nsteps, 100 * cl.rate,
            cl.nfired == 0 ? " -- as measured everywhere else on HIT, it never fires" :
            " ⚠️ every number below is then partly the stabiliser's, not the model's")

    rng = Xoshiro(SEED)
    pos = lead_positions(grid, leads)
    out = Dict{Symbol,Any}()
    for (label, level) in (("LEVEL q  -- primary (gotcha #27)", true),
                           ("CORRECTION dQ -- secondary (gotcha #28)", false))
        fc, tr = assemble(ens, truth, grid; level)
        ss = spread_skill_by_lead(fc, tr; grid, leads)
        rh = rank_histogram_by_lead(fc, tr; grid, leads, rng)
        ref = level ? truth.q : truth.dQ
        sat, lev = saturation(ss, ref, ens.M)
        report_scores(ss, rh, sat; label, io)
        tag = level ? :level : :correction
        out[tag] = (; leads, grid, ss.ratio, ss.spread, ss.skill, ss.pooled_ratio,
                    ss.K, ss.M, ss.correction,
                    saturation = [s === nothing ? -1 : s for s in sat],
                    clim_level = lev,
                    counts = [[h.counts for h in hs] for hs in rh.hist],
                    slope = [[h.slope for h in hs] for hs in rh.hist],
                    slope_ci = [[collect(h.slope_ci) for h in hs] for hs in rh.hist],
                    convexity = [[h.convexity for h in hs] for hs in rh.hist],
                    convexity_ci = [[collect(h.convexity_ci) for h in hs] for hs in rh.hist],
                    chi2_eff = [[h.chi2_eff for h in hs] for hs in rh.hist],
                    n_eff = [[h.n_eff for h in hs] for hs in rh.hist],
                    blocklen = [[h.blocklen for h in hs] for hs in rh.hist],
                    ties = [[tie_count(fc, tr, i, j) for j in pos[i]] for i in eachindex(leads)])
        nties = sum(sum.(out[tag].ties))
        @printf(io, "\n  ties (truth exactly equal to a member): %d of %d instances.%s\n",
                nties, ss.K * sum(length, leads),
                nties == 0 ? " As expected -- the clamp never fires on HIT." :
                " ⚠️ Unexpected; metrics.md #4's duplicate-by-construction case was ruled out.")
        nsat = count(s -> s !== nothing, sat)
        @printf(io, "\n  %d of %d QoIs saturate inside the grid.%s\n", nsat, length(sat),
                nsat == length(sat) ? "" :
                " ⚠️ The rest are reported as not reached; do not extrapolate.")
    end

    mkpath(outdir)
    p = joinpath(outdir, "d6_scores.jld2")
    # Named explicitly rather than splatted: `jldsave`'s keywords must be symbols, and a `Dict`
    # splat is the kind of thing that works until the dictionary's key type changes.
    jldsave(p; level = out[:level], correction = out[:correction],
            labels = LABELS, T_int = T_INT, dt = DT, truth = truth.source,
            ics = ens.ks, clamp_nfired = cl.nfired, clamp_nsteps = cl.nsteps,
            written = string(now()))
    @printf(io, "\nwrote %s (%.1f kB)\n", p, filesize(p) / 1024)
    println(io, "⚠️  The report goes into `analysis/results.md`, not into meta_files/ and not left " *
            "here:\n    `.gitignore:12` is `*output/`.")
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; preview_only = ("--preview" in ARGS))
end
