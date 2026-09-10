# Build D6's initial-condition packages: one small file per initial condition, sliced once from the
# 100 TU tracked record.
#
# D6 is the multi-IC ensemble the baseline metric set is missing -- `K` initial conditions x `M`
# members, each a genuine forecast -- and it exists because every archived online run is one
# trajectory from one initial condition (`6_online_TO_LRS.jl:56-59` hard-codes
# `ustart = data_track.fields[1].u` and varies only the model seed). Spread and skill are both
# expectations over initial conditions, so with one IC every lead has exactly one verification
# instance and an RMSE from one sample is not an RMSE.
#
# Why one file per IC rather than one big file: JLD2 stores `data_track` as a single compound
# dataset, so `fields` cannot be partially read from the source and the slicing has to happen once,
# locally. Paying it once turns the 5-IC pilot into a ~17 MB copy instead of a 1.29 GB one, and lets
# a Snellius array task read exactly the one file it needs.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/build_d6_ics.jl          # K = 180
#   julia --startup-file=no --project=analysis analysis/build_d6_ics.jl 5        # a 5-IC pilot set
#
# Writes `analysis/output/d6_ics/d6_ic_<k>.jld2` plus a manifest. That directory is gitignored
# (`.gitignore:12` is `*output/`), which is what we want for ~600 MB of velocity snapshots.

using JLD2, Printf, Dates

# --------------------------------------------------------------------------------------------
# the record's own geometry
# --------------------------------------------------------------------------------------------
#
# These are properties of `data_track2_dns512_les64_Re2000.0_tsim100.0.jld2` and every one of them
# is asserted against the file in `build_d6_ics`, never assumed. They are named here because
# `select_ics` has to be callable -- and testable -- without opening a 1.29 GB file.

"Step spacing between saved fields: `params_track.savefreq`."
const FIELD_STRIDE = 100

"Time spacing between saved fields, in TU. `FIELD_STRIDE * Δt` with `Δt = 2.5e-3`."
const FIELD_DT = 0.25

"Columns in the reference `q`/`dQ`, i.e. the length of the truth the forecasts are scored against."
const N_REF = 40000

"Saved fields in the record. `fields[k].n = FIELD_STRIDE * (k-1)`, `fields[k].t = FIELD_DT * (k-1)`."
const N_FIELDS = 401

"""
End of M0's fit window in TU. `LinReg1` was fitted on steps 400-4000, i.e. `t in [1, 10]`.

An initial condition inside that window measures short-lead spread on data the conditional mean has
already seen, so the pool starts strictly after it.
"""
const FIT_END_TU = 10.0

"Warm-up steps replayed from the record before the forecast starts (the driver's `spinnup_data`)."
const N_WARM = 100

"""
Forecast length in steps. 1208 steps = 3.02 TU = 10x the *slowest* QoI's integral timescale.

⚠️ `t_int` is not one number: `T_int` spans 0.0082-0.3017 TU across the six QoIs, a factor 36.8
(`meta_files/claude_memory.md` gotcha #30). 3.02 TU is set by the slowest, so that one set of runs
brackets saturation for every band; the lead grids the scorer uses are per-QoI and in physical time
for the same reason.
"""
const N_LEAD = 1208

"The slowest QoI's integral timescale in TU -- the yardstick the achieved IC spacing is judged by."
const T_INT_MAX = 0.3017

const DEFAULT_TRACK_FILE = get(ENV, "RIKFLOW_TRACK_FILE",
    raw"C:\Users\rik\Documents\julia_code\IncompressibleNavierStokes.jl\lib\RikFlow\exp_square_HIT\output\new\data_track2_dns512_les64_Re2000.0_tsim100.0.jld2")

"""
The **10 TU** tracked record, which is the one the archived online runs launched from.

🔴 Not the same record as `DEFAULT_TRACK_FILE`, and the difference is the whole point of the
validation IC. `paper_runs/online_sgs.jl:50` reads
`data_track_trackingnoise_std_0.0_Re2000.0_tsim10.0_replica1.jld2` (line 52 has the 100 TU file
commented out) and takes `ustart = fields[1].u` and `dQ_data = dQ[:, 1:100]` from it. The two
records are two realisations whose `dQ` decorrelates to ~1 standard deviation past step 1000
(`claude_memory.md` gotcha #39), so a validation run built from the 100 TU record could only ever
agree with the archive approximately. Built from *this* record, the inputs are the archive's own.
"""
const VALIDATION_TRACK_FILE = get(ENV, "RIKFLOW_TRACK10_FILE",
    raw"C:\Users\rik\Documents\julia_code\IncompressibleNavierStokes.jl\lib\RikFlow\exp_square_HIT\paper_runs\output\tracking\tracking\data_track_trackingnoise_std_0.0_Re2000.0_tsim10.0_replica1.jld2")

"""
The archived online driver's model-seed base: `Xoshiro(seeds.to + i + 2)` with `seeds.to = 234`
(`6_online_TO_LRS.jl:37-41,83` and `paper_runs/online_sgs.jl:84`), so replica `i` used
`Xoshiro(236 + i)`.

The validation run reuses it — that is what makes the comparison against the archive exact rather
than merely distributional. D6's scoring runs deliberately do **not**: they use
`hash((:d6, k, member))`, so no scored member shares a stream with an archived replica.
"""
const ARCHIVE_SEED_BASE = 236

const DEFAULT_IC_DIR = joinpath(@__DIR__, "output", "d6_ics")

"""
The keys of `params_track` an IC package carries forward.

Deliberately a subset. `params_track` also holds `ArrayType`, `backend`, `filters` and `ref_reader`,
and those are useless or harmful here: the first two are set by whichever machine runs the forecast,
`ref_reader` carries the whole tracked QoI series and would add ~2 MB to *every* IC file, and all
four come back from JLD2 as reconstructed placeholder types on a machine without CUDA. What is kept
is plain numbers, strings and tuples, and it is exactly the set `online_sgs` reads.
"""
const PARAM_KEYS = (:D, :Re, :lims, :qois, :nles, :Δt, :ou_bodyforce)

# --------------------------------------------------------------------------------------------
# IC selection
# --------------------------------------------------------------------------------------------

"""
    select_ics(; K = 180, kmin = 42, kmax = 387, nwarm = N_WARM, nlead = N_LEAD, nref = N_REF, ...)

Choose `K` field indices as evenly spaced over the usable pool as the pool allows, and assert both
constraints that define the pool for every one of them. Returns
`(; k, n, t, ordinal, spacing_tu, spacing_fields, pool, kmin, kmax)`.

**Two exclusions, both of which cost K.**

 1. **The training window.** `t_k > FIT_END_TU`, so `k >= 42`.
 2. 🔴 **The reference length.** Truth is the 40 001-column reference. A run from `n_k` replays
    `nwarm` warm-up steps and then forecasts `nlead`, so it needs `n_k + nwarm + nlead <= nref`,
    i.e. `n_k <= 38692` and `k <= 387`.

So the pool is `k in [42, 387]`, **346 fields**, and at exactly 0.5 TU spacing (every second field)
that yields **K = 173**, not 180. Reaching `K = 180` needs 0.48 TU spacing. Neither answer is
hard-coded: the bounds are re-derived here from `nwarm`, `nlead`, `nref` and the record's grid, the
supplied `kmin`/`kmax` are checked against them, and the achieved spacing is reported rather than
assumed.

⚠️ **The achieved spacing is not an independence claim.** 0.48 TU is 1.6x the slowest QoI's
`T_int = 0.3017`, so the initial conditions are only weakly independent. Every interval over these
`K` instances needs a block bootstrap over initialisation time with a **QoI-dependent** block
length, and nothing may claim `K` independent instances (`claude_memory.md` gotcha #30).
"""
function select_ics(; K::Integer = 180, kmin::Integer = 42, kmax::Integer = 387,
                    nwarm::Integer = N_WARM, nlead::Integer = N_LEAD, nref::Integer = N_REF,
                    nfields::Integer = N_FIELDS, nstride::Integer = FIELD_STRIDE,
                    dt_field::Real = FIELD_DT, tfit_end::Real = FIT_END_TU)
    K >= 1 || error("select_ics: K must be at least 1, got $K")

    step_of(k) = nstride * (k - 1)
    time_of(k) = dt_field * (k - 1)

    # Re-derive the pool rather than trusting the arguments, so that a change to nlead, to the
    # reference length or to the fit window shows up as a failure here instead of as a quietly
    # truncated forecast several hundred GPU-hours later.
    kmin_req = floor(Int, tfit_end / dt_field) + 2          # first k with t_k strictly > tfit_end
    kmax_req = min(nfields, div(nref - nwarm - nlead, nstride) + 1)

    kmin >= kmin_req || error("select_ics: kmin = $kmin is inside the fit window; " *
                              "t_$kmin = $(time_of(kmin)) is not > $tfit_end (need kmin >= $kmin_req)")
    kmax <= kmax_req || error("select_ics: kmax = $kmax overruns the reference; " *
                              "n_$kmax + $nwarm + $nlead = $(step_of(kmax) + nwarm + nlead) > $nref " *
                              "(need kmax <= $kmax_req)")

    pool = kmax - kmin + 1
    K <= pool || error("select_ics: K = $K exceeds the usable pool of $pool fields " *
                       "(k in [$kmin, $kmax] after excluding the fit window and the reference cap). " *
                       "Reduce K, or shorten nlead.")

    # Evenly as the pool allows, endpoints included, deterministic. `round` on a `range` cannot
    # collide while the step exceeds one field, and K <= pool guarantees that; `allunique` says so
    # rather than leaving it to the argument above.
    k = K == 1 ? [kmin] : round.(Int, range(kmin, kmax; length = K))
    allunique(k) || error("select_ics: K = $K produced repeated field indices")
    issorted(k) || error("select_ics: index selection is not sorted")

    n = step_of.(k)
    t = time_of.(k)

    for (i, ki) in pairs(k)
        t[i] > tfit_end ||
            error("select_ics: IC $i (k = $ki) is inside the fit window: t = $(t[i]) <= $tfit_end")
        n[i] + nwarm + nlead <= nref ||
            error("select_ics: IC $i (k = $ki) overruns the reference: " *
                  "$(n[i]) + $nwarm + $nlead > $nref")
    end

    dfield = K == 1 ? Float64[] : Float64.(diff(k))
    spacing_fields = K == 1 ? NaN : (kmax - kmin) / (K - 1)
    spacing_tu = spacing_fields * dt_field

    return (; k, n, t, ordinal = collect(1:K), spacing_tu, spacing_fields,
            spacing_min_tu = isempty(dfield) ? NaN : minimum(dfield) * dt_field,
            spacing_max_tu = isempty(dfield) ? NaN : maximum(dfield) * dt_field,
            pool, kmin, kmax, nwarm, nlead, nref, K)
end

"""
    report_ics(sel; io = stdout)

Print what `select_ics` achieved, including the two facts that must not be hidden: the requested
`K = 180` comes out at 0.48 TU spacing rather than the nominal 0.5, and that spacing is only
1.6x the slowest QoI's integral timescale.
"""
function report_ics(sel; io = stdout)
    @printf(io, "  pool           k in [%d, %d]  (%d fields, after the fit window and the reference cap)\n",
            sel.kmin, sel.kmax, sel.pool)
    @printf(io, "  selected       K = %d  ordinals 1..%d -> k = %d .. %d\n",
            sel.K, sel.K, first(sel.k), last(sel.k))
    @printf(io, "  steps          n = %d .. %d       last forecast column %d of %d\n",
            first(sel.n), last(sel.n), last(sel.n) + sel.nwarm + sel.nlead, sel.nref)
    @printf(io, "  times          t = %.2f .. %.2f TU  (fit window ends at %.1f TU)\n",
            first(sel.t), last(sel.t), FIT_END_TU)
    if sel.K > 1
        @printf(io, "  spacing        %.4f TU mean  (%.2f fields; %.2f .. %.2f TU realised)\n",
                sel.spacing_tu, sel.spacing_fields, sel.spacing_min_tu, sel.spacing_max_tu)
        @printf(io, "  ⚠️  that is %.2f x the slowest QoI's T_int = %.4f TU, so the ICs are only\n",
                sel.spacing_tu / T_INT_MAX, T_INT_MAX)
        @printf(io, "      weakly independent: a block bootstrap over initialisation time, with a\n")
        @printf(io, "      QoI-dependent block length, is mandatory. Never claim K independent instances.\n")
    end
    return nothing
end

# --------------------------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------------------------

"Path of the package for field index `k`."
ic_path(k::Integer, dir = DEFAULT_IC_DIR) = joinpath(dir, "d6_ic_$(k).jld2")

"Path of the manifest describing a whole IC set."
manifest_path(dir = DEFAULT_IC_DIR) = joinpath(dir, "d6_ic_manifest.jld2")

"""
    warmup_range(n_k, nwarm = N_WARM)

The warm-up slice of the reference `dQ` for a run launched from the field at step `n_k`.

🔑 This is the off-by-one to get right. `dQ[:, m]` is the correction **at** step `m`, and a run
launched from the field at step `n_k` takes its first solver step at `n_k + 1`, so the slice is
`n_k .+ (1:nwarm)`. Sanity: `k = 1` gives `n = 0` and `dQ[:, 1:100]`, which is exactly the archived
driver's hard-coded slice (`6_online_TO_LRS.jl:62`). A generalisation that does not reduce to that
at `k = 1` is wrong.
"""
warmup_range(n_k::Integer, nwarm::Integer = N_WARM) = n_k .+ (1:nwarm)

"""
    ic_q_column(n_k)

The reference `q` column holding the QoIs of the field at step `n_k`.

`q` has `nstep + 1` columns because `qoisaver` fires on the initial state as well
(`RikFlow.jl:294`), so step `m` is column `m + 1`. The packages carry this column so a forecast can
assert on the compute node that its own first `q` column -- the QoIs of `ustart` -- is the one the
record says it is, before it spends nineteen seconds forecasting from the wrong field.
"""
ic_q_column(n_k::Integer) = n_k + 1

"""
    build_d6_ics(; K = 180, track_file, outdir, force = false)

Slice the tracked record once and write one `d6_ic_<k>.jld2` per selected initial condition.

Each package holds the velocity field `u`, its step `n_k` and time `t_k`, the `nwarm`-column
warm-up slice of the reference `dQ`, the reference QoI column at the IC, the `PARAM_KEYS` subset of
`params_track`, and provenance. Returns the selection.
"""
function build_d6_ics(; K::Integer = 180, track_file = DEFAULT_TRACK_FILE, outdir = DEFAULT_IC_DIR,
                      force::Bool = false)
    isfile(track_file) || error("no such tracking file: $track_file")
    mkpath(outdir)

    @printf("reading %s (%.2f GB) ...\n", basename(track_file), filesize(track_file) / 2^30)
    t0 = time()
    d, params_track = jldopen(track_file, "r") do io
        haskey(io, "data_track") || error("no data_track in $track_file; keys = $(keys(io))")
        io["data_track"], io["params_track"]
    end
    @printf("  read in %.1f s\n", time() - t0)

    # The record's geometry, asserted rather than assumed. `select_ics` derives the pool from these
    # numbers, so if the record ever changes shape the failure belongs here and not downstream.
    fields = d.fields
    nfields = length(fields)
    nref = size(d.dQ, 2)
    nfields == N_FIELDS || error("record has $nfields fields, expected $N_FIELDS")
    nref == N_REF || error("record has $nref dQ columns, expected $N_REF")
    size(d.q, 2) == nref + 1 ||
        error("q has $(size(d.q, 2)) columns, expected nref + 1 = $(nref + 1); " *
              "the initial-state offset that `ic_q_column` and the scorer's truth alignment rely on")
    all(fields[k].n == FIELD_STRIDE * (k - 1) for k in 1:nfields) ||
        error("field step spacing is not $FIELD_STRIDE; the n_k = $FIELD_STRIDE (k-1) map is wrong")
    all(isapprox(fields[k].t, FIELD_DT * (k - 1); atol = 1e-5) for k in 1:nfields) ||
        error("field time spacing is not $FIELD_DT TU")
    params_track.savefreq == FIELD_STRIDE ||
        error("params_track.savefreq = $(params_track.savefreq), expected $FIELD_STRIDE")

    sel = select_ics(; K, nfields, nref)
    println("\nIC selection")
    report_ics(sel)

    params = NamedTuple{PARAM_KEYS}(map(k -> getproperty(params_track, k), PARAM_KEYS))
    provenance = (; source = abspath(track_file), source_bytes = filesize(track_file),
                  built = string(now()), nwarm = sel.nwarm, nlead = sel.nlead, nref = sel.nref,
                  K = sel.K, spacing_tu = sel.spacing_tu)

    println("\nwriting $(sel.K) packages to $outdir")
    written = 0
    bytes = 0
    for (i, k) in pairs(sel.k)
        out = ic_path(k, outdir)
        n_k = sel.n[i]
        t_k = sel.t[i]

        if isfile(out) && !force
            bytes += filesize(out)
            continue
        end

        rows = warmup_range(n_k, sel.nwarm)
        last(rows) <= nref || error("warm-up slice for k = $k runs past the record")
        dQ_warm = Array(d.dQ[:, rows])
        size(dQ_warm, 2) == sel.nwarm ||
            error("warm-up slice for k = $k has $(size(dQ_warm, 2)) columns, expected $(sel.nwarm)")
        any(isnan, dQ_warm) && error("warm-up slice for k = $k contains NaN")

        u = Array(fields[k].u)
        any(isnan, u) && error("field k = $k contains NaN")
        fields[k].n == n_k || error("field k = $k is at step $(fields[k].n), expected $n_k")

        jldsave(out; u, n_k, t_k, k, ordinal = i, dQ_warm,
                q_at_ic = Array(d.q[:, ic_q_column(n_k)]), params, provenance)
        written += 1
        bytes += filesize(out)
    end

    manifest = manifest_path(outdir)
    jldsave(manifest; k = sel.k, n = sel.n, t = sel.t, ordinal = sel.ordinal, K = sel.K,
            kmin = sel.kmin, kmax = sel.kmax, spacing_tu = sel.spacing_tu, provenance, params)

    @printf("  %d written, %d already present; %.1f MB total, %.2f MB each\n",
            written, sel.K - written, bytes / 2^20, bytes / 2^20 / sel.K)
    @printf("  manifest: %s\n", basename(manifest))
    println("\nto copy the first 5 (the pilot): " *
            join(basename.(ic_path.(sel.k[1:min(5, sel.K)], outdir)), ", "))
    return sel
end

# --------------------------------------------------------------------------------------------
# the validation IC
# --------------------------------------------------------------------------------------------

"Path of the validation package. A distinct name, for the reason in `build_validation_ic`."
validation_path(dir = DEFAULT_IC_DIR) = joinpath(dir, "d6_ic_validation.jld2")

"""
    build_validation_ic(; track_file = VALIDATION_TRACK_FILE, outdir, nwarm, nlead, force)

Build the one IC that is **not** for scoring: `fields[1]` of the 10 TU tracked record, i.e. the
initial condition every archived online run launched from.

🔑 **Why it exists.** A D6 run from here has `n_k = 0`, so `ou_advance = 0` and the OU chain starts
at zero — which is exactly what the archived driver does, and is the identity point of the whole
replay mechanism. Give it the archive's model seeds (`ARCHIVE_SEED_BASE`) and its `q` must
reproduce the archived replica's first `nwarm + nlead + 1` columns. That is a correctness check on
the entire D6 path — IC packaging, warm-up slicing, `ou_advance`, the driver, the output format —
against a trajectory produced years earlier by different code.

🔴 **It is deliberately kept out of `select_ics` and out of everything scored**, for two
independent reasons, and merging it in would break both:

 1. `t_1 = 0` is **inside** M0's fit window, so its short-lead spread would be measured on data the
    conditional mean has already seen. `select_ics` asserts `t_k > 10` precisely to exclude it.
 2. V28 requires D6's IC set to be **disjoint** from the archived runs' IC, which is this one.
    `test_d6_ics.jl` asserts `!(1 in select_ics(; K).k)`; that test is only meaningful while this
    package stays outside the selection.

Hence the separate filename, and hence `run_d6.jl` writing its members as `d6_valid_ic1_m*.jld2`
where the scorer's own glob cannot see them.
"""
function build_validation_ic(; track_file = VALIDATION_TRACK_FILE, outdir = DEFAULT_IC_DIR,
                             nwarm::Integer = N_WARM, nlead::Integer = N_LEAD,
                             force::Bool = false)
    isfile(track_file) || error("no such tracking file: $track_file")
    out = validation_path(outdir)
    if isfile(out) && !force
        @printf("validation IC exists, skipping: %s (%.2f MB)\n", basename(out),
                filesize(out) / 2^20)
        return out
    end
    mkpath(outdir)

    @printf("reading %s (%.2f GB) ...\n", basename(track_file), filesize(track_file) / 2^30)
    d, params_track = jldopen(track_file, "r") do io
        haskey(io, "data_track") || error("no data_track in $track_file; keys = $(keys(io))")
        io["data_track"], io["params_track"]
    end

    # The geometry checks that apply to *this* record. It is 10 TU, not 100, so `N_FIELDS` and
    # `N_REF` do not; what must hold is the field grid and that the warm-up slice fits.
    fields = d.fields
    fields[1].n == 0 || error("fields[1] is at step $(fields[1].n), expected 0")
    isapprox(fields[1].t, 0.0; atol = 1e-6) || error("fields[1].t = $(fields[1].t), expected 0")
    params_track.savefreq == FIELD_STRIDE ||
        error("params_track.savefreq = $(params_track.savefreq), expected $FIELD_STRIDE")
    size(d.q, 2) == size(d.dQ, 2) + 1 || error("q is not one column longer than dQ")
    size(d.dQ, 2) >= nwarm || error("record has $(size(d.dQ, 2)) dQ columns, need >= $nwarm")

    rows = warmup_range(0, nwarm)
    first(rows) == 1 && last(rows) == nwarm ||
        error("the warm-up slice at n = 0 is $(rows), expected 1:$nwarm")
    dQ_warm = Array(d.dQ[:, rows])
    any(isnan, dQ_warm) && error("warm-up slice contains NaN")
    u = Array(fields[1].u)
    any(isnan, u) && error("field contains NaN")

    params = NamedTuple{PARAM_KEYS}(map(k -> getproperty(params_track, k), PARAM_KEYS))
    provenance = (; source = abspath(track_file), source_bytes = filesize(track_file),
                  built = string(now()), nwarm, nlead, K = 0, spacing_tu = NaN,
                  nref = size(d.dQ, 2))
    jldsave(out; u, n_k = 0, t_k = 0.0, k = 1, ordinal = 0, dQ_warm,
            q_at_ic = Array(d.q[:, ic_q_column(0)]), params, provenance,
            validation = true, archive_seed_base = ARCHIVE_SEED_BASE)

    @printf("wrote %s (%.2f MB)\n", basename(out), filesize(out) / 2^20)
    println("  ⚠️  validation only: t_1 = 0 is inside M0's fit window and this is the archived " *
            "runs' own IC,\n      so it is excluded from `select_ics` and from everything scored. " *
            "Run it as ordinal 0.")
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "validation"
        build_validation_ic()
    else
        K = isempty(ARGS) ? 180 : parse(Int, ARGS[1])
        build_d6_ics(; K)
        println()
        build_validation_ic()
    end
end
