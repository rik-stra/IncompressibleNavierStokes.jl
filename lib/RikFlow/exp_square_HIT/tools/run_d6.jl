# D6: one initial condition, M members, on Snellius.
#
# One array task = one IC = M members run sequentially. `ARGS[1]` is the **ordinal** (1..K) of the
# initial condition, not the field index `k`; the map from one to the other is `select_ics`'s and is
# deterministic, so scaling `--array=1-5` to `--array=1-180` renumbers nothing and the pilot's five
# ICs are the first five of the same set the full run uses.
#
# Usage:
#   julia --project tools/run_d6.jl <ordinal>
#
# Environment overrides, all optional:
#   D6_IC_DIR    where the `d6_ic_<k>.jld2` packages live
#   D6_MODEL     the fitted M0 to deploy (default: LinReg1 under this experiment's output tree)
#   D6_OUT       where to write `d6_online_ic<k>_m<member>.jld2`
#   D6_MEMBERS   M, default 10
#
# 🔴 Three things in here are load-bearing and none of them is obvious.
#
#  1. **`ou_advance = n_k`.** The initial condition is the reference's field at step `n_k`, and the
#     OU forcing that produced it is `n_k` steps into its own chain. `Setup` starts every chain at
#     zero, so without the replay every member's forcing is out of phase with its own initial
#     condition -- which inflates skill without inflating spread and biases the spread-skill ratio
#     downward, toward a false "over-confident" verdict. Verified by measurement in
#     `analysis/ou_replay.jl`; see `claude_memory.md` gotcha #33.
#  2. **`savefreq > nt`, so no velocity fields are written.** `params_track` carries
#     `savefreq = 100`; at 1308 steps that is ~13 snapshots x 3.3 MB per run, i.e. **80 GB** over
#     1800 runs against **160 MB** of QoIs. `fieldsaver` appends only when `state.n % nupdate == 0`
#     and its handler fires only on updates (`processors.jl:306-316`), so `nupdate = nt + 1` yields
#     an empty list. Asserted after every run, not assumed.
#  3. **The IC and the forcing are shared across the M members; only the model seed varies.** That
#     is what makes the rank histogram measure the surrogate's own dispersion and nothing else.
#
# ⚠️ Julia buffers stdout when redirected, so an empty `.out` file says nothing about progress --
# two healthy jobs have already been killed on that misreading. Every progress line here is
# followed by `flush(stdout)`.

if false                                                        #src
    include("../../src/RikFlow.jl")                             #src
    include("../../../../src/IncompressibleNavierStokes.jl")    #src
end                                                             #src

using Random
using JLD2
using Printf
using Dates
using RikFlow
using IncompressibleNavierStokes
using CUDA

# `select_ics`, `ic_path`, `manifest_path`, `N_WARM`, `N_LEAD` -- the one definition of D6's IC set.
include(normpath(joinpath(@__DIR__, "..", "..", "analysis", "build_d6_ics.jl")))

const EXP_DIR = normpath(joinpath(@__DIR__, ".."))

"""
    ic_dir()

Where the IC packages are. `D6_IC_DIR` wins; otherwise this experiment's own output tree (where a
Snellius copy lands) and then the local build directory, so the same script runs on both machines
without editing.
"""
function ic_dir()
    haskey(ENV, "D6_IC_DIR") && return ENV["D6_IC_DIR"]
    for d in (joinpath(EXP_DIR, "output", "d6_ics"), DEFAULT_IC_DIR)
        isfile(manifest_path(d)) && return d
    end
    error("no IC packages found: set D6_IC_DIR, or run analysis/build_d6_ics.jl first")
end

model_file() = get(ENV, "D6_MODEL", joinpath(EXP_DIR, "output", "TO_LRS", "LinReg1", "LinReg.jld2"))
out_dir() = get(ENV, "D6_OUT", joinpath(EXP_DIR, "output", "D6"))
n_members() = parse(Int, get(ENV, "D6_MEMBERS", "10"))

"""
    member_seed(k, member)

The model seed for one member. Shared IC, shared forcing, distinct seed -- so the ensemble's spread
is the surrogate's own and nothing else's.

The value is written into the output file, which is what makes a run reproducible: `hash` is not
promised to be stable across Julia versions, so the recorded seed, not this expression, is the
record of what was drawn.
"""
member_seed(k::Integer, member::Integer) = hash((:d6, Int(k), Int(member)))

"""
    validation_seed(member)

The **archived** driver's seed for replica `member`: `Xoshiro(seeds.to + i + 2)` with
`seeds.to = 234` (`6_online_TO_LRS.jl:37-41,83`), i.e. `Xoshiro(236 + member)`.

Used only by ordinal 0. Reusing the archive's stream is what makes the validation run comparable
to the archived trajectory column by column instead of merely in distribution; every *scored*
member goes through `member_seed` and therefore shares a stream with nothing archived.
"""
validation_seed(member::Integer) = UInt64(ARCHIVE_SEED_BASE + Int(member))

"""
    load_ic(ordinal; dir = ic_dir())

The IC package for array-task `ordinal`, with the manifest cross-checked against `select_ics`.

The manifest is `select_ics`'s recorded output, so comparing the two is not circular bookkeeping: it
catches an IC set built with a different `K`, `nlead` or record from the one this driver assumes,
which would otherwise show up only as a silently misaligned truth at scoring time.
"""
function load_ic(ordinal::Integer; dir = ic_dir())
    # 🔑 Ordinal 0 is the validation IC -- `fields[1]` of the 10 TU record, the archived runs' own
    # initial condition. It is not in the manifest and must not be: it sits inside M0's fit window
    # and V28 needs D6's scored set disjoint from it. See `build_validation_ic`.
    if ordinal == 0
        p = validation_path(dir)
        isfile(p) || error("no validation IC at $p; run " *
                           "`julia --project=analysis analysis/build_d6_ics.jl validation`")
        pkg = load(p)
        get(pkg, "validation", false) || error("$p is not marked as a validation package")
        pkg["n_k"] == 0 || error("validation IC is at step $(pkg["n_k"]), expected 0")
        return pkg
    end
    man = load(manifest_path(dir))
    K = man["K"]
    1 <= ordinal <= K || error("ordinal $ordinal is outside 1..$K")
    sel = select_ics(; K)
    man["k"] == sel.k || error("manifest disagrees with select_ics(K = $K); the IC packages were " *
                               "built from a different configuration than this driver assumes")
    k = man["k"][ordinal]
    pkg = load(ic_path(k, dir))
    pkg["ordinal"] == ordinal || error("package $(ic_path(k, dir)) says ordinal $(pkg["ordinal"])")
    pkg["n_k"] == man["n"][ordinal] || error("package and manifest disagree on n_k for k = $k")
    return pkg
end

"""
    run_ic(ordinal; M = n_members(), force = false, nlead = nothing, od = out_dir())

Run all `M` members for one initial condition and write one file per member.

`nlead` shortens the forecast and `od` redirects the output; both exist for `smoke_d6.jl` and must
be left at their defaults for anything whose numbers are reported. A short run is a pipeline check,
not a measurement -- the lead grid's longest entry is 1207 steps.
"""
function run_ic(ordinal::Integer; M::Integer = n_members(), force::Bool = false,
                nlead = nothing, od = out_dir())
    T = Float32
    gpu = CUDA.functional()
    ArrayType = gpu ? CuArray : Array
    backend = gpu ? CUDABackend() : IncompressibleNavierStokes.CPU()

    pkg = load_ic(ordinal)
    validation = get(pkg, "validation", false)
    k, n_k, t_k = pkg["k"], pkg["n_k"], pkg["t_k"]
    params_ic = pkg["params"]
    dQ_warm = pkg["dQ_warm"]
    q_at_ic = pkg["q_at_ic"]
    nwarm = pkg["provenance"].nwarm
    nlead = nlead === nothing ? pkg["provenance"].nlead : Int(nlead)
    nlead == pkg["provenance"].nlead ||
        @warn "forecast shortened to $nlead steps from $(pkg["provenance"].nlead); this is a " *
              "pipeline check, not a measurement -- the lead grid reaches 1207 steps"

    Δt = T(params_ic.Δt)
    nt = nwarm + nlead
    tsim = T(Δt * nt)
    # `online_sgs` asserts this too, before it replays anything; asserting it here as well means the
    # job fails in the first second rather than after Julia has compiled the solver.
    round(Int, tsim / Δt) == nt || error("tsim/Δt is not $nt")
    T(tsim) / nt === Δt || error("tsim/nt = $(T(tsim)/nt) is not Δt = $Δt; the OU replay and the " *
                                 "reference would step the chain differently")

    mdl = model_file()
    isfile(mdl) || error("no model at $mdl (set D6_MODEL)")
    hist_len, hist_var = load(mdl, "hist_len", "hist_var")
    nq = size(params_ic.qois, 1)

    mkpath(od)

    @printf("D6 ordinal %d -> field k = %d, n_k = %d, t_k = %.2f TU\n", ordinal, k, n_k, t_k)
    @printf("  %s, %d members, %d steps (%d warm-up + %d forecast) = %.4f TU, Δt = %g\n",
            gpu ? "GPU (CuArray)" : "CPU (Array)", M, nt, nwarm, nlead, tsim, Δt)
    @printf("  model %s: hist_len = %d, hist_var = %s\n", basename(mdl), hist_len, hist_var)
    @printf("  ou_advance = %d, savefreq = %d (> nt, so no fields)\n", n_k, nt + 1)
    if validation
        println("  🔑 VALIDATION run, not a scored one. This is the archived runs' own initial")
        println("     condition, so ou_advance = 0 -- the identity point of the replay -- and the")
        @printf("     model seeds are the archive's, Xoshiro(%d + member). Output goes to\n",
                ARCHIVE_SEED_BASE)
        println("     d6_valid_ic1_m*.jld2, which the scorer's glob cannot see. Compare it with")
        println("     `compare_validation` in analysis/score_d6.jl.")
    end
    flush(stdout)

    ustart = ArrayType(pkg["u"])
    params = (; params_ic..., tsim, Δt, ArrayType, backend, savefreq = nt + 1)

    t_all = time()
    for member in 1:M
        out = joinpath(od, validation ? "d6_valid_ic$(k)_m$(member).jld2" :
                                        "d6_online_ic$(k)_m$(member).jld2")
        if isfile(out) && !force
            @printf("  member %2d/%d: exists, skipping\n", member, M)
            flush(stdout)
            continue
        end

        seed = validation ? validation_seed(member) : member_seed(k, member)
        q_hist = ArrayType{T}(zeros(T, nq, hist_len))
        hist_var == :q_star_q && (q_hist = cat(q_hist, q_hist, dims = 1))
        sampler = RikFlow.LinReg(mdl, Xoshiro(seed), ArrayType;
                                 q_hist, spinnup_data = ArrayType{T}(dQ_warm))

        t0 = time()
        data = online_sgs(; params..., ustart, time_series_method = sampler, ou_advance = n_k)
        wall = time() - t0

        # 🔴 The 80 GB guard. If this ever fires, stop -- do not "clean up afterwards".
        isempty(data.fields) ||
            error("member $member wrote $(length(data.fields)) velocity fields; savefreq = " *
                  "$(nt + 1) should have written none. 1800 runs of this is 80 GB.")
        size(data.q, 2) == nt + 1 ||
            error("q has $(size(data.q, 2)) columns, expected nt + 1 = $(nt + 1)")
        # The run's first `q` column is the QoIs of `ustart`, so it must be the reference column the
        # package was cut from. This is the non-circular half of "the package and the record agree
        # on what step the IC is at" -- the package carries the record's value, this compares it
        # against QoIs recomputed here from the field itself.
        q0 = Array(data.q)[:, 1]
        rel = maximum(abs.(q0 .- q_at_ic) ./ max.(abs.(q_at_ic), eps(T)))
        rel < 1e-3 || error("run's q[:,1] differs from the record's q at step $n_k by $rel " *
                            "relative; the IC is not at the step the package claims")

        jldsave(out; q = Array(data.q), dQ = Array(data.dQ), tau = Array(data.tau),
                k, n_k, t_k, ordinal, member, seed, ou_advance = n_k, validation,
                nwarm, nlead, M, model = abspath(mdl), hist_len, hist_var,
                tsim, Δt, wall_seconds = wall,
                julia = string(VERSION), device = gpu ? "cuda" : "cpu", written = string(now()))

        @printf("  member %2d/%d: %.1f s (%.2f s/TU), q0 agrees to %.1e, %.0f kB\n",
                member, M, wall, wall / tsim, rel, filesize(out) / 1024)
        flush(stdout)
    end

    total = time() - t_all
    @printf("done: ordinal %d (k = %d), %d members in %.1f s (%.1f s/member, %.3f s/TU)\n",
            ordinal, k, M, total, total / M, total / (M * tsim))
    @printf("  ⚠️  write the measured s/TU into handoff_p2c_d6.md section 2; the plan's two SBU\n")
    @printf("      figures differ by 10x and neither should be trusted.\n")
    flush(stdout)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: run_d6.jl <ordinal>   (1..K the array task id; " *
                           "0 = the validation IC, fields[1] of the 10 TU record)")
    run_ic(parse(Int, ARGS[1]))
end
