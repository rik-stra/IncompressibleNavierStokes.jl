# D6 pre-flight. Run this before requesting any allocation.
#
# plan P2's pre-flight was HIT, 1 TU, 400 steps, 1 replica, about 0.01 SBU. This is that, plus the
# two checks D6 adds:
#
#   * 🔴 `savefreq > nt` really produced **no** velocity fields. That is the difference between
#     160 MB and 80 GB over the full 1800 runs, and it is silent if wrong.
#   * 🔴 `ou_advance` is actually wired through and actually changes the trajectory. The unit tests
#     prove the replay is correct arithmetic and `analysis/ou_replay.jl` proves the advance count is
#     right, but neither touches `online_sgs`. If the keyword were dropped on the floor between the
#     driver and `Setup`, every member's forcing would be out of phase with its own initial
#     condition, the spread-skill ratio would be biased downward, and nothing would say so.
#
# Usage (from exp_square_HIT/):
#   julia --project tools/smoke_d6.jl
#
# Writes into output/D6_smoke/, which is throwaway. ⚠️ 400 steps is a pipeline check, not a
# measurement: the lead grid reaches 1207 steps and nothing here may be reported as a result.

using Random
using JLD2
using Printf
using RikFlow
using IncompressibleNavierStokes
using CUDA

include(joinpath(@__DIR__, "run_d6.jl"))

const SMOKE_DIR = joinpath(EXP_DIR, "output", "D6_smoke")
const SMOKE_LEAD = 300          # 100 warm-up + 300 = 400 steps = 1 TU, plan P2's pre-flight

"""
    trajectory(pkg; ou_advance, nlead = 20, seed = 1)

One short forecast from an IC package, returning its `q`. Used only to compare `ou_advance = 0`
against `ou_advance = n_k` with everything else -- the field, the model, the seed -- held fixed.
"""
function trajectory(pkg; ou_advance::Int, nlead::Int = 20, seed = 1)
    T = Float32
    gpu = CUDA.functional()
    ArrayType = gpu ? CuArray : Array
    backend = gpu ? CUDABackend() : IncompressibleNavierStokes.CPU()
    p = pkg["params"]
    nwarm = pkg["provenance"].nwarm
    nt = nwarm + nlead
    Δt = T(p.Δt)
    mdl = model_file()
    hist_len, hist_var = load(mdl, "hist_len", "hist_var")
    nq = size(p.qois, 1)
    q_hist = ArrayType{T}(zeros(T, nq, hist_len))
    hist_var == :q_star_q && (q_hist = cat(q_hist, q_hist, dims = 1))
    sampler = RikFlow.LinReg(mdl, Xoshiro(seed), ArrayType;
                             q_hist, spinnup_data = ArrayType{T}(pkg["dQ_warm"]))
    d = online_sgs(; p..., tsim = T(Δt * nt), Δt, ArrayType, backend, savefreq = nt + 1,
                   ustart = ArrayType(pkg["u"]), time_series_method = sampler, ou_advance)
    return Array(d.q)
end

function main()
    @printf("D6 pre-flight, %s\n", CUDA.functional() ? "GPU" : "CPU")
    flush(stdout)

    # --- 1. the pre-flight run -----------------------------------------------------------------
    run_ic(1; M = 1, nlead = SMOKE_LEAD, od = SMOKE_DIR, force = true)

    files = filter(f -> startswith(f, "d6_online_"), readdir(SMOKE_DIR))
    @assert length(files) == 1 "expected one member file, got $files"
    d = load(joinpath(SMOKE_DIR, files[1]))
    for key in ("q", "dQ", "tau", "k", "n_k", "t_k", "seed", "ou_advance", "nwarm", "nlead",
                "model", "hist_len", "hist_var", "wall_seconds")
        @assert haskey(d, key) "output is missing key $key"
    end
    nt = d["nwarm"] + d["nlead"]
    @assert size(d["q"], 2) == nt + 1 "q has $(size(d["q"], 2)) columns, expected $(nt + 1)"
    @assert size(d["dQ"], 2) == nt "dQ has $(size(d["dQ"], 2)) columns, expected $nt"
    @assert !any(isnan, d["q"]) "q contains NaN"
    @assert !any(isnan, d["dQ"]) "dQ contains NaN"
    @assert d["ou_advance"] == d["n_k"] "ou_advance is $(d["ou_advance"]), expected n_k = $(d["n_k"])"
    @printf("  keys, shapes, finiteness: ok  (%.0f kB, %.2f s/TU)\n",
            filesize(joinpath(SMOKE_DIR, files[1])) / 1024,
            d["wall_seconds"] / (nt * 2.5e-3))

    # 🔴 The 80 GB check. `run_ic` asserts it too; asserted again here because it is the one that
    # decides whether the full array is affordable.
    @assert !haskey(d, "fields") "output carries velocity fields; savefreq did not suppress them"
    println("  no velocity fields written: ok")

    # --- 2. the clamp, counted exactly ---------------------------------------------------------
    # A step on which the stabiliser fired has an identically zero `dQ` column. Measured everywhere
    # else on HIT to never fire; if it does here, every later number is partly the clamp's.
    nfired = count(c -> all(iszero, @view d["dQ"][:, c]), (d["nwarm"] + 1):nt)
    @printf("  clamp fired on %d of %d forecast steps%s\n", nfired, nt - d["nwarm"],
            nfired == 0 ? " (as expected)" : "  ⚠️ UNEXPECTED on HIT")
    flush(stdout)

    # --- 3. ou_advance is wired through and does something --------------------------------------
    pkg = load_ic(1)
    q_on = trajectory(pkg; ou_advance = pkg["n_k"])
    q_off = trajectory(pkg; ou_advance = 0)
    @assert size(q_on) == size(q_off)
    @assert q_on[:, 1] ≈ q_off[:, 1] "the two runs did not start from the same field"
    same = q_on == q_off
    dev = maximum(abs.(q_on .- q_off) ./ max.(abs.(q_off), eps(Float32)))
    @assert !same "ou_advance = $(pkg["n_k"]) produced a trajectory identical to ou_advance = 0; " *
                  "the keyword is not reaching the OU chain and every member's forcing would be " *
                  "$(pkg["n_k"]) steps out of phase with its own initial condition"
    @printf("  ou_advance changes the trajectory: max relative deviation %.3g over %d steps\n",
            dev, size(q_on, 2) - 1)

    println("\npre-flight passed. ⚠️ 400 steps is a pipeline check, not a measurement — the lead " *
            "grid\nreaches 1207 steps. Submit batch_scripts/run_d6.sh with --array=1-5 next, and " *
            "write the\nmeasured s/TU into meta_files/handoff_p2c_d6.md section 2.")
    flush(stdout)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
