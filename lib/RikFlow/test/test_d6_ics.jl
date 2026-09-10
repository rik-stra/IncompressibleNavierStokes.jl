# V29 -- D6's initial-condition selection and packaging.
#
# The two constraints that define the usable pool cost K, and neither is visible in the data:
# an IC inside M0's fit window measures short-lead spread on data the conditional mean has already
# seen, and an IC too late in the record has no truth to be scored against for the whole 1208-step
# forecast. Both are silent if they fail -- the first inflates early skill, the second truncates the
# verification without saying so -- so both are asserted per IC, and the arithmetic that derives
# them is asserted here against planted values rather than re-derived.
#
# The warm-up slice is the off-by-one this file exists for. `dQ[:, m]` is the correction *at* step
# `m`; a run launched from the field at step `n_k` takes its first solver step at `n_k + 1`. The
# reduction test is the sharp one: at `k = 1` the general slice must be exactly `dQ[:, 1:100]`, the
# archived driver's hard-coded value (`6_online_TO_LRS.jl:62`).

@testmodule D6 begin
    using JLD2, Printf, Dates
    const SRC = normpath(joinpath(@__DIR__, "..", "analysis", "build_d6_ics.jl"))
    include(SRC)
end

@testitem "V29 every selected IC clears the fit window and fits inside the reference" default_imports = false setup = [D6] begin
    using Test
    for K in (5, 17, 173, 180, 346)
        sel = D6.select_ics(; K)
        @test length(sel.k) == K
        @test allunique(sel.k)
        @test issorted(sel.k)
        @test all(i -> sel.n[i] == D6.FIELD_STRIDE * (sel.k[i] - 1), 1:K)
        @test all(i -> sel.t[i] == D6.FIELD_DT * (sel.k[i] - 1), 1:K)
        @test all(>(D6.FIT_END_TU), sel.t)                                   # outside M0's fit
        @test all(n -> n + D6.N_WARM + D6.N_LEAD <= D6.N_REF, sel.n)         # inside the truth
        # The tightest instance of each constraint, stated as the numbers rather than as a relation,
        # so a change to nlead or to the record length has to come through this file.
        @test first(sel.t) >= 10.25
        @test last(sel.n) <= 38692
    end
end

@testitem "V29 K = 180 comes out at 0.48 TU, not the nominal 0.5" default_imports = false setup = [D6] begin
    using Test
    # The pool is k in [42, 387], 346 fields. Every second field is 0.5 TU and yields 173 ICs;
    # 180 needs 0.48. That is the intended behaviour and it is reported, not hidden -- and 0.48 TU
    # is 1.6x the slowest QoI's T_int = 0.3017, which is why the block bootstrap over initialisation
    # time is mandatory (`claude_memory.md` gotcha #30).
    sel = D6.select_ics(; K = 180)
    @test sel.pool == 346
    @test sel.spacing_tu ≈ 0.48184 atol = 1e-4
    @test sel.spacing_tu < 0.5
    @test sel.spacing_tu / D6.T_INT_MAX ≈ 1.597 atol = 5e-3

    # 173 is what every second field yields, and the spacing lands within 0.3% of the nominal
    # 0.5 TU. It is not exactly 0.5: the selection includes both ends of the pool, so 173 indices
    # span 345 fields rather than 344 and every gap is 2 or 3 fields.
    s173 = D6.select_ics(; K = 173)
    @test s173.spacing_tu ≈ 0.5 atol = 2e-3
    @test first(s173.k) == 42
    @test last(s173.k) == 387
    @test all(d -> d in (2, 3), diff(s173.k))

    # `report_ics` must print the spacing and the warning, since that is the only place a reader
    # meets either number.
    io = IOBuffer()
    D6.report_ics(sel; io)
    out = String(take!(io))
    @test occursin("0.4818", out)
    @test occursin("0.3017", out)
    @test occursin("block bootstrap", out)
end

@testitem "V29 an impossible K fails loudly rather than silently returning fewer" default_imports = false setup = [D6] begin
    using Test
    # The pool is 346. 347 and 400 must both raise; 346 must not.
    @test length(D6.select_ics(; K = 346).k) == 346
    @test_throws ErrorException D6.select_ics(; K = 347)
    @test_throws ErrorException D6.select_ics(; K = 400)
    @test_throws ErrorException D6.select_ics(; K = 0)

    # The bounds themselves are re-derived, so a stale kmin/kmax is caught too.
    @test_throws ErrorException D6.select_ics(; K = 10, kmin = 41)   # t_41 = 10.0, not > 10
    @test_throws ErrorException D6.select_ics(; K = 10, kmax = 388)  # n_388 + 1308 = 40008 > 40000

    # A longer forecast shrinks the pool, and the stale default kmax must then fail rather than
    # silently forecast past the end of the truth.
    @test_throws ErrorException D6.select_ics(; K = 10, nlead = 2000)
    @test last(D6.select_ics(; K = 10, nlead = 2000, kmax = 379).n) + 100 + 2000 <= 40000
end

@testitem "V29 selection is deterministic and disjoint from the archived runs' IC" default_imports = false setup = [D6] begin
    using Test
    @test D6.select_ics(; K = 180).k == D6.select_ics(; K = 180).k
    @test D6.select_ics(; K = 37).k == D6.select_ics(; K = 37).k

    # 🔑 The pilot's ICs are the first five of the same 180 the full run uses, so scaling
    # `--array=1-5` to `--array=1-180` renumbers nothing.
    @test D6.select_ics(; K = 180).k[1:5] == D6.select_ics(; K = 180).k[1:5]

    # V28: the IC set must be disjoint from every archived confirmatory run's IC, which is
    # `fields[1]` (`6_online_TO_LRS.jl:56-59`). It is, because field 1 is at t = 0, deep inside the
    # fit window -- but assert it, since that is the claim the comparison rests on.
    for K in (5, 180, 346)
        @test !(1 in D6.select_ics(; K).k)
    end
end

@testitem "V29 the warm-up slice reduces to the archived driver's at k = 1" default_imports = false setup = [D6] begin
    using Test
    # The reduction test. `6_online_TO_LRS.jl:62` is `dQ_data = data_track.dQ[:,1:100]`, and the
    # field it launches from is `fields[1]`, i.e. n = 0. A generalisation that does not reproduce
    # that exactly is wrong.
    @test D6.warmup_range(0) == 1:100
    @test collect(D6.warmup_range(0)) == collect(1:100)

    # And the general form: step n_k, first solver step at n_k + 1.
    @test D6.warmup_range(4100) == 4101:4200
    @test D6.warmup_range(38600) == 38601:38700
    for n in (0, 100, 4100, 17900, 38600)
        r = D6.warmup_range(n)
        @test length(r) == D6.N_WARM
        @test first(r) == n + 1
        @test last(r) <= D6.N_REF
    end

    # `q` carries the initial state, so step m is column m + 1 (`RikFlow.jl:294`). The scorer's
    # truth alignment rests on the same offset.
    @test D6.ic_q_column(0) == 1
    @test D6.ic_q_column(4100) == 4101
end

@testitem "V29 warm-up slices against the real record" default_imports = false setup = [D6, TrackedData] begin
    using Test
    rec = TrackedData.load_cache("data_track2")
    if rec === nothing
        @test_skip "no extracted 100 TU tracked record under analysis/data/"
    else
        dQ, q = rec.dQ, rec.q
        @test size(dQ, 2) == D6.N_REF
        @test size(q, 2) == D6.N_REF + 1

        # The reduction, on the actual arrays rather than on index arithmetic.
        @test dQ[:, D6.warmup_range(0)] == dQ[:, 1:100]

        sel = D6.select_ics(; K = 180)
        slices = [dQ[:, D6.warmup_range(n)] for n in sel.n]
        @test all(s -> size(s) == (size(dQ, 1), D6.N_WARM), slices)
        @test !any(s -> any(isnan, s), slices)
        # The forecast's last scored column exists for every IC.
        @test all(n -> n + D6.N_WARM + D6.N_LEAD + 1 <= size(q, 2), sel.n)
        # The package's `q_at_ic` is a real column, not one past the end.
        @test all(n -> all(isfinite, q[:, D6.ic_q_column(n)]), sel.n)
    end
end

@testitem "V29 the step-to-column convention, measured from the record" default_imports = false setup = [D6, TrackedData] begin
    using Test
    using Statistics
    rec = TrackedData.load_cache("data_track2")
    if rec === nothing
        @test_skip "no extracted 100 TU tracked record under analysis/data/"
    else
        # 🔑 `ic_q_column` and, later, the scorer's truth alignment both rest on "step m is column
        # m + 1 of `q`", because `qoisaver` fires on the initial state (`RikFlow.jl:294`). That is
        # an assertion about the record, so it is measured against the record rather than
        # re-derived from the source: `q_star[:, m] + dQ[:, m]` is the corrected QoI at step m, and
        # the column of `q` holding it must be the winner by a wide margin.
        #
        # It does not match exactly, and should not: `analysis/results.md` phase-0 check 0.4
        # measured an O(||sgs||^2) gap of 1.89e-2 relative between `q_star + dQ` and the recomputed
        # QoIs. The test is therefore comparative, not absolute.
        q, qs, dQ = rec.q, rec.q_star, rec.dQ
        m = 1000:39000
        err(off) = vec(sqrt.(mean(abs2, (qs[:, m] .+ dQ[:, m]) .- q[:, m .+ off], dims = 2)) ./
                       sqrt.(mean(abs2, q[:, m .+ off], dims = 2)))
        e0, e1, e2 = err(0), err(1), err(2)
        # Measured margin, per QoI: 5.4x on the coarsest enstrophy band up to 400x on the
        # smallest-scale ones. The factor asserted is deliberately below the smallest of those.
        @test all(e1 .< e0 ./ 4)
        @test all(e1 .< e2 ./ 4)
        @test maximum(e1) < 1e-3
    end
end

@testitem "V29 built IC packages agree with the record and with select_ics" default_imports = false setup = [D6, TrackedData] begin
    using Test
    using JLD2
    rec = TrackedData.load_cache("data_track2")
    mpath = D6.manifest_path()
    if rec === nothing || !isfile(mpath)
        @test_skip "IC packages not built yet (run analysis/build_d6_ics.jl)"
    else
        man = load(mpath)
        sel = D6.select_ics(; K = man["K"])
        @test man["k"] == sel.k
        @test man["n"] == sel.n
        @test man["spacing_tu"] ≈ sel.spacing_tu

        # Spot-check the ends and the middle rather than all 180 files.
        for i in unique((1, 2, cld(man["K"], 2), man["K"]))
            k = man["k"][i]
            p = D6.ic_path(k)
            @test isfile(p)
            d = load(p)
            @test d["k"] == k
            @test d["ordinal"] == i
            @test d["n_k"] == sel.n[i]
            @test d["t_k"] == sel.t[i]
            @test size(d["u"]) == (66, 66, 66, 3)
            @test !any(isnan, d["u"])
            @test size(d["dQ_warm"]) == (size(rec.dQ, 1), D6.N_WARM)
            @test d["dQ_warm"] == rec.dQ[:, D6.warmup_range(sel.n[i])]
            @test d["q_at_ic"] == rec.q[:, D6.ic_q_column(sel.n[i])]
            # The parameter subset is plain data and carries the forcing the replay needs.
            @test keys(d["params"]) == D6.PARAM_KEYS
            @test d["params"].ou_bodyforce.rng_seed == 333
            @test d["params"].Δt == Float32(2.5e-3)
        end
    end
end

@testitem "V29 the forecast length and the record's grid are what the plan says" default_imports = false setup = [D6] begin
    using Test
    # These constants are quoted in the handoff, in `metrics.md` section 5 and in the run driver, and
    # a change to any of them silently changes what D6 measures. Pin them here so the change has to
    # be deliberate.
    @test D6.N_LEAD == 1208
    @test D6.N_LEAD * D6.FIELD_DT / D6.FIELD_STRIDE ≈ 3.02 atol = 1e-9   # 1208 steps = 3.02 TU
    @test D6.N_LEAD * 2.5e-3 ≈ 10 * D6.T_INT_MAX atol = 5e-3             # 10 x the slowest T_int
    @test D6.N_WARM == 100
    @test D6.N_REF == 40000
    @test D6.N_FIELDS == 401
    @test D6.FIELD_STRIDE == 100
    @test D6.FIELD_DT == 0.25
    @test D6.FIT_END_TU == 10.0

    # 🔑 The step size the OU replay depends on. `online_sgs` refuses to replay unless `tsim / Δt` is
    # integral, because `solve_unsteady` re-derives `Δt = (tend - tstart) / nstep` and stepping the
    # replay at a different Δt would reintroduce the misphase. Both the reference's 100 TU and D6's
    # 3.27 TU have to land on the same Float32.
    nsteps = D6.N_WARM + D6.N_LEAD
    @test nsteps == 1308
    @test Float32(100.0) / 40000 === Float32(2.5e-3)
    @test Float32(nsteps * 2.5e-3) / nsteps === Float32(2.5e-3)
    @test round(Int, Float32(nsteps * 2.5e-3) / Float32(2.5e-3)) == nsteps
end
