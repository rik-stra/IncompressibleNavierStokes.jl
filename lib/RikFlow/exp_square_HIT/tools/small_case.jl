#=
    small_case.jl — the differential test for the upstream INS merge.

A shrunk copy of the production HIT pipeline (`2_HF_ref.jl` stage 1, `3_track_ref.jl` stage 2),
small enough to run on a CPU in minutes, whose output is compared **bit for bit** across the
merge.

    julia --startup-file=no --project=lib/RikFlow \
        lib/RikFlow/exp_square_HIT/tools/small_case.jl generate   # write golden data
    julia --startup-file=no --project=lib/RikFlow \
        lib/RikFlow/exp_square_HIT/tools/small_case.jl check      # run and compare

🔴 The golden data must be generated on the **pre-merge** solver and committed. Generating it
after the merge compares the merged solver against itself and proves nothing.

## Why the bands are not the production bands

Production QoI bands are `[0,6] [7,15] [16,32]` on a 64³ LES, whose Nyquist index is 32 — so the
top band *touches* the Nyquist plane. That is the entire mechanism of `claude_memory.md` gotchas
#45/#46: `get_masks_and_partials` builds the masks from `k` **before** zeroing the Nyquist entry,
and builds `∂` **after**, so the top band contains modes whose derivative is deliberately zero.
`∂` feeds `get_vi_functions`, hence `tau` and `dQ`, so this is in the model, not just the
diagnostic.

On a 32³ LES the Nyquist index is 16, so the bands here are `[0,3] [4,8] [9,16]` — top band
reaching 16. A band set that stopped short of Nyquist would exercise none of what this merge
endangers, and the case would pass while being blind. `assert_nyquist_in_top_band` checks the
property directly rather than trusting the arithmetic.

## Parameters

One block, environment-overridable, no shadowing. ⚠️ `2_HF_ref.jl` defines the production
parameters and then silently overwrites them with a "small test parameters" block 6 lines later,
so as committed it does not run what it appears to run. That pattern is deliberately not
reproduced here: every parameter is defined exactly once.

The defaults below are the small-case values, not the production ones — this file *is* the small
case, and `2_HF_ref.jl` remains the production driver. Override with `SMALL_CASE_*` to shrink
further when iterating.
=#

using IncompressibleNavierStokes
using JLD2
using Printf
using Random
using RikFlow

# ---------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------

envint(key, default) = parse(Int, get(ENV, key, string(default)))
envf32(key, default) = parse(Float32, get(ENV, key, string(default)))

"Parameters of the small case. Mirrors `2_HF_ref.jl` / `3_track_ref.jl`, shrunk."
function small_case_params()
    T = Float32

    n_dns = envint("SMALL_CASE_N_DNS", 64)      # production: 512
    n_les = envint("SMALL_CASE_N_LES", 32)      # production: 64
    Re = envf32("SMALL_CASE_RE", 2_000)         # production: 2000
    dt_dns = envf32("SMALL_CASE_DT_DNS", 2.5e-4)  # production: 2.5e-4
    dt_les = envf32("SMALL_CASE_DT_LES", 2.5e-3)  # production: 2.5e-3
    tsim = envf32("SMALL_CASE_TSIM", 0.25)      # production: 100
    tburn = envf32("SMALL_CASE_TBURN", 0.05)    # production: 4

    # QoI bands, rescaled to the 32³ LES Nyquist of 16. See the header.
    qois = [
        ["Z", 0, 3],
        ["E", 0, 3],
        ["Z", 4, 8],
        ["E", 4, 8],
        ["Z", 9, 16],
        ["E", 9, 16],
    ]

    # Forcing: identical to production, including the seed.
    forcing = (; T_L = 0.01, e_star = 0.1, k_f = sqrt(2), rng_seed = 333)

    # DNS steps between QoI samples. 10 gives 100 samples over tsim, as in production
    # (savefreq 10 at Δt 2.5e-4 samples the QoIs at the LES Δt of 2.5e-3).
    savefreq = envint("SMALL_CASE_SAVEFREQ", 10)
    # DNS steps between stored filtered fields.
    plotfreq = envint("SMALL_CASE_PLOTFREQ", 100)

    (;
        T,
        n_dns,
        n_les,
        Re,
        dt_dns,
        dt_les,
        tsim,
        tburn,
        qois,
        forcing,
        savefreq,
        plotfreq,
        lims = ((T(0), T(1)), (T(0), T(1)), (T(0), T(1))),
        D = 3,
        ArrayType = Array,
        backend = IncompressibleNavierStokes.CPU(),
    )
end

golden_dir() = normpath(joinpath(@__DIR__, "..", "..", "test", "data"))
golden_path() = joinpath(golden_dir(), "small_case_golden.jld2")

# ---------------------------------------------------------------------------------------------
# The Nyquist assertion (checklist item 8)
# ---------------------------------------------------------------------------------------------

"""
    assert_nyquist_in_top_band(p)

Fail loudly unless the **top QoI band's mask contains modes on a Nyquist plane**, and unless the
partial-derivative arrays `∂` are zero on that plane.

Both halves matter, and they are the two halves of gotcha #46. The mask is built from `k` before
the Nyquist entry is zeroed; `∂` is built after. If a future refactor moved the zeroing before the
mask construction, the masks would silently lose the Nyquist shell and this case would stop
testing the thing it exists to test — without any error.

Returns the number of retained Nyquist-plane modes in the top band, for the record.
"""
function assert_nyquist_in_top_band(p)
    setup = les_setup(p)
    to_setup = RikFlow.TO_Setup(;
        p.qois,
        to_mode = :CREATE_REF,
        p.ArrayType,
        setup,
        nstep = 1,
    )

    N = setup.grid.Np
    @assert all(==(N[1]), N) "small case assumes a cubic LES grid, got $N"
    iseven(N[1]) || error("LES grid $N has no Nyquist mode; the small case needs an even grid")

    # Index of the Nyquist entry in `fftfreq(N, N)`: the first negative-most wavenumber.
    inyq = N[1] ÷ 2 + 1

    # It must be the top **Z** band. `E` never passes through `∂` (`compute_QoI` uses `u_hat` for
    # `E` and `w_hat` for `Z`, and only `w_hat` comes from `curl`), so a Nyquist-plane mode in an
    # `E` band exercises nothing. Masks for `E` and `Z` over the same band are identical, which is
    # exactly why checking the wrong one would look like it passed.
    iz = findlast(q -> q[1] == "Z", p.qois)
    isnothing(iz) && error("small case needs at least one Z QoI; got $(p.qois)")

    topmask = Array(to_setup.masks[iz])
    # Modes lying on at least one Nyquist plane.
    onnyq = falses(size(topmask))
    onnyq[inyq, :, :] .= true
    onnyq[:, inyq, :] .= true
    onnyq[:, :, inyq] .= true

    nretained = count(topmask .& onnyq)
    nretained > 0 || error(
        "top Z band $(p.qois[iz]) retains no Nyquist-plane modes on a $(N[1])³ LES. " *
        "The small case would then exercise none of the #45/#46 mechanism it exists to guard. " *
        "Widen the top band or fix get_masks_and_partials.",
    )

    # The other half: ∂ must be zero on the Nyquist plane (this is what `09954be1` added).
    for (a, d) in enumerate(to_setup.∂)
        dv = Array(d)
        iszero(dv[inyq]) || error(
            "∂[$a] is nonzero at the Nyquist index $inyq. Gotcha #46: i·k_Nyq maps a real " *
            "field to a non-real one, and the zeroing in get_masks_and_partials is what " *
            "prevents it. The merge must not have reverted this.",
        )
    end

    @printf("Nyquist check: top Z band %s retains %d Nyquist-plane modes; ∂ zeroed at index %d\n",
        string(p.qois[iz]), nretained, inyq)
    nretained
end

"LES `Setup` for the small case, with no forcing (used only for mask inspection)."
function les_setup(p)
    Setup(;
        x = ntuple(α -> LinRange(p.lims[α]..., p.n_les + 1), p.D),
        p.Re,
        p.ArrayType,
        p.backend,
    )
end

# ---------------------------------------------------------------------------------------------
# Stage 0 — burn-in
# ---------------------------------------------------------------------------------------------

"""
    burn_in(p)

Burn the DNS in from rest under the OU forcing, exactly as `spinnup` in `create_ref_data.jl`
does, but without the Makie-backed `realtimeplotter` that `spinnup` registers — this case has to
run headless.

The OU chain is seeded per `Setup`, so this solve and stage 1's each start the chain at
`rng_seed`, which is what production does too (spin-up and HF reference are separate scripts).
"""
function burn_in(p)
    (; T) = p
    dns = Setup(;
        x = ntuple(α -> LinRange(p.lims[α]..., p.n_dns + 1), p.D),
        p.Re,
        p.ArrayType,
        p.backend,
        ou_bodyforce = (; p.forcing..., freeze = 10),
    )
    psolver = psolver_spectral(dns)
    ustart = vectorfield(dns)

    @printf("burn-in: %d^3 DNS, tburn = %g, %d steps\n",
        p.n_dns, p.tburn, round(Int, p.tburn / p.dt_dns))
    (; u, t), _ = solve_unsteady(;
        setup = dns,
        ustart,
        docopy = false,
        tlims = (T(0), p.tburn),
        Δt = p.dt_dns,
        processors = (; log = timelogger(; nupdate = 100)),
        psolver,
    )
    any(isnan, u) && error("burn-in produced NaNs")
    u
end

# ---------------------------------------------------------------------------------------------
# Stage 1 — DNS + filter + QoI  (mirrors 2_HF_ref.jl)
# ---------------------------------------------------------------------------------------------

"""
    stage1(p, ustart)

Run the DNS, filter it to the LES grid and record the QoI history. Returns the filtered fields
and `qoi_hist`, which together are stage 1's golden output.

This validates the solver itself: convection, diffusion, the pressure projection, the OU forcing
and the QoI evaluation. It does **not** touch the TO path — that is stage 2's job.
"""
function stage1(p, ustart)
    @printf("stage 1: DNS %d^3 -> LES %d^3, tsim = %g, %d DNS steps\n",
        p.n_dns, p.n_les, p.tsim, round(Int, p.tsim / p.dt_dns))

    data = create_ref_data(;
        p.D,
        p.Re,
        p.lims,
        p.qois,
        p.tsim,
        Δt = p.dt_dns,
        nles = [ntuple(α -> p.n_les, p.D)],
        ndns = ntuple(α -> p.n_dns, p.D),
        filters = (FaceAverage(),),
        p.ArrayType,
        p.backend,
        ou_bodyforce = (; p.forcing..., freeze = 10),
        savefreq = p.savefreq,
        plotfreq = p.plotfreq,
        ustart,
        # ⚠️ `n_checkpoints` must be given, and 0 is the only value that writes nothing.
        # `create_ref_data.jl:62` computes `0:round(nt/(n_checkpoints+1)):nt` *before* the
        # `isnothing(checkpoints)` guard in `filtersaver`, so its own documented default of
        # `nothing` throws `MethodError: +(::Nothing, ::Int64)`. That default has therefore never
        # been exercised — `2_HF_ref.jl` always passes 1. With 0 the range is `0:nt:nt`, whose
        # `[2:end-1]` slice is empty, so no checkpoint file is written and no `checkpoint_name` is
        # needed. Left as a call-site workaround rather than a source fix: the golden data has to
        # come from the pre-merge solver unmodified.
        n_checkpoints = 0,
    )

    qoi_hist = stack(data.data[1].qoi_hist)
    fields = data.data[1].u
    @printf("stage 1 done: qoi_hist %s, %d stored fields\n", string(size(qoi_hist)), length(fields))
    (; qoi_hist, fields)
end

# ---------------------------------------------------------------------------------------------
# Stage 2 — tracking run  (mirrors 3_track_ref.jl)
# ---------------------------------------------------------------------------------------------

"""
    stage2(p, s1)

Run the LES tracking the stage-1 QoI reference. Returns `q`, `dQ` and `tau` — stage 2's golden
output.

This is the half that validates the TO path: the direction vectors `V_i`, `tau`, the
`fieldsaver`-before-`qoisaver` processor ordering (gotcha #41) and the OU phase. Stage 1 exercises
none of it.
"""
function stage2(p, s1)
    (; T) = p
    nt = round(Int, p.tsim / p.dt_les)
    ncol = nt + 1
    size(s1.qoi_hist, 2) >= ncol || error(
        "stage 1 produced $(size(s1.qoi_hist, 2)) QoI columns but stage 2 needs $ncol; " *
        "check SMALL_CASE_SAVEFREQ against the LES/DNS Δt ratio",
    )

    qoi_ref = s1.qoi_hist[:, 1:ncol]
    ref_reader = Reference_reader(qoi_ref)
    ustart = s1.fields[1]

    @printf("stage 2: LES %d^3 tracking, tsim = %g, %d steps\n", p.n_les, p.tsim, nt)
    data = track_ref(;
        ustart,
        ref_reader,
        p.D,
        p.Re,
        p.lims,
        p.qois,
        nles = [ntuple(α -> p.n_les, p.D)],
        p.tsim,
        Δt = p.dt_les,
        p.ArrayType,
        p.backend,
        # freeze = 1: HIT's tracking convention, and the one gotcha #33's advance count is
        # measured at.
        ou_bodyforce = (; p.forcing..., freeze = 1),
        savefreq = 100,
    )

    @printf("stage 2 done: q %s, dQ %s, tau %s\n",
        string(size(data.q)), string(size(data.dQ)), string(size(data.tau)))
    (; q = data.q, dQ = data.dQ, tau = data.tau)
end

# ---------------------------------------------------------------------------------------------
# Run / compare
# ---------------------------------------------------------------------------------------------

"Run both stages and the Nyquist assertion. Returns everything the golden file holds."
function run_small_case(p = small_case_params())
    nnyq = assert_nyquist_in_top_band(p)
    ustart = burn_in(p)
    s1 = stage1(p, ustart)
    s2 = stage2(p, s1)
    (;
        qoi_hist = s1.qoi_hist,
        fields = s1.fields,
        q = s2.q,
        dQ = s2.dQ,
        tau = s2.tau,
        nnyq,
    )
end

"""
    compare_against_golden(cur, gold)

Bit-level comparison, per array, naming what differs.

⚠️ Bit-identity is the right bar and it is achievable: same seed, same initial condition, same
arithmetic, same machine. Any deviation at all is a port defect until proven otherwise. If
upstream legitimately changed a numerical scheme, that is a finding to escalate — **not** a
tolerance to widen.

Returns `true` if everything matches.
"""
function compare_against_golden(cur, gold)
    ok = Ref(true)

    function cmp_array(name, a, b)
        if size(a) != size(b)
            @printf("  %-10s FAIL  size %s vs golden %s\n", name, string(size(a)), string(size(b)))
            ok[] = false
            return
        end
        if a == b
            @printf("  %-10s ok    %s bit-identical\n", name, string(size(a)))
            return
        end
        ok[] = false
        d = abs.(Float64.(a) .- Float64.(b))
        scale = maximum(abs, Float64.(b))
        i = argmax(d)
        nbad = count(!iszero, d)
        @printf("  %-10s FAIL  %d of %d entries differ; max abs %.6e at %s; max rel %.6e\n",
            name, nbad, length(a), maximum(d), string(Tuple(i)),
            scale == 0 ? 0.0 : maximum(d) / scale)
    end

    println("comparing against golden data:")
    cmp_array("qoi_hist", cur.qoi_hist, gold.qoi_hist)
    cmp_array("q", cur.q, gold.q)
    cmp_array("dQ", cur.dQ, gold.dQ)
    cmp_array("tau", cur.tau, gold.tau)

    if length(cur.fields) != length(gold.fields)
        @printf("  %-10s FAIL  %d fields vs golden %d\n",
            "fields", length(cur.fields), length(gold.fields))
        ok[] = false
    else
        for i in eachindex(cur.fields)
            cmp_array("field[$i]", cur.fields[i], gold.fields[i])
        end
    end

    if cur.nnyq != gold.nnyq
        @printf("  %-10s FAIL  %d Nyquist-plane modes in top band vs golden %d\n",
            "nnyq", cur.nnyq, gold.nnyq)
        ok[] = false
    end

    ok[]
end

function generate(p = small_case_params())
    res = run_small_case(p)
    mkpath(golden_dir())
    path = golden_path()
    jldsave(path; res..., params = strip_params(p))
    @printf("golden data written: %s (%.2f MB)\n", path, filesize(path) / 1024^2)
    path
end

"Drop the non-serialisable entries (`backend`, `ArrayType`) before storing the parameters."
strip_params(p) = (;
    p.n_dns, p.n_les, p.Re, p.dt_dns, p.dt_les, p.tsim, p.tburn,
    p.qois, p.forcing, p.savefreq, p.plotfreq, p.D,
)

function check(p = small_case_params())
    path = golden_path()
    isfile(path) || error(
        "no golden data at $path. It must be generated on the PRE-merge solver and committed; " *
        "generating it now would compare the current solver against itself.",
    )
    gold = load(path)
    gp = gold["params"]
    cp = strip_params(p)
    for k in keys(cp)
        getproperty(cp, k) == gp[k] || error(
            "parameter $k is $(getproperty(cp, k)) but the golden data was made with $(gp[k]); " *
            "the comparison would be meaningless",
        )
    end

    cur = run_small_case(p)
    goldres = (;
        qoi_hist = gold["qoi_hist"],
        fields = gold["fields"],
        q = gold["q"],
        dQ = gold["dQ"],
        tau = gold["tau"],
        nnyq = gold["nnyq"],
    )
    ok = compare_against_golden(cur, goldres)
    if ok
        println("\nSMALL CASE PASSED — every array bit-identical to the golden data.")
    else
        println("\nSMALL CASE FAILED. Do not widen a tolerance; this is a port defect or a")
        println("deliberate upstream numerics change, and both are escalations.")
    end
    ok
end

function main(args)
    mode = isempty(args) ? "check" : args[1]
    if mode == "generate"
        generate()
        true
    elseif mode == "check"
        check()
    else
        error("unknown mode $mode; use \"generate\" or \"check\"")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS) ? 0 : 1)
end
