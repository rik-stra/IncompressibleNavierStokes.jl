# Disk accounting for the HF reference run.
#
# Why this exists. `create_ref_data` accumulates every stored field in host memory for the whole
# run and writes it all at the end, and it writes a checkpoint that carries the full DNS field
# *plus* everything accumulated so far. At 512^3 Float64 those are 3.0 GiB and 2.6 GiB, so the run
# needs several times more free space than its final output, and it finds that out at hour 39 of
# 40 if nobody checks first. `ref_data_storage` computes the requirement from the same parameters
# the run uses, and `check_output_space` refuses to start without it.

"""
    ref_data_storage(; ndns, nles, tsim, Δt, savefreq, plotfreq, n_checkpoints, D, T, nqois)

Bytes that a [`create_ref_data`](@ref) run will write and hold, as a NamedTuple.

Fields: `nt`, `nfields`, `nqoi`, `storeevery`, `field` (one stored LES field), `dnsfield`,
`final` (the output file), `checkpoints` (a vector of per-file sizes), `peak` (final + all
checkpoints, since checkpoints are never removed) and `hostram` (what `results` occupies while the
run is going).

🔴 `storeevery` is `lcm(savefreq, plotfreq)`, not `plotfreq`. `filtersaver` only fires its inner
observable on multiples of `savefreq` and only stores a field when *that* `n` is also a multiple of
`plotfreq`, so a `plotfreq` that is not a multiple of `savefreq` silently stores far fewer fields
than it names — at `savefreq = 10, plotfreq = 1500` the real interval is 3000. The stored count
follows the lcm so this shows up in the table instead of in the output file.
"""
function ref_data_storage(;
    ndns,
    nles,
    tsim,
    Δt,
    savefreq = 1,
    plotfreq = 1000,
    n_checkpoints = 0,
    D = 3,
    T = Float64,
    nqois = 6,
)
    ndns = ndns isa Tuple ? ndns : ntuple(Returns(ndns), D)
    nles = nles isa Tuple ? nles : ntuple(Returns(nles), D)
    el = sizeof(T)
    nt = round(Int, tsim / Δt)

    # One ghost layer on each side: `vectorfield` allocates `(N..., D)` with `N = n + 2`, and
    # `Array(Φu)` writes all of it, ghosts included.
    field = prod(nles .+ 2) * D * el
    dnsfield = prod(ndns .+ 2) * D * el

    storeevery = lcm(savefreq, plotfreq)
    nfields = nt ÷ storeevery + 1      # n = 0 is stored as well
    nqoi = nt ÷ savefreq + 1
    qoibytes = nqoi * nqois * el

    final = nfields * field + qoibytes

    # Checkpoint i sits at n_i and carries the whole DNS field plus everything `results` holds at
    # that moment, so the files get bigger as the run goes on.
    ns = n_checkpoints <= 0 ? Int[] :
         [round(Int, i * nt / (n_checkpoints + 1)) for i = 1:n_checkpoints]
    checkpoints = [dnsfield + (n ÷ storeevery + 1) * field + (n ÷ savefreq + 1) * nqois * el
                   for n in ns]

    (;
        nt,
        storeevery,
        nfields,
        nqoi,
        field,
        dnsfield,
        final,
        checkpoint_steps = ns,
        checkpoints,
        peak = final + sum(checkpoints; init = 0),
        hostram = nfields * field + qoibytes,
    )
end

"Bytes as a human-readable string."
humanbytes(b) =
    b >= 2^30 ? string(round(b / 2^30, digits = 2), " GiB") :
    b >= 2^20 ? string(round(b / 2^20, digits = 2), " MiB") :
    string(round(b / 2^10, digits = 2), " KiB")

"""
    check_output_space(dirs, needed; margin = 1.15, hard = true)

Compare `needed` bytes against the free space on each of `dirs`, and stop the run if it does not
fit with `margin` to spare.

`dirs` may be one path or several; identical filesystems are only counted once, and directories
that do not exist yet are resolved to their nearest existing parent. `hard = false` warns instead
of erroring, for a probe that wants to report rather than refuse.

⚠️ Free space is not a reservation. Another job on the same project filesystem can take it while
this run is going; the margin is what covers the ordinary case, not a busy one.
"""
function check_output_space(dirs, needed; margin = 1.15, hard = true)
    dirs = dirs isa AbstractString ? [dirs] : collect(dirs)
    seen = Set{Tuple{UInt64,UInt64}}()
    ok = true
    for d in dirs
        p = abspath(d)
        while !ispath(p) && dirname(p) != p
            p = dirname(p)
        end
        st = try
            Base.diskstat(p)
        catch err
            @warn "could not read free space; skipping the check for this path" path = p err
            continue
        end
        # One filesystem, one report: `output/` and `output/checkpoints/` are normally the same
        # mount and would otherwise be counted as two independent budgets.
        key = (UInt64(st.total), UInt64(st.available))
        key in seen && continue
        push!(seen, key)
        want = needed * margin
        @info "free space" path = p available = humanbytes(st.available) needed =
            humanbytes(needed) with_margin = humanbytes(want)
        if st.available < want
            ok = false
            msg =
                "not enough free space at $p: $(humanbytes(st.available)) available, " *
                "$(humanbytes(want)) needed ($(humanbytes(needed)) plus a $(round((margin-1)*100))% margin). " *
                "Free space, point the output elsewhere, or lower n_checkpoints / raise plotfreq."
            hard ? error(msg) : @warn msg
        end
    end
    ok
end

"Print the storage table for a `create_ref_data` run. Returns the `ref_data_storage` result."
function report_ref_data_storage(s; label = "")
    println()
    println("storage for this run" * (isempty(label) ? "" : " ($label)"))
    println("  steps                    $(s.nt)")
    println("  QoI samples              $(s.nqoi)")
    println("  stored fields            $(s.nfields)  (one every $(s.storeevery) steps)")
    println("  one stored field         $(humanbytes(s.field))")
    println("  one DNS field            $(humanbytes(s.dnsfield))")
    println("  final output file        $(humanbytes(s.final))")
    for (n, b) in zip(s.checkpoint_steps, s.checkpoints)
        println("  checkpoint at n = $n     $(humanbytes(b))")
    end
    println("  peak disk (kept at once)  $(humanbytes(s.peak))")
    println("  host RAM held during run  $(humanbytes(s.hostram))")
    println()
    s
end

export ref_data_storage, check_output_space, report_ref_data_storage, humanbytes
