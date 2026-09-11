# Does the HF reference compute AND store the QoIs only every `savefreq`-th DNS step?
#
# The storage rate is easy to read off `length(qoi_hist)`. The *computation* rate is the question
# that matters for cost: filtering 512^3 -> 64^3 and taking three FFTs every step instead of every
# tenth would be a large waste that no output would reveal.
#
# `lesdatagen`'s `on(dnsobs)` handler does the filter, get_u_hat, get_w_hat and compute_QoI and
# then pushes - all in one block - so counting how often the *filter* is invoked counts how often
# the QoIs are computed. A filter that counts its own calls is therefore a direct probe.

using Printf
using IncompressibleNavierStokes
using RikFlow

mutable struct CountingFilter <: RikFlow.AbstractFilter
    inner::FaceAverage
    calls::Int
end
CountingFilter() = CountingFilter(FaceAverage(), 0)
function (f::CountingFilter)(v, u, setup_les, comp)
    f.calls += 1
    f.inner(v, u, setup_les, comp)
end

const T = Float32

function run_case(; nstep, savefreq, plotfreq)
    cf = CountingFilter()
    dt = T(1e-3)
    data = create_ref_data(;
        D = 3,
        Re = T(2000),
        lims = ((T(0), T(1)), (T(0), T(1)), (T(0), T(1))),
        qois = [["Z", 0, 2], ["E", 0, 2], ["Z", 3, 4], ["E", 3, 4]],
        tsim = dt * nstep,
        Δt = dt,
        nles = [(8, 8, 8)],
        ndns = (16, 16, 16),
        filters = (cf,),
        ArrayType = Array,
        backend = IncompressibleNavierStokes.CPU(),
        savefreq,
        plotfreq,
        n_checkpoints = 0,
        ustart = nothing,
    )
    d = data.data[1]
    (; calls = cf.calls, nq = length(d.qoi_hist), nf = length(d.u))
end

function main()
    nstep = 100
    @printf("%d DNS steps, 16^3 -> 8^3\n\n", nstep)
    @printf("%-9s %-9s %10s %10s %10s %12s\n",
        "savefreq", "plotfreq", "QoI calls", "stored q", "stored u", "expected")
    ok = true
    for (s, p) in ((10, 50), (1, 50), (5, 100))
        r = run_case(; nstep, savefreq = s, plotfreq = p)
        expect = fld(nstep, s) + 1
        good = r.calls == expect == r.nq
        ok &= good
        @printf("%-9d %-9d %10d %10d %10d %12d %s\n",
            s, p, r.calls, r.nq, r.nf, expect, good ? "ok" : "MISMATCH")
    end
    println()
    if ok
        println("PASS: the QoIs are computed exactly as often as they are stored, once every")
        println("  savefreq-th DNS step (plus one at n = 0 from the initialiser). The filter and")
        println("  the three FFTs are not run on the other steps.")
        println()
        println("At production settings - 400000 DNS steps, savefreq = 10 - that is 40001 QoI")
        println("  evaluations, matching the archive's qoi_hist width.")
    else
        println("FAIL: computation and storage rates disagree, or neither matches savefreq.")
    end
    ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main() ? 0 : 1)
end
