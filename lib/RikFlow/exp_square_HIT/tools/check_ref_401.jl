#=
    check_ref_401.jl — full-scale acceptance test for the merge and for the HF DNS re-run.

Paper 2's archived HF reference stores **401 filtered 64³ velocity fields** at
`data_train.data[1].u` (every 0.25 TU over 100 TU, 3.29 MB each). Those fields are a free
regression asset, and the reason is worth stating precisely:

🔑 **The archived DNS *trajectory* is correct and is unaffected by `09954be1`.** `∂` enters only
the QoI evaluation and the TO direction vectors, and a *reference* run has no TO feedback — it is
a plain forced DNS with a filter hanging off it. So a re-run of the reference on the merged solver
must reproduce those 401 fields **bit for bit**, whatever happened to the Nyquist convention.

🔴 **The `qoi_hist` in the same file is a different matter, and that is the point.** The QoIs are
computed through `∂` (`Z` goes via `curl`, `E` does not), so gotchas #45/#46 predict a specific,
falsifiable pattern:

| QoI | archive vs re-run | why |
|---|---|---|
| `E[0,6]`, `E[7,15]`, `E[16,32]` | match | `E` never touches `∂` |
| `Z[0,6]`, `Z[7,15]` | match | their masks exclude `|k| ≥ 15.5`, so they never meet Nyquist |
| `Z[16,32]` | **differs, ~1.06e-3 relative** | its mask retains the Nyquist shell while `∂` is zeroed there |

So this script gates on the fields and *reports* the QoIs against that prediction. A `Z[16,32]`
mismatch is expected and is not a failure. A mismatch anywhere else is.

⚠️ Do not "fix" a `Z[16,32]` mismatch by touching the masks. Gotcha #46 settled that: the
pre-`09954be1` convention was wrong, not merely different, and the masks stay as they are.

## Usage

    RIKFLOW_ARCHIVE=<dir> RIKFLOW_NEW_REFERENCE=<file.jld2> \
        julia --startup-file=no --project=lib/RikFlow \
        lib/RikFlow/exp_square_HIT/tools/check_ref_401.jl

⚠️ Both files are ~1.4 GB and JLD2 stores each as one compound dataset, so `fields` cannot be
skipped on read. Expect a large resident set and a slow load; this is not a script to run in a
loop.
=#

using JLD2
using Printf
using Statistics

const REFERENCE_FILE = "data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"

archive_path() = joinpath(
    get(
        ENV,
        "RIKFLOW_ARCHIVE",
        raw"C:\Users\rik\Documents\julia_code\INS_paper_summer2025\lib\RikFlow\exp_square_HIT\output\paper_data_HIT",
    ),
    REFERENCE_FILE,
)

new_reference_path() = get(ENV, "RIKFLOW_NEW_REFERENCE", "")

"QoI labels in the order `2_HF_ref.jl` defines them."
const QOI_LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]

"The one QoI gotcha #45 predicts will differ, by index into `QOI_LABELS`."
const NYQUIST_QOI = 5

"""
    load_reference(path)

Return `(; fields, qoi_hist)` from an HF reference file, whichever of the two shapes it is in.
"""
function load_reference(path)
    @printf("loading %s (%.2f GB)\n", path, filesize(path) / 1024^3)
    t0 = time()
    data = load(path, "data_train")
    @printf("  loaded in %.1f s\n", time() - t0)
    d = data.data[1]
    fields = d.u
    qoi_hist = stack(d.qoi_hist)
    @printf("  %d fields, qoi_hist %s\n", length(fields), string(size(qoi_hist)))
    (; fields, qoi_hist)
end

"""
    compare_fields(new, old)

Bit-level comparison of the 401 filtered fields. Returns `(ok, nmismatch, firstbad)`.

Bit-identity is the bar. Same seed, same initial condition, same arithmetic — a reference run is
deterministic, and the Nyquist change does not reach it.
"""
function compare_fields(new, old)
    if length(new) != length(old)
        @printf("FIELDS FAIL: %d fields against the archive's %d\n", length(new), length(old))
        return (false, -1, 0)
    end
    nbad = 0
    firstbad = 0
    worst = 0.0
    for i in eachindex(old)
        a, b = new[i], old[i]
        if size(a) != size(b)
            @printf("  field %d: size %s against %s\n", i, string(size(a)), string(size(b)))
            nbad += 1
            firstbad == 0 && (firstbad = i)
            continue
        end
        a == b && continue
        nbad += 1
        firstbad == 0 && (firstbad = i)
        d = maximum(abs, Float64.(a) .- Float64.(b))
        s = maximum(abs, Float64.(b))
        rel = s == 0 ? 0.0 : d / s
        worst = max(worst, rel)
        nbad <= 5 && @printf("  field %d differs: max abs %.6e, max rel %.6e\n", i, d, rel)
    end
    if nbad == 0
        @printf("FIELDS PASS: all %d filtered fields bit-identical to the archive\n", length(old))
    else
        @printf("FIELDS FAIL: %d of %d differ, first at %d, worst relative %.6e\n",
            nbad, length(old), firstbad, worst)
    end
    (nbad == 0, nbad, firstbad)
end

"""
    compare_qois(new, old)

Report the QoI histories against gotcha #45's prediction: five match, `Z[16,32]` does not.

Returns `true` when the observed pattern **is** the predicted one — that is the success
condition, not "everything matches".
"""
function compare_qois(new, old)
    ncol = min(size(new, 2), size(old, 2))
    if size(new, 2) != size(old, 2)
        @printf("note: comparing the first %d of %d / %d columns\n",
            ncol, size(new, 2), size(old, 2))
    end
    a = Float64.(new[:, 1:ncol])
    b = Float64.(old[:, 1:ncol])

    @printf("%-10s %14s %14s %10s\n", "QoI", "max rel dev", "rms rel dev", "verdict")
    ok = true
    for i in axes(b, 1)
        s = sqrt(mean(abs2, b[i, :]))
        d = abs.(a[i, :] .- b[i, :])
        maxrel = s == 0 ? 0.0 : maximum(d) / s
        rmsrel = s == 0 ? 0.0 : sqrt(mean(abs2, d)) / s

        # Float32 round-off floor; the five unaffected QoIs sat at ~1.9e-7 when #45 was measured.
        matched = maxrel < 1e-5
        expected_to_differ = (i == NYQUIST_QOI)

        verdict = if expected_to_differ && !matched
            "as predicted"
        elseif expected_to_differ && matched
            ok = false
            "UNEXPECTED MATCH"
        elseif !expected_to_differ && matched
            "ok"
        else
            ok = false
            "FAIL"
        end
        @printf("%-10s %14.6e %14.6e %10s\n", QOI_LABELS[i], maxrel, rmsrel, verdict)
    end

    if ok
        println("QOIS PASS: the deviation pattern is exactly what gotchas #45/#46 predict —")
        println("  five QoIs at the Float32 round-off floor, Z[16,32] offset by the Nyquist change.")
    else
        println("QOIS FAIL: the deviation pattern is NOT the predicted one. Either a QoI that")
        println("  cannot touch Nyquist has moved (a port defect), or Z[16,32] has stopped")
        println("  differing (the masks or the zeroing changed). Both are escalations.")
    end
    ok
end

function main()
    newpath = new_reference_path()
    isempty(newpath) && error(
        "set RIKFLOW_NEW_REFERENCE to the re-run reference file. This script compares a newly " *
        "generated HF reference against paper 2's archive; it does not generate one.",
    )
    isfile(newpath) || error("no such file: $newpath")
    oldpath = archive_path()
    isfile(oldpath) || error("no archive at $oldpath; set RIKFLOW_ARCHIVE")

    old = load_reference(oldpath)
    new = load_reference(newpath)

    println()
    fields_ok, _, _ = compare_fields(new.fields, old.fields)
    println()
    qois_ok = compare_qois(new.qoi_hist, old.qoi_hist)
    println()

    if fields_ok && qois_ok
        println("401-FIELD CHECK PASSED.")
        println("  The re-run reproduces the archived DNS trajectory bit for bit, and the QoI")
        println("  deviations are confined to Z[16,32] exactly as the Nyquist analysis predicts.")
        true
    else
        println("401-FIELD CHECK FAILED. Do not widen a tolerance.")
        false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main() ? 0 : 1)
end
