#=
    check_ref_401.jl — full-scale acceptance test for the merge and for the HF DNS re-run.

Paper 2's archived HF reference stores **401 filtered 64³ velocity fields** at
`data_train.data[1].u` (every 0.25 TU over 100 TU, 3.29 MB each). Those fields are a free
regression asset, and the reason is worth stating precisely:

🔑 **The archived DNS *trajectory* is correct and is unaffected by `09954be1`.** `∂` enters only
the QoI evaluation and the TO direction vectors, and a *reference* run has no TO feedback — it is
a plain forced DNS with a filter hanging off it. So a re-run of the reference on the merged solver
must reproduce those 401 fields **to Float32 round-off**, whatever happened to the Nyquist
convention. (It was *bit for bit* until the upstream merge rewrote the operator contractions; see
`GATE_EPS32` for the measurement and the decision.)

🔴 **The `qoi_hist` in the same file is a different matter, and that is the point.** The QoIs are
computed through `∂` (`Z` goes via `curl`, `E` does not), so gotchas #45/#46 predict a specific,
falsifiable pattern:

| QoI | archive vs re-run | why |
|---|---|---|
| `E[0,6]`, `E[7,15]`, `E[16,32]` | match | `E` never touches `∂` |
| `Z[0,6]`, `Z[7,15]` | match | their masks exclude `|k| ≥ 15.5`, so they never meet Nyquist |
| `Z[16,32]` | **differs, ~1.06e-3 relative** | its mask retains the Nyquist shell while `∂` is zeroed there |

So this script gates on the fields, at the round-off bound, and *reports* the QoIs against that
prediction. A `Z[16,32]` mismatch is expected and is not a failure. A mismatch anywhere else is.

⚠️ Do not "fix" a `Z[16,32]` mismatch by touching the masks. Gotcha #46 settled that: the
pre-`09954be1` convention was wrong, not merely different, and the masks stay as they are.

## Usage

    # the initial-condition half: no time stepping, no GPU, runnable today
    julia --startup-file=no --project=lib/RikFlow \
        lib/RikFlow/exp_square_HIT/tools/check_ref_401.jl ic

    # the full 401-field comparison: needs the DNS re-run to have produced a reference
    RIKFLOW_ARCHIVE=<dir> RIKFLOW_NEW_REFERENCE=<file.jld2> \
        julia --startup-file=no --project=lib/RikFlow \
        lib/RikFlow/exp_square_HIT/tools/check_ref_401.jl

⚠️ Both files are ~1.4 GB and JLD2 stores each as one compound dataset, so `fields` cannot be
skipped on read. Expect a large resident set and a slow load; this is not a script to run in a
loop.
=#

using IncompressibleNavierStokes
using JLD2
using Printf
using RikFlow
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
Float32 round-off bound for the field comparison, in units of `eps(Float32)`.

🔴 **Was bit-identity until 2026-09-11; changed by Rik's decision for the same measured reason as
`small_case.jl`.** A reference run is deterministic, and the Nyquist change of `09954be1` genuinely
does not reach the DNS trajectory — both halves of the original argument still hold. What broke the
bit-level bar is the upstream merge: `operators.jl` was rewritten into a mathematically equivalent
but differently-ordered contraction form, so the arithmetic order changed and the last bits move.
Measured on the 64³ small case, filtered fields drift from 2.2 to 7.3 eps(Float32) over 1000 DNS
steps; 16 leaves headroom without admitting anything a real defect could hide in, since a port
defect moves these by O(1e-2) or worse.

⚠️ 401 fields at 0.25 TU apart span 100 TU, far longer than the small case's 0.25 TU, so the drift
here may well exceed 7.3 eps32. If it exceeds the bound, that is a **finding to report with the
measured growth curve**, not a reason to raise the number — the question then is whether round-off
has amplified chaotically over 100 TU, which is a physical statement about the reference, not a
tolerance question.
"""
const GATE_EPS32 = 16

"""
    compare_fields(new, old)

Compare the 401 filtered fields at the Float32 round-off bound. Returns `(ok, nmismatch, worst)`.
"""
function compare_fields(new, old)
    if length(new) != length(old)
        @printf("FIELDS FAIL: %d fields against the archive's %d\n", length(new), length(old))
        return (false, -1, 0.0)
    end
    nbad = 0
    firstbad = 0
    worst = 0.0
    worst_i = 0
    nident = 0
    for i in eachindex(old)
        a, b = new[i], old[i]
        if size(a) != size(b)
            @printf("  field %d: size %s against %s\n", i, string(size(a)), string(size(b)))
            nbad += 1
            firstbad == 0 && (firstbad = i)
            continue
        end
        if a == b
            nident += 1
            continue
        end
        d = maximum(abs, Float64.(a) .- Float64.(b))
        s = maximum(abs, Float64.(b))
        rel = s == 0 ? 0.0 : d / s
        if rel > worst
            worst = rel
            worst_i = i
        end
        if rel / eps(Float32) > GATE_EPS32
            nbad += 1
            firstbad == 0 && (firstbad = i)
            nbad <= 5 && @printf("  field %d over bound: max rel %.4e = %.1f eps32\n",
                i, rel, rel / eps(Float32))
        end
    end
    @printf("  %d of %d fields bit-identical; worst relative %.4e = %.1f eps32 at field %d\n",
        nident, length(old), worst, worst / eps(Float32), worst_i)
    if nbad == 0
        @printf("FIELDS PASS: all %d filtered fields within %d eps(Float32) of the archive\n",
            length(old), GATE_EPS32)
    else
        @printf("FIELDS FAIL: %d of %d exceed %d eps32, first at %d\n",
            nbad, length(old), GATE_EPS32, firstbad)
    end
    (nbad == 0, nbad, worst)
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

"""
    check_ic()

The half of this check that needs **no time stepping**, and therefore no GPU and no re-run.

🔑 The archive's first stored field is the filtered *initial condition*: `data_train.data[1].u[1]`
at `t = 0`, which is `u_start_spinnup_512_...jld2` pushed through `FaceAverage` at compression 8.
Reproducing it costs zero solver steps, so it can run on a workstation today — while the other 400
fields need the 19.3 GPU-hour re-run.

What it actually tests is not trivial. It exercises, at full 512³ → 64³ production scale:

  - `FaceAverage`, whose `setup_les` destructuring changed in the merge;
  - `rf_setup` at production size, including `Re` and `ArrayType`;
  - `get_masks_and_partials`, `get_u_hat`, `curl` and `compute_QoI` — the whole `∂` path;
  - and hence the #45/#46 prediction itself, on the real archived field rather than the 64³ toy.

🔴 The QoI half is the sharp one. Gotcha #45 measured, on this very field, that restoring the
pre-`09954be1` behaviour reproduces the archived `Z[16,32]` to 1.9e-7 while the current code gives
**1.058e-3**. That number is a fixed, published property of the archive, so this is a check with a
known answer: five QoIs at the Float32 round-off floor and `Z[16,32]` at ~1.06e-3. Anything else
means the merge moved the QoI path.

⚠️ Reads ~3 GB (a 1.63 GB initial condition plus the 1.39 GB archive, which JLD2 cannot partially
load). Run it once, not in a loop.
"""
function check_ic()
    oldpath = archive_path()
    isfile(oldpath) || error("no archive at $oldpath; set RIKFLOW_ARCHIVE")
    icpath = get(
        ENV,
        "RIKFLOW_SPINNUP",
        joinpath(dirname(oldpath), "u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2"),
    )
    isfile(icpath) || error("no spin-up initial condition at $icpath; set RIKFLOW_SPINNUP")

    T = Float32
    n_dns, n_les = 512, 64
    Re = T(2000)
    lims = ((T(0), T(1)), (T(0), T(1)), (T(0), T(1)))
    qois = [["Z", 0, 6], ["E", 0, 6], ["Z", 7, 15], ["E", 7, 15], ["Z", 16, 32], ["E", 16, 32]]

    @printf("loading spin-up IC %s (%.2f GB)\n", icpath, filesize(icpath) / 1024^3)
    ustart = load(icpath, "u_start")
    ustart isa Tuple && (ustart = stack(ustart))
    ustart = Array{T}(ustart)
    @printf("  ustart %s\n", string(size(ustart)))

    dns = rf_setup(; x = ntuple(a -> LinRange(lims[a]..., n_dns + 1), 3), Re)
    les = rf_setup(; x = ntuple(a -> LinRange(lims[a]..., n_les + 1), 3), Re)
    comp = n_dns ÷ n_les

    @info "filtering 512^3 -> 64^3"
    phi = vectorfield(les)
    FaceAverage()(phi, ustart, les, comp)
    IncompressibleNavierStokes.apply_bc_u!(phi, T(0), les)

    to_setup = RikFlow.TO_Setup(; qois, to_mode = :CREATE_REF, ArrayType = Array,
        setup = les, nstep = 1)
    u_hat = RikFlow.get_u_hat(phi, les, to_setup)
    w_hat = RikFlow.get_w_hat_from_u_hat(u_hat, to_setup)
    q = RikFlow.compute_QoI(u_hat, w_hat, to_setup, les)

    old = load_reference(oldpath)
    println()
    ok, _, _ = compare_fields([phi], [old.fields[1]])

    println("\nQoIs of the filtered initial condition, against the archive's first column:")
    qok = compare_qois(reshape(Float64.(q), :, 1), reshape(Float64.(old.qoi_hist[:, 1]), :, 1))

    println()
    if ok && qok
        println("IC CHECK PASSED — the merged filter and QoI path reproduce the archive at full")
        println("  512^3 -> 64^3 scale, with the deviation confined to Z[16,32] as #45/#46 predict.")
        println("  The remaining 400 fields need the DNS re-run; nothing else here does.")
    else
        println("IC CHECK FAILED — this needs no time stepping, so a failure here is the filter,")
        println("  the masks or the QoI evaluation, not chaos and not round-off accumulation.")
    end
    ok && qok
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
        println("  The re-run reproduces the archived DNS trajectory to Float32 round-off, and")
        println("  the QoI deviations are confined to Z[16,32] as the Nyquist analysis predicts.")
        true
    else
        println("401-FIELD CHECK FAILED. Report the growth curve; do not raise the bound.")
        false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # `ic` runs only the initial-condition half, which needs no time stepping and so no GPU and
    # no re-run. With no argument, the full 401-field comparison runs and needs
    # RIKFLOW_NEW_REFERENCE.
    mode = isempty(ARGS) ? "full" : ARGS[1]
    exit((mode == "ic" ? check_ic() : main()) ? 0 : 1)
end
