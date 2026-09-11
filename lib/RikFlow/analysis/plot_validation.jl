# Figures for the ordinal-0 validation run.
#
# Usage:
#   julia --startup-file=no --project=analysis analysis/plot_validation.jl
#
# Reads the validation members from `exp_square_HIT/output/D6/`, the archived LinReg1 ensemble and
# the HF reference from `analysis/data/`, and writes two PNGs into `analysis/figures/`.
#
# 🔑 Why two figures and not one. The trajectory plot shows *that* the run separates from the
# archive; it cannot show *when*, because by the time the lines are visibly apart the question has
# already been decided. The second figure is the one that carries the result: deviation against
# column on a log axis, where the replayed warm-up, the round-off growth through it and the jump at
# the first sampled step are all legible. `compare_validation`'s verdict is a reading of that curve.

using CairoMakie
using JLD2
using Printf
using Statistics

const HERE = @__DIR__
const FIGS = joinpath(HERE, "figures")
const DATA = joinpath(HERE, "data")
const D6 = normpath(joinpath(HERE, "..", "exp_square_HIT", "output", "D6"))
const ARCH = joinpath(DATA, "online_LinReg1_frozen_qois.jld2")
const REF = joinpath(DATA, "hf_reference_tsim100.0_qois.jld2")

const LABELS = ["Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]"]
const IZ1632 = 5
const DT = 2.5e-3

CairoMakie.activate!(; type = "png", px_per_unit = 2)

# Same palette as `plot_paper4.jl`: colour-blind safe, and the reference is the darker, thinner
# line so it reads as truth rather than as a third model.
const C_REF = RGBf(0.15, 0.15, 0.18)
const C_RUN = RGBf(0.00, 0.45, 0.70)      # blue -- this validation run
const C_ARCH = RGBf(0.84, 0.37, 0.00)     # vermillion -- paper 2's archived replicas
const C_WARM = RGBf(0.93, 0.93, 0.90)

"""
    load_all()

The three trajectory sets on one column axis.

🔴 **The alignment is not a choice here, it is `score_d6.jl`'s.** For the validation IC `n_k = 0`,
and `truth_column(0, ℓ) == forecast_column(ℓ)`, so run column `c` is reference column `c`. That
identity is asserted below on the first column rather than assumed: the archived run starts from the
reference's own field, so `q_ref[:, 1]` and the archive's `q[:, 1]` must agree to round-off. If that
assertion ever fires, the reference is not the one these runs were tracking and no panel below means
anything.
"""
function load_all()
    isdir(D6) || error("no D6 output at $D6")
    pat = r"^d6_valid_ic(\d+)_m(\d+)\.jld2$"
    files = sort([(parse(Int, m[2]), joinpath(D6, f))
                  for f in readdir(D6) for m in (match(pat, f),) if m !== nothing])
    isempty(files) && error("no validation runs in $D6 (run `tools/run_d6.jl 0`)")
    runs = [Float64.(load(p, "q")) for (_, p) in files]
    nwarm = load(files[1][2], "nwarm")

    isfile(ARCH) || error("no archived ensemble at $ARCH (run analysis/extract_archive.jl)")
    a = load(ARCH)
    arch = [Float64.(q) for q in a["q"]]

    isfile(REF) || error("no HF reference at $REF (run analysis/extract_archive.jl)")
    ref = Float64.(load(REF, "q_ref"))

    n = minimum(size.(runs, 2))
    sd = vec(std(view(arch[1], :, 1:n); dims = 2))
    d0 = maximum(abs.(view(ref, :, 1) .- view(arch[1], :, 1)) ./ sd)
    d0 < 1e-2 || error("reference column 1 disagrees with the archive's by $d0 of a sd; the " *
                       "n_k = 0 identity `truth_column(0, l) == forecast_column(l)` does not hold " *
                       "for this reference and the panels would be misaligned")
    @printf("loaded %d members, %d archived replicas, reference; n = %d columns, nwarm = %d\n",
            length(runs), length(arch), n, nwarm)
    @printf("  reference vs archive at column 1: %.2e of a sd (alignment check)\n", d0)
    return (; runs, arch, ref, n, nwarm, sd, members = first.(files))
end

"""
    fig_trajectories(d)

Six panels, one per QoI: this run's members, the archived replicas, and the HF reference.

The warm-up band is shaded because it is a different experiment from the rest of the panel -- there
the sampler is replaying the archive's own `dQ`, so agreement is required rather than informative,
and the free-running comparison only begins at its right edge.
"""
function fig_trajectories(d)
    t = (0:(d.n - 1)) .* DT
    fig = Figure(size = (1150, 720))
    for (i, lab) in pairs(LABELS)
        r, c = fldmod1(i, 3)
        ax = Axis(fig[r, c]; title = lab, xlabel = r == 2 ? "t [TU]" : "",
                  ylabel = c == 1 ? "QoI" : "", titlesize = 13)
        vspan!(ax, 0, d.nwarm * DT; color = C_WARM)
        for q in d.arch
            lines!(ax, t, view(q, i, 1:d.n); color = (C_ARCH, 0.55), linewidth = 0.9)
        end
        for q in d.runs
            lines!(ax, t, view(q, i, 1:d.n); color = (C_RUN, 0.55), linewidth = 0.9)
        end
        lines!(ax, t, view(d.ref, i, 1:d.n); color = C_REF, linewidth = 1.4)
        i == IZ1632 && text!(ax, 0.02, 0.04; text = "gotcha #45: different quantity",
                             space = :relative, fontsize = 9, color = C_REF)
    end
    Legend(fig[3, 1:3],
           [PolyElement(color = C_WARM),
            LineElement(color = C_RUN, linewidth = 2),
            LineElement(color = C_ARCH, linewidth = 2),
            LineElement(color = C_REF, linewidth = 2)],
           ["replayed warm-up (dQ pinned to the archive)",
            "D6 validation members (this code, GPU)",
            "archived LinReg1 replicas (paper 2)",
            "HF reference"];
           orientation = :horizontal, framevisible = false, labelsize = 11,
           tellheight = true, tellwidth = false)
    Label(fig[0, 1:3],
          "Ordinal 0: the archived runs' own initial condition, re-run by the D6 path";
          fontsize = 15, font = :bold)
    p = joinpath(FIGS, "d6_validation_trajectories.png")
    save(p, fig)
    println("wrote ", p)
    return p
end

"""
    fig_divergence(d)

Per-column deviation between member `m` and archived replica `m`, in units of each QoI's own sd, on
a log axis -- the figure the verdict is actually read from.

Three things are marked because each is a separate claim:

  * the **warm-up edge**, past which the sampler runs and separation is expected;
  * the **gate**, `1e-2` over the warm-up, which is the only thing `compare_validation` votes on;
  * the **saturation band**, the rms between two *archived* replicas of the same configuration.
    Anything at that level is as different as two correct runs are from each other, which is what
    the retired whole-run criterion was mistaking for a defect.
"""
function fig_divergence(d)
    cols = 1:d.n
    fig = Figure(size = (1150, 640))
    sat = let s = Float64[]
        for i in 1:length(d.arch), j in (i + 1):length(d.arch)
            a, b = d.arch[i], d.arch[j]
            push!(s, maximum(vec(sqrt.(mean(abs2, view(a, :, 1:d.n) .- view(b, :, 1:d.n);
                                            dims = 2))) ./ d.sd))
        end
        (minimum(s), maximum(s))
    end
    for (i, lab) in pairs(LABELS)
        r, c = fldmod1(i, 3)
        ax = Axis(fig[r, c]; title = lab, yscale = log10,
                  xlabel = r == 2 ? "column" : "", ylabel = c == 1 ? "|run - archive| / sd" : "",
                  titlesize = 13)
        hspan!(ax, sat[1], sat[2]; color = (C_ARCH, 0.25))
        vspan!(ax, 1, d.nwarm; color = C_WARM)
        hlines!(ax, [1e-2]; color = C_REF, linestyle = :dash, linewidth = 1)
        for (q, m) in zip(d.runs, d.members)
            m <= length(d.arch) || continue
            e = abs.(view(q, i, cols) .- view(d.arch[m], i, cols)) ./ d.sd[i]
            # a floor so exact agreement is drawable on a log axis; well below anything meaningful
            lines!(ax, cols, max.(e, 1e-9); color = (C_RUN, 0.7), linewidth = 0.9)
        end
        ylims!(ax, 1e-9, 30)
        i == 1 && text!(ax, 0.03, 0.9; text = "gate 1e-2", space = :relative, fontsize = 9,
                        color = C_REF)
    end
    Legend(fig[3, 1:3],
           [PolyElement(color = C_WARM),
            LineElement(color = C_RUN, linewidth = 2),
            LineElement(color = C_REF, linewidth = 2, linestyle = :dash),
            PolyElement(color = (C_ARCH, 0.13))],
           ["replayed warm-up: dQ bit-identical, q moves under the solver and tau alone",
            "member vs its archived replica",
            "gate (warm-up window only)",
            "saturation: two archived replicas of the same configuration"];
           orientation = :horizontal, nbanks = 2, framevisible = false, labelsize = 11,
           tellheight = true, tellwidth = false)
    Label(fig[0, 1:3],
          "Where the validation run parts company with the archive, and why that is not a defect";
          fontsize = 15, font = :bold)
    p = joinpath(FIGS, "d6_validation_divergence.png")
    save(p, fig)
    println("wrote ", p)
    return p
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(FIGS)
    d = load_all()
    fig_trajectories(d)
    fig_divergence(d)
end
