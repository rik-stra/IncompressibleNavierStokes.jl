"""
# Makie extension for RikFlow

Implements the plotting functions RikFlow declares but leaves empty, in the same way
IncompressibleNavierStokes does (see `ext/IncompressibleNavierStokesMakieExt.jl` and the note at
`src/processors.jl:350-353`). Loaded only when Makie is in the environment, through GLMakie or
CairoMakie.

🔴 **Why this exists.** `src/post_processing_funcs.jl` used to `using LaTeXStrings` and
`using CairoMakie` at module level, and `RikFlow.jl:72` includes that file -- so *every* load of
RikFlow dragged in Makie, CairoMakie, Cairo and their trees. On a GPU batch job that plots nothing
this is pure precompilation cost, and on a Julia newer than the pinned Makie it is a hard failure:
`MakieCore` reads `Core.TypeName.mt`, a field removed in Julia 1.12, so RikFlow could not load at
all. One caller needs these functions -- `exp_square_HIT/figs_paper.jl` -- and it loads CairoMakie
itself, which activates this extension.
"""
module RikFlowMakieExt

using Makie
using RikFlow
using RikFlow: getspectrum

import RikFlow: energy_spectra_comparison, rf_energy_spectrum_plot

function energy_spectra_comparison(
    models_states,
    labels;
    setup,
    sloperange = [0.6, 0.9],
    slopeoffset = 1.3,
    scale_numbers = nothing,
    linestyles = [:solid,],
    kwargs...,
)
    
    if size(linestyles) == 1
        linestyles = size(labels)*linestyles
    end

    (; dimension, xp, Np, xlims) = setup
    T = eltype(xp[1])
    D = dimension()
    dx = xlims[1][2]-xlims[1][1]
    Δx = dx / Np[1]
    ehats = []
    ks = []
    k=0
    for model_data in models_states
        temp_ehats = []
        for d in model_data
            if (d isa NamedTuple)
                #d = (;u = d, t=0.0)
                d = d.u
            end
            ehat, k = getspectrum(d; setup, kwargs...)
            push!(temp_ehats, ehat)
        end

        if length(temp_ehats)>1
            s_ehats = stack(temp_ehats)
            ehat_ave = mean(s_ehats, dims=2) 
            
        else
            ehat_ave = temp_ehats[1]
        end
        push!(ehats, ehat_ave)
        push!(ks, k)
    end

    kmax = maximum(maximum.(ks))
    # Build inertial slope above energy
    krange = kmax .^ sloperange
    slope, slopelabel = D == 2 ? (-T(3), L"$k^{-3}$") : (-T(5 / 3), L"$k^{-5/3}$")

    τ = 2π |> T
    C_K = 1.58 |> T
    kpoints = sloperange
    slopepoints = @. C_K * scale_numbers.ϵ^T(2 / 3) * (τ * kpoints)^slope*slopeoffset
    
    l_points = kpoints
    
    inertia = [Point2f(l_points[1], slopepoints[1]), Point2f(l_points[2], slopepoints[2])]



    # Nice ticks

    xlabel = "Wave number ||k||"
    logmax = round(Int, log2(kmax + 1))
    xticks = (T(2) .^ (0:logmax))

    fig = Figure(size=(600,400))
    fig[1,1] = ax = Axis(
        fig;
        xlabel,
        ylabel = "E(||k||)",
        xscale = log10,
        yscale = log10,
        #limits = (dx/kmax, dx, T(1e-15), T(1)),
    )
    for (ehat, κ, label, linestyle, colid) in zip(ehats, ks, labels, linestyles, [8,3,1,2])
        if label == "Ref"
            lines!(ax, κ, reshape(ehat,(:)); label = label, linewidth = 5, color = :black)
        else
            lines!(ax, κ, reshape(ehat,(:)); label = label, linewidth = 3, linestyle = linestyle, color = Cycled(colid))
        end
    end
    lines!(ax, inertia; label = slopelabel, linestyle = :dot, linewidth = 3, color = Cycled(2))
    axislegend(ax; position = :lb)


    #xlims!(ax,Δx*0.7, dx)
    ax.xticks = xticks

    # autolimits!(ax)
    #on(e -> autolimits!(ax), ehat)
    #autolimits!(ax)
    fig
end


"""
    rf_energy_spectrum_plot(state; setup, sloperange, slopeoffset, v_lines, scale_numbers,
                            plot_wavelength, figure_size, plot_n_spectra, kwargs...)

RikFlow's energy spectrum figure, carried across the upstream merge.

⚠️ **Not** `IncompressibleNavierStokes.energy_spectrum_plot`. This fork had extended that function
with `v_lines`, `scale_numbers`, `plot_wavelength`, `plot_n_spectra` and multi-setup support;
upstream's is still the plain two-keyword version, and the merge took upstream's `ext/MakieExt.jl`
wholesale. Rather than lose the paper's figure code, the extended version moved here under a
RikFlow name, alongside `energy_spectra_comparison`, which already did the same kind of work.

Ported at the merge: `setup[1].grid` flattened, `scale_numbers.ϵ` renamed to `.diss` to match
`turbulence_statistics`, and `observespectrum` replaced by RikFlow's `getspectrum` so the shell
binning stays the one the published spectra used.

🔴 Untested. Makie is not in the RikFlow test environment, so this was ported by inspection. It is
figure code, outside the compute path the small HIT case guards.
"""
function rf_energy_spectrum_plot(
    state;
    setup,
    sloperange = [0.6, 0.9],
    slopeoffset = 1.3,
    v_lines = nothing,
    scale_numbers = nothing,
    plot_wavelength = false,
    figure_size = (600, 400),
    plot_n_spectra = 1,
    kwargs...,
)
    #state isa Observable || (state = Observable(state))

    
    (; dimension, xp, Np, xlims) = setup[1]
    T = eltype(xp[1])
    D = dimension()
    dx = xlims[1][2]-xlims[1][1]
    Δx = dx / Np[1]

    ehat_l = []; κ_l = []
    for i in 1:plot_n_spectra
        # `observespectrum` is IncompressibleNavierStokes', and upstream re-signatured the
        # `spectral_stuff` behind it (`kmax` instead of `npoint`/`a`) as well as re-binning it.
        # RikFlow's `getspectrum` keeps the original binning - see `rf_spectral_stuff`.
        (; _ehat, κ) = getspectrum(state[i].u; setup = setup[i], kwargs...)
        ehat = _ehat
        push!(ehat_l, ehat)
        push!(κ_l, κ)
    end

    kmax = maximum(κ_l[1])
    # Build inertial slope above energy
    #krange = kmax .^ sloperange
    slope, slopelabel = D == 2 ? (-T(3), L"$k^{-3}$") : (-T(5 / 3), L"$k^{-5/3}$")

    τ = 2π |> T
    C_K = 1.58 |> T
    kpoints = sloperange
    if isnothing(scale_numbers)
        slopepoints = @. C_K ^T(2 / 3) * (τ * kpoints)^slope*slopeoffset
    else
        slopepoints = @. C_K * scale_numbers.diss^T(2 / 3) * (τ * kpoints)^slope*slopeoffset
    end
    
    if plot_wavelength
        l_points = dx./kpoints
    else
        l_points = kpoints
    end
    inertia = [Point2f(l_points[1], slopepoints[1]), Point2f(l_points[2], slopepoints[2])]


    # inertia = lift(ehat) do ehat
    #     (m, i) = findmax(ehat ./ κ .^ slope)
    #     slopeconst = m
    #     dk = exp(log(kmax) * 0.5)
    #     # kpoints = κ[i] / dk, κ[i] * dk
    #     kpoints = κ[i] / (dk / 3), min(κ[i] * dk, kmax)
    #     @show kpoints, dk, i
    #     slopepoints = @. slopeoffset * slopeconst * kpoints^slope
    #     [Point2f(kpoints[1], slopepoints[1]), Point2f(kpoints[2], slopepoints[2])]
    # end

    # Nice ticks
    
    if plot_wavelength
        xlabel = "Wave length"
    else
        xlabel = "Wave number ||k||"
        logmax = round(Int, log2(kmax + 1))
        xticks = (T(2) .^ (0:logmax))
    end

    fig = Figure(size=figure_size)
    fig[1,1] = ax = Axis(
        fig;
        xlabel,
        ylabel = "Kinetic energy",
        xscale = log10,
        yscale = log10,
        #limits = (dx/kmax, dx, T(1e-15), T(1)),
    )
    ls = [:solid, :dash]
    for i in 1:plot_n_spectra
        if plot_wavelength
            l = dx./(κ_l[i])
        else
            l = κ_l[i]
        end
        lines!(ax, l, ehat_l[i]; label = "t = $(state[i].t)", linewidth = 2, linestyle = ls[i])
    end
    
    
    lines!(ax, inertia; label = slopelabel, linestyle = :dash, linewidth = 2, color = Cycled(3))
    axislegend(ax; position = :lb)
    if !isnothing(v_lines)
        vlines!(ax, v_lines; linestyle = :dash)
    end

    #xlims!(ax,Δx*0.7, dx)
    #ylims!(ax, 1e-15, 1)
    #xlims!(ax, dx, Δx*0.7)
    if plot_wavelength
        ax.xreversed = true
    else
        ax.xticks = xticks
    end
    # autolimits!(ax)
    #on(e -> autolimits!(ax), ehat)
    #autolimits!(ax)
    fig
end


end
