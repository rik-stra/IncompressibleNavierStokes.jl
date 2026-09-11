if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
    using .SymmetryClosure                             #src
    using .IncompressibleNavierStokes                  #src
end   

function ks_dist(data1::Vector,data2::Vector)
    sort!(data1)
    sort!(data2)
    n1 = length(data1)
    n2 = length(data2)
    if min(n1, n2) == 0
        Error("Data passed to ks_dist must not be empty")
    end
    data_all = cat(data1, data2, dims=1)
    cdf1 = map(x->searchsortedlast(data1, x),data_all) / n1
    cdf2 = map(x->searchsortedlast(data2, x),data_all) / n2
    cddiffs = abs.(cdf1 - cdf2)
    argmaxS = argmax(cddiffs)
    loc_maxS = data_all[argmaxS]
    d_1 = cddiffs[argmaxS]
    return d_1, loc_maxS
end


# The plotting half of this file now lives in `ext/RikFlowMakieExt.jl`. It moved because
# `using LaTeXStrings` / `using CairoMakie` at the top of this file were **module-level** -- this
# file is `include`d by `RikFlow.jl:72` -- so every load of RikFlow, including in a GPU batch job
# that plots nothing, pulled in the whole Makie stack. Only one caller needs it
# (`exp_square_HIT/figs_paper.jl`, which loads CairoMakie itself), and the extension activates
# for exactly that case. Same pattern as IncompressibleNavierStokes' own Makie extension.
function energy_spectra_comparison end

"""
    rf_spectral_stuff(setup; npoint, a)

Wavenumber shells for [`getspectrum`](@ref). RikFlow's own, carried across the upstream merge.

⚠️ This is **not** `IncompressibleNavierStokes.spectral_stuff`, and the difference is deliberate.
Ours bins linearly in 3D — `[κ - 1/2, κ + 1/2)` — which is what gives the k^(-5/3) slope, and
dyadically in 2D for k^(-3); upstream's (`src/spectral.jl`) puts every integer wavenumber in its
own shell and takes `kmax` rather than `npoint`/`a`. Swapping to upstream's would silently re-bin
every energy spectrum figure in the paper, so the local version moved here at the merge instead of
being dropped. Returns `(; inds, κ, K)`.
"""
function rf_spectral_stuff(setup; npoint = 100, a = typeof(setup.Re)(1 + sqrt(5)) / 2)
    (; dimension, xp, Ip, xlims) = setup
    T = eltype(xp[1])
    D = dimension()
    domain_length = [(xlims[d][2] - xlims[d][1]) for d in 1:D]
    K = size(Ip) .÷ 2
    k = zeros(T, K)
    if D == 2
        kx = reshape(0:K[1]-1, :)./domain_length[1]
        ky = reshape(0:K[2]-1, 1, :)./domain_length[2]
        @. k = sqrt(kx^2 + ky^2)
    elseif D == 3
        kx = reshape(0:K[1]-1, :) #./domain_length[1]
        ky = reshape(0:K[2]-1, 1, :) #./domain_length[2]
        kz = reshape(0:K[3]-1, 1, 1, :) #./domain_length[3]
        @. k = sqrt(kx^2 + ky^2 + kz^2)
    end
    k = reshape(k, :)

    # Sum or average wavenumbers between k and k+1
    kmax = minimum([(K[d]) for d in 1:D]) 
    isort = sortperm(k)
    ksort = k[isort]

    IntArray = typeof(similar(xp[1], Int, 0))
    inds = IntArray[]

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    # logκ = LinRange(T(0), log(T(kmax) - 1), npoint)
    logκ = LinRange(T(0), log(T(kmax)), npoint)
    # logκ = LinRange(log(a), log(T(kmax) / a), npoint)
    # logκ = LinRange(T(0), log(T(kmax)), npoint)
    κ = exp.(logκ)
    κ = sort(unique(round.(Int, κ)))
    npoint = length(κ)

    for i = 1:npoint
        if D == 2
            # Dyadic binning - this gives the k^-3 slope in 2D
            jstart = findfirst(≥(κ[i] / a), ksort)
            jstop = findfirst(≥(κ[i] * a), ksort)
        elseif D == 3
            # Linear binning - this gives the k^-5/3 slope in 3D
            jstart = findfirst(≥(κ[i] - T(0.5)), ksort)
            jstop = findfirst(≥(κ[i] + T(0.5)), ksort)
            # jstart = findfirst(≥(κ[i] - T(1.01)), ksort)
            # jstop = findfirst(≥(κ[i] + T(1.01)), ksort)
        end

        # jstart = findfirst(≥(κ[i] - T(1.01)), ksort)
        # jstop = findfirst(≥(κ[i] + T(1.01)), ksort)
        isnothing(jstop) && (jstop = length(ksort) + 1)
        jstop -= 1
        push!(inds, adapt(IntArray, isort[jstart:jstop]))
    end

    (; inds, κ, K)
end


function getspectrum(u; setup, npoint = 100, a = typeof(setup.Re)(1 + sqrt(5)) / 2)


    (; dimension, xp, Ip, Np) = setup
    T = eltype(xp[1])
    D = dimension()

    (; inds, κ, K) = rf_spectral_stuff(setup; npoint, a)

    # Energy
    uhat = similar(xp[1], Complex{T}, Np)
    # up = interpolate_u_p(state[].u, setup)
    _ehat = zeros(T, length(κ))

    up = u

    e = sum(eachslice(up; dims = D + 1)) do u
            copyto!(uhat, view(u, Ip))
            fft!(uhat)
            uhathalf = view(uhat, ntuple(α -> 1:K[α], D)...)
            abs2.(uhathalf) ./ (2 * prod(Np)^2)
    end
    e = map(i -> sum(view(e, i)), inds)
        # e = max.(e, eps(T)) # Avoid log(0)
    copyto!(_ehat, e)
    

    (; _ehat, κ)
end