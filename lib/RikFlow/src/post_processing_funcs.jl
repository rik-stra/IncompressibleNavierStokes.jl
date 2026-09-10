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

function getspectrum(u; setup, npoint = 100, a = typeof(setup.Re)(1 + sqrt(5)) / 2)


    (; dimension, xp, Ip, Np) = setup.grid
    T = eltype(xp[1])
    D = dimension()

    (; inds, κ, K) = IncompressibleNavierStokes.spectral_stuff(setup; npoint, a)

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