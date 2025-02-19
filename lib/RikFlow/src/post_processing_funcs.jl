if false                                               #src
    include("../src/RikFlow.jl")                  #src
    #include("../NeuralClosure/src/NeuralClosure.jl")   #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
    using .SymmetryClosure                             #src
    #using .NeuralClosure                               #src
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


function energy_spectra_comparison(
    models_states,
    labels;
    setup,
    sloperange = [0.6, 0.9],
    slopeoffset = 1.3,
    scale_numbers = nothing,
    kwargs...,
)
    

    (; dimension, xp, Np, xlims) = setup.grid
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
    for (ehat, κ, label) in zip(ehats, ks, labels)
        if label == "Ref"
            lines!(ax, κ, reshape(ehat,(:)); label = label, linewidth = 4, color = :black)
        else
            lines!(ax, κ, reshape(ehat,(:)); label = label, linewidth = 2)
        end
    end
    lines!(ax, inertia; label = slopelabel, linestyle = :dash, linewidth = 2, color = Cycled(2))
    axislegend(ax; position = :lb)


    #xlims!(ax,Δx*0.7, dx)
    ax.xticks = xticks

    # autolimits!(ax)
    #on(e -> autolimits!(ax), ehat)
    #autolimits!(ax)
    fig
end

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