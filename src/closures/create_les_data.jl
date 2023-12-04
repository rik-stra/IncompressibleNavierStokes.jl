# Body force
function gaussian_force(
    x,
    y;
    σ = eltype(x)(0.05),
    A = eltype(x)(0.002),
    rng = Random.default_rng(),
)
    T = eltype(x)
    Lx = x[end] - x[1]
    Ly = y[end] - y[1]
    xc = x[1] + rand(rng, T) * Lx
    yc = y[1] + rand(rng, T) * Ly
    σx = σ * Lx
    σy = σ * Ly
    ϕ = T(2π) * rand(rng, T)
    f = sum(
        [
            A * exp(-(x - xc - lx)^2 / 2σx^2 - (y - yc - ly)^2 / 2σy^2) for
            y ∈ y[2:end], x ∈ x[2:end], ly ∈ (-Ly, T(0), Ly), lx ∈ (-Lx, T(0), Lx)
        ];
        dims = (3, 4),
    )[
        :,
        :,
        1,
        1,
    ]
    force = cat(sin(ϕ) * f, cos(ϕ) * f; dims = 3)
    force = reshape(force, :)
    force = force .- sum(force) / length(force)
    force
end

function lesdatagen(dnsobs, les, compression, pressure_solver)
    Φu = zero.(face_average(dnsobs[].u, les, compression))
    q = zero(pressure(pressure_solver, Φu, dnsobs[].t, les))
    M = zero(q)
    ΦF = zero.(Φu)
    FΦ = zero.(Φu)
    GΦ = zero.(Φu)
    c = zero.(Φu)
    results = (; u = fill(Array.(dnsobs[].u), 0), c = fill(Array.(dnsobs[].u), 0))
    on(dnsobs) do (; u, F, t)
        face_average!(Φu, u, les, compression)
        apply_bc_u!(Φu, t, les)
        face_average!(ΦF, F, les, compression)
        momentum!(FΦ, Φu, t, les)
        apply_bc_u!(FΦ, t, les; dudt = true)
        divergence!(M, FΦ, les)
        @. M *= les.grid.Ω
        poisson!(pressure_solver, q, M)
        apply_bc_p!(q, t, les)
        pressuregradient!(GΦ, q, les)
        for α = 1:length(u)
            FΦ[α] .-= GΦ[α]
            c[α] .= ΦF[α] .- FΦ[α]
        end
        push!(results.u, Array.(Φu))
        push!(results.c, Array.(c))
    end
    results
end

filtersaver(dns, les, compression, pressure_solver; nupdate = 1) =
    processor() do state
        (; dimension, x) = dns.grid
        T = eltype(x[1])
        D = dimension()
        F = zero.(state[].u)
        G = zero.(state[].u)
        dnsobs = Observable((; state[].u, F, state[].t))
        data = [
            lesdatagen(dnsobs, les[i], compression[i], pressure_solver[i]) for
            i = 1:length(les)
        ]
        results = (;
            t = fill(zero(eltype(x[1])), 0),
            u = [d.u for d in data],
            c = [d.c for d in data],
        )
        on(state) do (; u, p, t, n)
            n % nupdate == 0 || return
            momentum!(F, u, t, dns)
            pressuregradient!(G, p, dns)
            for α = 1:D
                F[α] .-= G[α]
            end
            push!(results.t, t)
            dnsobs[] = (; u, F, t)
        end
        state[] = state[] # Save initial conditions
        results
    end

"""
    create_les_data(
        T;
        D = 2,
        Re = T(2_000),
        lims = (T(0), T(1)),
        nles = [64],
        ndns = 256,
        tburn = T(0.1),
        tsim = T(0.1),
        Δt = T(1e-4),
        ArrayType = Array,
    )


Create filtered DNS data.
"""
function create_les_data(
    T;
    D = 2,
    Re = T(2_000),
    lims = (T(0), T(1)),
    nles = [64],
    ndns = 256,
    tburn = T(0.1),
    tsim = T(0.1),
    Δt = T(1e-4),
    savefreq = 1,
    ArrayType = Array,
    ic_params = (;),
)
    compression = @. ndns ÷ nles
    @assert all(@.(compression * nles == ndns))

    # Build setup and assemble operators
    dns = Setup(ntuple(α -> LinRange(lims..., ndns + 1), D)...; Re, ArrayType)
    les = [
        Setup(ntuple(α -> LinRange(lims..., nles + 1), D)...; Re, ArrayType) for
        nles in nles
    ]

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    pressure_solver = SpectralPressureSolver(dns)
    pressure_solver_les = SpectralPressureSolver.(les)

    # Number of time steps to save
    nt = round(Int, tsim / Δt)
    Δt = tsim / nt

    # datasize = Base.summarysize(filtered) / 1e6
    datasize =
        (nt ÷ savefreq + 1) * sum(nles .^ D) * 3 * 2 * length(bitstring(zero(T))) / 8 / 1e6
    @info "Generating $datasize Mb of LES data"

    # Initial conditions
    u₀, p₀ = random_field(dns, T(0); pressure_solver, ic_params...)

    # Random body force
    # force_dns =
    #     gaussian_force(xdns...) +
    #     gaussian_force(xdns...) +
    #     # gaussian_force(xdns...) +
    #     # gaussian_force(xdns...) +
    #     gaussian_force(xdns...)
    # force_dns = zero.(u₀)
    # force_les = face_average(force_dns, les, compression)

    _dns = dns
    _les = les
    # _dns = (; dns..., bodyforce = force_dns)
    # _les = (; les..., bodyforce = force_les)

    # Solve burn-in DNS
    (; u, p, t), outputs = solve_unsteady(_dns, u₀, p₀, (T(0), tburn); Δt, pressure_solver)

    # Solve DNS and store filtered quantities
    (; u, p, t), outputs = solve_unsteady(
        _dns,
        u,
        p,
        (T(0), tsim);
        Δt,
        processors = (;
            f = filtersaver(
                _dns,
                _les,
                compression,
                pressure_solver_les;
                nupdate = savefreq,
            ),
        ),
        pressure_solver,
    )

    # Store result for current IC
    outputs[1]
end

"""
    create_io_arrays(data, setups)

Create ``(\\bar{u}, c)`` pairs for training.
"""
function create_io_arrays(data, setups)
    nsample = length(data)
    ngrid = length(setups)
    nt = length(data[1].u[1]) - 1
    T = eltype(data[1].u[1][1][1])
    map(1:ngrid) do ig
        (; dimension, N, Iu) = setups[ig].grid
        D = dimension()
        u = zeros(T, (N .- 2)..., D, nt + 1, nsample)
        c = zeros(T, (N .- 2)..., D, nt + 1, nsample)
        ifield = ntuple(Returns(:), D)
        for is = 1:nsample, it = 1:nt+1, α = 1:D
            copyto!(view(u, ifield..., α, it, is), view(data[is].u[ig][it][α], Iu[α]))
            copyto!(view(c, ifield..., α, it, is), view(data[is].c[ig][it][α], Iu[α]))
        end
        (; u = reshape(u, (N .- 2)..., D, :), c = reshape(c, (N .- 2)..., D, :))
    end
end
