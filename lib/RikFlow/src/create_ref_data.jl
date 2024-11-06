include("plotter.jl")


function lesdatagen(dnsobs, Φ, les, compression, to_setup, n_plot)
    Φu = zero.(Φ(dnsobs[].u, les, compression))
    p = zero(Φu[1])
    div = zero(p)
    ΦF = zero.(Φu)
    FΦ = zero.(Φu)
    c = zero.(Φu)
    #results = (; u = fill(Array.(dnsobs[].u), 0), c = fill(Array.(dnsobs[].u), 0))
    results = (; u = fill(Array.(dnsobs[].u), 0), qoi_hist = fill(zeros(typeof(les.Re),0), 0))
    temp = nothing
    on(dnsobs) do (; u, t, n)
        Φ(Φu, u, les, compression)
        apply_bc_u!(Φu, t, les)
        #Φ(ΦF, F, les, compression)
        #momentum!(FΦ, Φu, temp, t, les)
        #apply_bc_u!(FΦ, t, les; dudt = true)
        #project!(FΦ, les; psolver, div, p)
        #for α = 1:length(u)
        #    c[α] .= ΦF[α] .- FΦ[α]
        #end
        u_hat = get_u_hat(Φu, les)
        w_hat = get_w_hat_from_u_hat(u_hat, to_setup)
        q = compute_QoI(u_hat, w_hat, to_setup,les)
        #println("QoI: ", q)
        push!(results.qoi_hist, Array(q))

        n % n_plot == 0 || return
        push!(results.u, Array.(Φu))
        
        #push!(results.c, Array.(c))
    end
    results
end

"""
Save filtered DNS data.
"""
filtersaver(dns, les, filters, compression, to_setup_les; nupdate = 1, n_plot = 1000) =
    processor(
        (results, state) -> (; results..., comptime = time() - results.comptime),
    ) do state
        comptime = time()
        (; x) = dns.grid
        T = eltype(x[1])
        #F = zero.(state[].u)
        #div = zero(state[].u[1])
        #p = zero(state[].u[1])
        dnsobs = Observable((; state[].u, state[].t, state[].n))
        data = [
            lesdatagen(dnsobs, Φ, les[i], compression[i], to_setup_les[i], n_plot) for
            i = 1:length(les), Φ in filters
        ]
        results = (; data, t = zeros(T, 0), comptime)
        #temp = nothing
        on(state) do (; u, t, n)
            
            #momentum!(F, u, temp, t, dns)
            #apply_bc_u!(F, t, dns; dudt = true)
            #project!(F, dns; psolver = psolver_dns, div, p)
            
            #push!(results.t, t)
            if n % nupdate == 0
                dnsobs[] = (; u, t, n)
            end
        end
        state[] = state[] # Save initial conditions
        results
    end

"""
Create filtered DNS data.
"""
function create_ref_data(;
    D = 3,
    Re = 2e3,
    lims = ntuple(α -> (typeof(Re)(0), typeof(Re)(1)), D),
    qois = [["Z", 0, 4], ["E", 0, 4], ["Z", 5, 10], ["E", 5, 10]],
    nles = [ntuple(α -> 32, D)],
    ndns = ntuple(α -> 64, D),
    filters = (FaceAverage(),),
    tburn = nothing,
    tsim = typeof(Re)(0.1),
    Δt = typeof(Re)(1e-4),
    create_psolver = psolver_spectral,
    savefreq = 1,
    plotfreq = 1000,
    ArrayType = Array,
    ustart = nothing,
    ou_bodyforce = nothing,
    kwargs...,
)
    T = typeof(Re)

    compression = [ndns[1] ÷ nles[1] for nles in nles]
    for (c, n) in zip(compression, nles), α = 1:D
        @assert c * n[α] == ndns[α]
    end

    # Build setup and assemble operators
    dns = Setup(;
        x = ntuple(α -> LinRange(lims[α]..., ndns[α] + 1), D),
        Re,
        ArrayType,
        ou_bodyforce,
        kwargs...,
    )

    les = [
        Setup(;
            x = ntuple(α -> LinRange(lims[α]..., nles[α] + 1), D),
            Re,
            ArrayType,
            kwargs...,
        ) for nles in nles
    ]

    # Number of time steps to save
    nt = round(Int, tsim / Δt)
    Δt = tsim / nt

    # Build TO operators
    to_setup_les = [
        RikFlow.TO_Setup(; qois, 
        qoi_refs_location= "none", 
        to_mode = :CREATE_REF, 
        ArrayType, 
        setup = les[i], 
        nstep=nt) for i in 1:length(nles)]

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    psolver = create_psolver(dns)



    # datasize = Base.summarysize(filtered) / 1e6
    datasize_fields =
        length(filters) *
        (nt ÷ savefreq + 1) *
        sum(prod.(nles)) *
        D *
        length(bitstring(zero(T))) / 8 / 1e6
    datasize_QoIs = 
        length(qois) *
        (nt + 1) *
        length(nles) *
        length(bitstring(zero(T))) / 8 / 1e6
    datasize = datasize_fields + datasize_QoIs
    @info "Generating $datasize Mb of filtered DNS data"


    _dns = dns
    _les = les

    @info "Solving DNS"
    # Solve DNS and store filtered quantities
    (; u, t), outputs = solve_unsteady(;
        setup = _dns,
        ustart,
        docopy = false,
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            f = filtersaver(
                _dns,
                _les,
                filters,
                compression,
                to_setup_les;
                nupdate = savefreq,
                n_plot = plotfreq,
            ),
            #vort = realtimeplotter(;
            #    setup = _dns,
            #    plot = vortplot,
            #    nupdate = 10,
            #    displayupdates = true,
            #    displayfig = true,
            #),
            log = timelogger(; nupdate = 100),
        ),
        psolver,
    )

    # Store result for current IC
    outputs.f
end

function spinnup(;
    D = 3,
    Re = 1e3,
    lims = ntuple(α -> (typeof(Re)(0), typeof(Re)(1)), D),
    ndns = ntuple(α -> 64, D),
    tburn = typeof(Re)(0.1),
    create_psolver = psolver_spectral,
    savefreq = 100,
    ArrayType = Array,
    ou_bodyforce = nothing,
    kwargs...,
)
    T = typeof(Re)


    # Build setup and assemble operators
    dns = Setup(;
        x = ntuple(α -> LinRange(lims[α]..., ndns[α] + 1), D),
        Re,
        ArrayType,
        ou_bodyforce,
    )

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    psolver = create_psolver(dns)

    ustart = vectorfield(dns);
    any(u -> any(isnan, u), ustart) && @warn "Initial conditions contain NaNs"

    _dns = dns

    # Solve burn-in DNS
    
    @info "Solving burn-in DNS"
    (; u, t), outputs =
        solve_unsteady(;
        #method = RKMethods.Wray3(),
        setup = _dns, ustart, tlims = (T(0), tburn),
        docopy = false,
        kwargs...,
        processors = (;
            log = timelogger(; nupdate = 100),
            ehist = realtimeplotter(;
                setup = _dns,
                plot = energy_history_plot,
                nupdate = 200,
                displayupdates = false,
                displayfig = false,
            ),
            # espec = realtimeplotter(;
            #     setup= _dns,
            #     plot = energy_spectrum_plot,
            #     nupdate = 200,
            #     displayupdates = false,
            #     displayfig = false,
            # ),
            #states = fieldsaver(setup = _dns, nupdate = savefreq),
        #     vort = realtimeplotter(;
        #     setup = _dns,
        #     plot = vortplot,
        #     nupdate = savefreq,
        #     displayupdates = true,
        #     displayfig = true,
        # ),
        ),
        psolver)


    # Store result for current IC
    u, outputs.ehist
end
