# this file contains standard setups for the high fidelity HIT simulations

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
    backend,
    ustart = nothing,
    ou_bodyforce = nothing,
    n_checkpoints = nothing,
    checkpoint_name = nothing,
    kwargs...,
)
    T = typeof(Re)

    compression = [ndns[1] ÷ nles[1] for nles in nles]
    for (c, n) in zip(compression, nles), α = 1:D
        @assert c * n[α] == ndns[α]
    end

    # Build setup and assemble operators
    dns = rf_setup(;
        x = ntuple(α -> LinRange(lims[α]..., ndns[α] + 1), D),
        Re,
        ArrayType,
        backend,
        kwargs...,
    )

    # Forcing is no longer part of the setup: it is the right-hand side and its cache.
    force!, force_cache = if isnothing(ou_bodyforce)
        navierstokes!, nothing
    else
        ou_navierstokes!, ou_force_cache(dns; ou_bodyforce...)
    end

    if isnothing(ustart)
        ustart = vectorfield(dns)
    end

    les = [
        rf_setup(;
            x = ntuple(α -> LinRange(lims[α]..., nles[α] + 1), D),
            Re,
            ArrayType,
            backend,
            kwargs...,
        ) for nles in nles
    ]

    # Number of time steps to save
    nt = round(Int, tsim / Δt)
    Δt = tsim / nt
    checkpoints= 0:round(nt/(n_checkpoints+1)):nt
    checkpoints = checkpoints[2:end-1]

    # Build TO operators
    to_setup_les = [
        RikFlow.TO_Setup(; qois, 
        to_mode = :CREATE_REF, 
        ArrayType, 
        setup = les[i], 
        nstep=nt) for i in 1:length(nles)]

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    psolver = create_psolver(dns)



    _dns = dns
    _les = les

    @info "Solving DNS"
    # Solve DNS and store filtered quantities
    (; u, t), outputs = solve_unsteady(;
        setup = _dns,
        # 🔴 Pin the time stepper. Upstream changed `solve_unsteady`'s default from
        # `RKMethods.RK44` to `LMWray3` at the merge, so every call that relied on the default
        # silently changed time integrator - including this one, which produces the HF reference.
        # Left implicit, the 19.3 GPU-hour DNS re-run would have used a different scheme from the
        # archive it has to reproduce. Pinned rather than trusted.
        method = RKMethods.RK44(; T = eltype(ustart)),
        start = (; u = ustart),
        force!,
        force_cache,
        params = rf_params(_dns),
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
                checkpoints,
                checkpoint_name,
            ),
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
    backend,
    lims = ntuple(α -> (typeof(Re)(0), typeof(Re)(1)), D),
    ndns = ntuple(α -> 64, D),
    tburn = typeof(Re)(0.1),
    create_psolver = psolver_spectral,
    savefreq = 100,
    ou_bodyforce = nothing,
    checkpoint_file_name = "./u",
    Δt = nothing,
    ArrayType = Array,
    kwargs...,
)
    T = typeof(Re)

    # Build setup and assemble operators
    dns = rf_setup(;
        x = ntuple(α -> LinRange(lims[α]..., ndns[α] + 1), D),
        Re,
        backend,
        ArrayType,
    )

    # Forcing is the right-hand side now, not a setup field.
    force!, force_cache = if isnothing(ou_bodyforce)
        navierstokes!, nothing
    else
        ou_navierstokes!, ou_force_cache(dns; ou_bodyforce...)
    end

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    psolver = create_psolver(dns)

    ustart = vectorfield(dns);
    any(u -> any(isnan, u), ustart) && @warn "Initial conditions contain NaNs"

    _dns = dns
    # Solve burn-in DNS
    nt = round(Int, tburn / Δt)
    n_checkpoints = 9
    checkpoints= 0:round(nt/(n_checkpoints+1)):nt
    @info "Solving burn-in DNS"
    (; u, t), outputs =
        solve_unsteady(;
        #method = RKMethods.Wray3(),
        # Upstream changed solve_unsteady's default method from RKMethods.RK44 to LMWray3 at the
        # merge; pinned so this keeps the pre-merge integrator.
        method = RKMethods.RK44(; T = eltype(ustart)),
        setup = _dns, start = (; u = ustart), tlims = (T(0), tburn),
        force!,
        force_cache,
        params = rf_params(_dns),
        docopy = false,
        Δt,
        kwargs...,
        processors = (;
            log = timelogger(; nupdate = 50),
            ehist = realtimeplotter(;
                setup = _dns,
                plot = energy_history_plot,
                nupdate = 200,
                displayupdates = false,
                displayfig = false,
            ),
            cp = checkpointer(checkpoints, checkpoint_file_name, _dns)
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
