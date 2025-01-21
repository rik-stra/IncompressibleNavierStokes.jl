function track_ref(;
    ustart,
    ref_reader,
    D = 3,
    Re = 1e3,
    lims = ntuple(α -> (typeof(Re)(0), typeof(Re)(1)), D),
    qois = [["Z", 0, 4], ["E", 0, 4], ["Z", 5, 10], ["E", 5, 10]],
    nles = [ntuple(α -> 32, D)],
    tsim = typeof(Re)(0.1),
    Δt = typeof(Re)(1e-4),
    create_psolver = psolver_spectral,
    savefreq = 1,
    ArrayType = Array,
    backend,
    ou_bodyforce = none,
    tracking_noise = 0.0,
    kwargs...,
)
T = typeof(Re)

# Build setup and assemble operators

setup = 
Setup(;
    x = ntuple(α -> LinRange(lims[α]..., nles[1][α] + 1), D),
    Re,
    ArrayType,
    ou_bodyforce,
    backend,
)

# Number of time steps to save
nt = round(Int, tsim / Δt)

to_setup_les = RikFlow.TO_Setup(; 
        qois, 
        to_mode = :TRACK_REF, 
        ArrayType, 
        setup, 
        nstep=nt,
        time_series_method = ref_reader,
        tracking_noise = tracking_noise)

psolver = create_psolver(setup)

# Solve
@info "Solving LF sim (track ref)"
println("setup.ou ", setup.ou_bodyforce)
(; u, t), outputs =
        solve_unsteady(; setup, 
        ustart,
        method = TOMethod(; to_setup = to_setup_les), 
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 100),
            fields = fieldsaver(; setup, nupdate = savefreq), # by calling this BEFORE qoisaver, we also save the field at t=0!
            qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1),
            #vort = realtimeplotter(;
            #    setup,
            #    plot = vortplot,
            #    nupdate = 10,
            #    displayupdates = true,
            #    displayfig = true,
            #),
        ),
        psolver)
q = stack(outputs.qoihist)
dQ = to_setup_les.outputs.dQ
tau = to_setup_les.outputs.tau
q_star = to_setup_les.outputs.q_star
fields = outputs.fields
return (;dQ, tau, q, q_star, fields)
end


function online_sgs(;
    ustart,
    time_series_method,
    D = 3,
    Re = 1e3,
    lims = ntuple(α -> (typeof(Re)(0), typeof(Re)(1)), D),
    qois = [["Z", 0, 4], ["E", 0, 4], ["Z", 5, 10], ["E", 5, 10]],
    nles = [ntuple(α -> 32, D)],
    tsim = typeof(Re)(0.1),
    Δt = typeof(Re)(1e-4),
    create_psolver = psolver_spectral,
    savefreq = 1,
    ArrayType = Array,
    backend,
    ou_bodyforce = none,
    kwargs...,
)
T = typeof(Re)

# Build setup and assemble operators

setup = 
Setup(;
    x = ntuple(α -> LinRange(lims[α]..., nles[1][α] + 1), D),
    Re,
    ArrayType,
    backend,
    ou_bodyforce,
)

# Number of time steps to save
nt = round(Int, tsim / Δt)

to_setup_les = RikFlow.TO_Setup(; 
        qois,
        to_mode = :ONLINE,
        time_series_method,
        ArrayType, 
        setup, 
        nstep=nt)

psolver = create_psolver(setup)

# Solve
@info "Solving LF sim (online SGS)"
(; u, t), outputs =
        solve_unsteady(; setup, 
        ustart,
        method = TOMethod(; to_setup = to_setup_les), 
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 100),
            fields = fieldsaver(; setup, nupdate = savefreq),  # by calling this BEFORE qoisaver, we also save the field at t=0!
            qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1),
            # vort = realtimeplotter(;
            #     setup,
            #     plot = vortplot,
            #     nupdate = 10,
            #     displayupdates = true,
            #     displayfig = true,
            # ),
        ),
        psolver)
q = stack(outputs.qoihist)
dQ = to_setup_les.outputs.dQ
tau = to_setup_les.outputs.tau
fields = outputs.fields
return (;dQ, tau, q, fields)
end