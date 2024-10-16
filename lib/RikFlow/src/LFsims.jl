function track_ref(;
    ustart,
    qoi_ref,
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
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y),
    kwargs...,
)
T = typeof(Re)

# Build setup and assemble operators

setup = 
Setup(;
    x = ntuple(α -> LinRange(lims[α]..., nles[1][α] + 1), D),
    Re,
    ArrayType,
    bodyforce,
)

# Number of time steps to save
nt = round(Int, tsim / Δt)

to_setup_les = RikFlow.TO_Setup(; 
        qois, 
        qoi_refs_location= qoi_ref, 
        to_mode = :TRACK_REF, 
        ArrayType, 
        setup, 
        nstep=nt)

psolver = create_psolver(setup)

# Solve
@info "Solving LF sim (track ref)"
(; u, t), outputs =
        solve_unsteady(; setup, 
        ustart,
        method = TOMethod(; to_setup = to_setup_les), 
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 10),
            fields = fieldsaver(; setup, nupdate = savefreq), # by calling this BEFORE qoisaver, we also save the field at t=0!
            qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1),
            vort = realtimeplotter(;
                setup,
                plot = vortplot,
                nupdate = 10,
                displayupdates = true,
                displayfig = true,
            ),
        ),
        psolver)
q = stack(outputs.qoihist)
dQ = to_setup_les.outputs.dQ
tau = to_setup_les.outputs.tau
return (;dQ, tau, q, outputs.fields)
end


function online_sgs(;
    dQ_data,
    ustart,
    sampling_method = :mvg,
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
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y),
    rng = Xoshiro(234),
    kwargs...,
)
T = typeof(Re)

# Build setup and assemble operators

setup = 
Setup(;
    x = ntuple(α -> LinRange(lims[α]..., nles[1][α] + 1), D),
    Re,
    ArrayType,
    bodyforce,
)

# Number of time steps to save
nt = round(Int, tsim / Δt)

to_setup_les = RikFlow.TO_Setup(; 
        qois, 
        qoi_refs_location= "none", 
        rng,
        dQ_data,
        to_mode = :ONLINE,
        sampling_method,
        ArrayType, 
        setup, 
        nstep=nt)

psolver = create_psolver(setup)

# Solve
@info "Solving LF sim (track ref)"
(; u, t), outputs =
        solve_unsteady(; setup, 
        ustart,
        method = TOMethod(; to_setup = to_setup_les), 
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 10),
            fields = fieldsaver(; setup, nupdate = savefreq),  # by calling this BEFORE qoisaver, we also save the field at t=0!
            qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1),
            vort = realtimeplotter(;
                setup,
                plot = vortplot,
                nupdate = 10,
                displayupdates = true,
                displayfig = true,
            ),
        ),
        psolver)
q = stack(outputs.qoihist)
dQ = to_setup_les.outputs.dQ
tau = to_setup_les.outputs.tau
return (;dQ, tau, q)
end