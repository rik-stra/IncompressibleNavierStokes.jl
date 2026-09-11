# This file contains standard setups for the low-fidelity HIT simulations.

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
    tracking_noise_seed = 56,
    kwargs...,
    )
    T = typeof(Re)

    # Build setup and assemble operators

    setup = 
    rf_setup(;
        x = ntuple(α -> LinRange(lims[α]..., nles[1][α] + 1), D),
        Re,
        ArrayType,
        backend,
    )

    # Forcing is the right-hand side and its cache now, not a setup field.
    force!, force_cache = if isnothing(ou_bodyforce)
        navierstokes!, nothing
    else
        ou_navierstokes!, ou_force_cache(setup; ou_bodyforce...)
    end

    # Number of time steps to save
    nt = round(Int, tsim / Δt)

    to_setup_les = RikFlow.TO_Setup(; 
            qois, 
            to_mode = :TRACK_REF, 
            ArrayType, 
            setup, 
            nstep=nt,
            time_series_method = ref_reader,
            tracking_noise = tracking_noise,
            tracking_noise_seed = tracking_noise_seed)

    psolver = create_psolver(setup)

    # Solve
    @info "Solving LF sim (track ref)"
    println("ou forcing: ", isnothing(ou_bodyforce) ? "none" : ou_bodyforce)
    (; u, t), outputs =
            solve_unsteady(; setup, 
            start = (; u = ustart),
            force!,
            force_cache,
            params = rf_params(setup),
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
    ou_advance::Int = 0,
    kwargs...,
)
T = typeof(Re)

# Build setup and assemble operators

setup = 
rf_setup(;
    x = ntuple(α -> LinRange(lims[α]..., nles[1][α] + 1), D),
    Re,
    ArrayType,
    backend,
)

# Forcing is the right-hand side and its cache now, not a setup field.
force!, force_cache = if isnothing(ou_bodyforce)
    navierstokes!, nothing
else
    ou_navierstokes!, ou_force_cache(setup; ou_bodyforce...)
end

# Number of time steps to save
nt = round(Int, tsim / Δt)

# Replay the OU forcing chain to the reference step this initial condition was taken from.
#
# `ou_advance = 0` is the default and leaves `OU_setup`'s zero state untouched, so every archived run
# reproduces exactly and nothing already computed shifts. It is non-zero only for the multi-IC
# ensemble D6, where `ustart` is `fields[k].u` at reference step `n_k` and the forcing that produced
# that field is `n_k` steps into its own chain. Launching such a run from a zero state leaves every
# member's forcing out of phase with its initial condition, which inflates skill without inflating
# spread and biases the spread-skill ratio downward -- see `OU_advance!` and
# `meta_files/handoff_p2c_d6.md` section 3 step 2.
#
# 🔑 The chain must be replayed with the step size the *reference* used, and continued with the step
# size *this* run uses. `solve_unsteady` does not step with the `Δt` it is given: it re-derives
# `Δt = (tend - tstart) / nstep` (the non-adaptive branch of `solve_unsteady`). The two agree only when `tsim / Δt` is
# integral, so that is asserted rather than assumed -- silently stepping the replay at a different
# Δt would reintroduce exactly the misphase this keyword exists to remove.
if ou_advance != 0
    isnothing(ou_bodyforce) &&
        error("online_sgs: ou_advance = $ou_advance was given but ou_bodyforce is nothing")
    Δt_solver = T(tsim) / nt
    @assert Δt_solver == T(Δt) "online_sgs: ou_advance needs tsim/Δt integral so the replay and " *
        "the reference step the OU chain identically; got Δt = $(Δt), tsim/nt = $(Δt_solver)"
    # 🔴 `freeze` must be 1, and this refuses rather than generalises.
    #
    # `solve_unsteady` advances the chain only on iterations where `mod(stepper.n, freeze) == 0`,
    # and it advances by `Δt * freeze` (the OU block in `solve_unsteady`). So `n_k` reference solver steps
    # correspond to `ceil(n_k / freeze)` advances of size `Δt * freeze`, not to `n_k` advances of
    # size `Δt`. At `freeze = 1` the two coincide, which is exactly why a freeze-blind replay looks
    # right on HIT -- `params_track.ou_bodyforce.freeze = 1` there -- and would be wrong on both
    # count and step size for the `freeze = 10` DNS family that sits in the same archive.
    #
    # The generalisation is not written here because the cancellation of the priming advance would
    # have to be re-measured for it, and `analysis/ou_replay.jl` measures the count per `freeze`
    # for exactly that purpose. Until someone needs it, refusing is the honest behaviour.
    @assert ou_bodyforce.freeze == 1 "online_sgs: ou_advance = $ou_advance with freeze = " *
        "$(ou_bodyforce.freeze). The replay is only derived for freeze == 1; at freeze != 1 the " *
        "solver advances the chain every freeze steps by Δt*freeze, so n_k advances of Δt is " *
        "wrong on both count and step size. See analysis/ou_replay.jl."
    OU_advance!(; force_cache.ou_setup, Δt = Δt_solver, n = ou_advance)
end

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
        start = (; u = ustart),
        force!,
        force_cache,
        params = rf_params(setup),
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