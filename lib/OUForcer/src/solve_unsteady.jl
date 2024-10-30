using IncompressibleNavierStokes: timestep!, create_stepper, get_state, default_psolver, 
    ode_method_cache, AbstractODEMethod, AbstractRungeKuttaMethod, RKMethods, processor
function solve_unsteady(;
    setup,
    tlims,
    ustart,
    tempstart = nothing,
    method = RKMethods.RK44(; T = eltype(ustart[1])),
    psolver = default_psolver(setup),
    Δt = nothing,
    cfl = eltype(ustart[1])(0.9),
    n_adapt_Δt = 1,
    docopy = true,
    processors = (;),
    θ = nothing,
)
    docopy && (ustart = copy.(ustart))
    docopy && !isnothing(tempstart) && (tempstart = copy(tempstart))

    tstart, tend = tlims
    isadaptive = isnothing(Δt)
    if isadaptive
        cflbuf = scalarfield(setup)
    end

    # Cache arrays for intermediate computations
    cache = ode_method_cache(method, setup, ustart, tempstart)

    # Time stepper
    stepper =
        create_stepper(method; setup, psolver, u = ustart, temp = tempstart, t = tstart)

    # Initialize processors for iteration results
    state = Observable(get_state(stepper))
    initialized = (; (k => v.initialize(state) for (k, v) in pairs(processors))...)

    if isadaptive
        while stepper.t < tend
            if stepper.n % n_adapt_Δt == 0
                # Change timestep based on operators
                # Δt = get_timestep(stepper, cfl)
                Δt = cfl * get_cfl_timestep!(cflbuf, stepper.u, setup)
            end

            # Make sure not to step past `t_end`
            Δt = min(Δt, tend - stepper.t)
            # update forcing
            if !isnothing(ou_bodyforce)
                OU_forcing_step!(; setup.ou_setup, Δt=Δt)
                OU_get_force!(setup.ou_setup, stepper.t, setup)
            end
            # Perform a single time step with the time integration method
            stepper = timestep!(method, stepper, Δt; θ, cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
        end
    else
        nstep = round(Int, (tend - tstart) / Δt)
        Δt = (tend - tstart) / nstep
        for it = 1:nstep
            # update forcing
            if !isnothing(ou_bodyforce)
                OU_forcing_step!(; setup.ou_setup, Δt=Δt)
                OU_get_force!(setup.ou_setup, stepper.t, setup)
            end
            # Perform a single time step with the time integration method
            stepper = timestep!(method, stepper, Δt; θ, cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
        end
    end

    # Final state
    (; u, temp, t) = stepper

    # Processor outputs
    outputs = (;
        (k => processors[k].finalize(initialized[k], state) for k in keys(processors))...
    )

    # Return state and outputs
    (; u, temp, t), outputs
end

"Create problem setup (stored in a named tuple)."
function Setup(;
    x,
    boundary_conditions = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x)),
    bodyforce = nothing,
    ou_bodyforce = nothing,  # to use OU forcing pass named tuple (T_L, e_star, k_f)
    issteadybodyforce = true,
    closure_model = nothing,
    projectorder = :last,
    ArrayType = Array,
    workgroupsize = 64,
    temperature = nothing,
    Re = isnothing(temperature) ? convert(eltype(x[1]), 1_000) : 1 / temperature.α1,
)
    setup = (;
        grid = Grid(x, boundary_conditions; ArrayType),
        boundary_conditions,
        Re,
        bodyforce,
        issteadybodyforce = false,
        closure_model,
        projectorder,
        ArrayType,
        T = eltype(x[1]),
        workgroupsize,
        temperature,
    )
    if !isnothing(ou_bodyforce)  # Calculate OU body force
        (;T_L, e_star, k_f) = ou_bodyforce
        ou_setup = OU_setup(; T_L, e_star, k_f, setup)
        bodyforce = vectorfield(setup)
        setup = (; setup..., ou_setup, bodyforce, issteadybodyforce = true)
    else
        if !isnothing(bodyforce) && issteadybodyforce  # Calculate steady body force
            (; dimension, x, N) = setup.grid
            T = eltype(x[1])
            u = vectorfield(setup)
            F = vectorfield(setup)
            bodyforce = applybodyforce!(F, u, T(0), setup)
            setup = (; setup..., issteadybodyforce = true, bodyforce)
        end
    end
    setup
end