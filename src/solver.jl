"""
Solve unsteady problem using `method`.

If `Δt` is a real number, it is rounded such that `(t_end - t_start) / Δt` is
an integer.
If `Δt = nothing`, the time step is chosen every `n_adapt_Δt` iteration with
CFL-number `cfl`. If `Δt_min` is given, the adaptive time step never goes below it.

The `processors` are called after every time step.

Note that the `state` observable passed to the `processor.initialize` function
contains vector living on the device, and you may have to move them back to
the host using `Array(u)` in the processor.

Return `(; u, t), outputs`, where `outputs` is a  named tuple with the
outputs of `processors` with the same field names.
"""
function solve_unsteady(;
    setup,
    tlims,
    ustart,
    tempstart = nothing,
    method = RKMethods.RK44(; T = eltype(ustart)),
    psolver = default_psolver(setup),
    Δt = nothing,
    Δt_min = nothing,
    cfl = eltype(ustart)(0.9),
    n_adapt_Δt = 1,
    docopy = true,
    processors = (;),
    θ = nothing,
    # Cache arrays for intermediate computations
    cache = ode_method_cache(method, setup),
)
    docopy && (ustart = copy(ustart))
    docopy && !isnothing(tempstart) && (tempstart = copy(tempstart))

    tstart, tend = tlims
    isadaptive = isnothing(Δt)
    if isadaptive
        cflbuf = scalarfield(setup)
    end

    # Time stepper
    stepper =
        create_stepper(method; setup, psolver, u = ustart, temp = tempstart, t = tstart)

    # Initialize processors for iteration results
    state = Observable(get_state(stepper))
    initialized = (; (k => v.initialize(state) for (k, v) in pairs(processors))...)

    # print one random forcing field to check if the same random seed is used ect.
    if !isnothing(setup.ou_bodyforce)
        OU_forcing_step!(; setup.ou_setup, Δt=Δt*setup.ou_bodyforce.freeze)
        OU_get_force!(setup.ou_setup, setup)
        @info "ou random state 1: $(setup.ou_setup.state[1:2])", typeof(setup.ou_setup.state)
        @info "ou force corner $(Array(setup.bodyforce[1])[end-1,end-1,end-1])"
        @info "ou force 1 cel from corner  $(Array(setup.bodyforce[1])[end-1,end-2,end-2])"
        @info "ou force 2 cel from corner  $(Array(setup.bodyforce[1])[end-1,end-3,end-3])"
        @info "ou force 4 cel from corner  $(Array(setup.bodyforce[1])[end-1,end-5,end-5])"
        @info "ou force 8 cel from corner  $(Array(setup.bodyforce[1])[end-1,end-9,end-9])"
    end

    if isadaptive
        @assert setup.ou_bodyforce.freeze ==1 "Can't freeze the bodyforce over multiple adaptive time steps"
        while stepper.t < tend
            if stepper.n % n_adapt_Δt == 0
                # Change timestep based on operators
                # Δt = get_timestep(stepper, cfl)
                Δt = cfl * get_cfl_timestep!(cflbuf, stepper.u, setup)
                Δt = isnothing(Δt_min) ? Δt : max(Δt, Δt_min)
            end

            # Make sure not to step past `t_end`
            Δt = min(Δt, tend - stepper.t)
            # update forcing
            if !isnothing(setup.ou_bodyforce)
                OU_forcing_step!(; setup.ou_setup, Δt=Δt)
                OU_get_force!(setup.ou_setup, setup)
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
            if !isnothing(setup.ou_bodyforce) && mod(stepper.n, setup.ou_bodyforce.freeze) == 0
                OU_forcing_step!(; setup.ou_setup, Δt=Δt*setup.ou_bodyforce.freeze)
                OU_get_force!(setup.ou_setup, setup)
            end
            # Perform a single time step with the time integration method
            stepper = timestep!(method, stepper, Δt; θ, cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
            
            if setup.nans_detected[]
                @warn "NaNs detected in the solution. Stopping the simulation."
                break
            end
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

"Get state `(; u, temp, t, n)` from stepper."
function get_state(stepper)
    (; u, temp, t, n) = stepper
    (; u, temp, t, n)
end

"Get proposed maximum time step for convection and diffusion terms."
function get_cfl_timestep!(buf, u, setup)
    (; Re, grid) = setup
    (; dimension, Δ, Δu, Iu) = grid
    D = dimension()

    # Initial maximum step size
    Δt = eltype(u)(Inf)

    # Check maximum step size in each dimension
    for (α, uα) in enumerate(eachslice(u; dims = D + 1))
        # Diffusion
        Δαmin = minimum(view(Δu[α], Iu[α].indices[α]))
        Δt_diff = Re * Δαmin^2 / 2

        # Convection
        Δα = reshape(Δu[α], ntuple(Returns(1), α - 1)..., :)
        @. buf = Δα / abs(uα)
        Δt_conv = minimum(view(buf, Iu[α]))

        # Update time step
        Δt = min(Δt, Δt_diff, Δt_conv)
    end

    Δt
end
