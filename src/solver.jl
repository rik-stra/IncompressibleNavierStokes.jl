"Navier-Stokes momentum forcing (convection + diffusion)."
function navierstokes!(force, state, t; setup, cache, viscosity)
    (; u) = state
    fill!(force.u, 0)
    convectiondiffusion!(force.u, state.u, setup, viscosity)
end

"Navier-Stokes momentum forcing (convection + diffusion)."
function navierstokes(state, t; setup, viscosity)
    c = convection(state.u, setup)
    d = diffusion(state.u, setup, viscosity)
    (; u = c + d)
end

"Boussinesq forcing (Navier-Stokes + gravity for `u`, convection-diffusion for `temp`)."
function boussinesq!(
    force,
    state,
    t;
    setup,
    cache,
    viscosity,
    conductivity,
    gdir,
    gravity,
    dodissipation,
)
    (; u, temp) = state
    fill!(force.u, 0)
    fill!(force.temp, 0)
    convectiondiffusion!(force.u, u, setup, viscosity)
    applygravity!(force.u, temp, setup, gdir, gravity)
    convection_diffusion_temp!(force.temp, u, temp, setup, conductivity)
    dodissipation && dissipation!(force.temp, u, setup, viscosity)
end

"Boussinesq forcing (Navier-Stokes + gravity for `u`, convection-diffusion for `temp`)."
function boussinesq(state, t; setup, viscosity, conductivity, gdir, gravity, dodissipation)
    (; u, temp) = state
    d = diffusion(u, setup, viscosity)
    c = convection(u, setup)
    g = applygravity(temp, setup, gdir, gravity)
    fu = @. c + d + g
    ftemp = convection_diffusion_temp(u, temp, setup, conductivity)
    dodissipation && (ftemp += dissipation(u, setup, viscosity))
    (; u = fu, temp = ftemp)
end

get_cache(::typeof(navierstokes!), setup) = nothing
get_cache(::typeof(boussinesq!), setup) = nothing

"""
Solve unsteady problem using `method`.

The initial `start` state is a named tuple of fields, e.g. `(; u)` or
`(; u, temp)`. The right-hand side `force!` is called as
`force!(force, state, t; setup, cache, params...)`, where `params` is a named
tuple of parameters passed as keyword arguments (e.g.
`params = (; viscosity)` for the default `navierstokes!`, or
`(; viscosity, conductivity, gdir, gravity, dodissipation)` for
`boussinesq!`).

If `Δt` is a real number, it is rounded such that `(t_end - t_start) / Δt` is
an integer.
If `Δt = nothing`, the time step is chosen every `n_adapt_Δt` iteration with
CFL-number `cfl`. If `Δt_min` is given, the adaptive time step never goes below it.

The `processors` are called after every time step.

Note that the `state` observable passed to the `processor.initialize` function
contains fields living on the device, and you may have to move them back to
the host using `Array` in the processor.

Return `(; state..., t), outputs`, where `outputs` is a named tuple with the
outputs of `processors` with the same field names.
"""
function solve_unsteady(;
    setup,
    tlims,
    start,
    force! = navierstokes!,
    docopy = true,
    method = LMWray3(; T = eltype(start.u)),
    psolver = default_psolver(setup),
    Δt = nothing,
    Δt_min = nothing,
    cfl = eltype(start.u)(0.9),
    n_adapt_Δt = 1,
    processors = (;),
    params,
    # Cache arrays for intermediate computations
    ode_cache = get_cache(method, start, setup),
    force_cache = get_cache(force!, setup),
)
    tstart, tend = tlims
    isadaptive = isnothing(Δt)

    state = docopy ? deepcopy(start) : start

    # Time stepper
    stepper = create_stepper(method; setup, psolver, state, t = tstart)

    # Initialize processors for iteration results
    state = Observable(get_state(stepper))
    initialized = (; (k => v.initialize(state) for (k, v) in pairs(processors))...)

    # ------------------------------------------------------------------------------------------
    # OU forcing — ours, not upstream's. Do not fold this into `force!`.
    #
    # 🔴 `force!` is called once per RK *stage*, not once per step: `timestep!` for an
    # `ExplicitRungeKuttaMethod` calls it inside `for i = 1:nstage`, which is 4 times per step
    # under RK44. Advancing the OU chain there would advance it 4× per step and silently destroy
    # `claude_memory.md` gotcha #33's measured advance count, producing a trajectory that looks
    # entirely plausible. The advance therefore stays here, once per step, exactly where it was
    # before the upstream merge.
    #
    # What did move is the storage: the chain and its force buffer now live in `force_cache`
    # (built by `ou_force_cache`), because upstream's `setup` is a pure grid description with no
    # room for `ou_setup` or `bodyforce`. The *structure* gotcha #33 measured — one priming call
    # before the loop, one advance per iteration gated on `freeze` — is unchanged, so
    # `lib/RikFlow/analysis/ou_replay.jl` still measures the count the same way. It must be re-run
    # after this merge regardless (checklist item 27).
    isou = force_cache isa NamedTuple && haskey(force_cache, :ou_setup)
    if isou
        # 🔴 In adaptive mode `Δt` is `nothing` here — the first step size is not known until the
        # loop proposes it — so the priming advance has to ask for that proposal itself. Without
        # this, OU forcing plus adaptive stepping throws `MethodError: *(::Nothing, ::Int64)`
        # before the first step.
        #
        # ⚠️ Pre-existing, not introduced by the upstream merge: the same `Δt * freeze` sat in the
        # same place before it. The combination had simply never been run — every production case
        # uses a fixed Δt.
        #
        # The non-adaptive path is untouched: `Δt_prime === Δt` there, so the advance count and the
        # step size that `claude_memory.md` gotcha #33 measured are bit-for-bit what they were.
        Δt_prime = isadaptive ?
            cfl * propose_timestep(force!, stepper.state, setup, params) : Δt

        # Print one random forcing field, to check the same random seed is being used etc.
        OU_forcing_step!(; force_cache.ou_setup, Δt = Δt_prime * force_cache.freeze)
        OU_get_force!(force_cache.ou_setup, force_cache.bodyforce, setup)
    end
    # ------------------------------------------------------------------------------------------

    if isadaptive
        if isou
            @assert force_cache.freeze == 1 "Can't freeze the bodyforce over multiple adaptive time steps"
        end
        while stepper.t < tend
            if stepper.n % n_adapt_Δt == 0
                # Change timestep based on operators
                Δt = cfl * propose_timestep(force!, stepper.state, setup, params)
                Δt = isnothing(Δt_min) ? Δt : max(Δt, Δt_min)
                if Δt < 1e-10 || Δt > 100 || isnan(Δt)
                    @warn "Proposed time step $Δt is out of bounds. Stopping simulation."
                    break
                end
            end

            # Make sure not to step past `t_end`
            # (keep `Δt` itself unclipped, it is reused until the next adaptation)
            Δt_step = min(Δt, tend - stepper.t)

            # Update forcing (ours)
            if isou
                OU_forcing_step!(; force_cache.ou_setup, Δt = Δt_step)
                OU_get_force!(force_cache.ou_setup, force_cache.bodyforce, setup)
            end

            # Perform a single time step with the time integration method
            stepper =
                timestep!(method, force!, stepper, Δt_step; params, ode_cache, force_cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)
        end
    else
        nstep = round(Int, (tend - tstart) / Δt)
        Δt = (tend - tstart) / nstep
        for it = 1:nstep
            # Update forcing (ours). Gated on `freeze`, and stepped by `Δt * freeze`: the chain
            # advances only on iterations where `mod(stepper.n, freeze) == 0`. Gotcha #33's rider
            # — the count is `nstep + 1` only at `freeze == 1` — is a property of these two lines.
            if isou && mod(stepper.n, force_cache.freeze) == 0
                OU_forcing_step!(; force_cache.ou_setup, Δt = Δt * force_cache.freeze)
                OU_get_force!(force_cache.ou_setup, force_cache.bodyforce, setup)
            end

            # Perform a single time step with the time integration method
            stepper = timestep!(method, force!, stepper, Δt; params, ode_cache, force_cache)

            # Process iteration results with each processor
            state[] = get_state(stepper)

            # NaN guard (ours). `RikFlow`'s `qoisaver` sets this flag; `setup` carries it because
            # RikFlow extends upstream's setup NamedTuple with its own fields.
            if haskey(setup, :nans_detected) && setup.nans_detected[]
                @warn "NaNs detected in the solution. Stopping the simulation."
                break
            end
        end
    end

    # Processor outputs
    outputs = (;
        (k => processors[k].finalize(initialized[k], state) for k in keys(processors))...
    )

    # Return state and outputs
    (; stepper.state..., stepper.t), outputs
end

"Get state `(; u, temp, t, n)` from stepper."
function get_state(stepper)
    (; state, t, n) = stepper
    (; state..., t, n)
end

function propose_timestep(::typeof(diffusion!), state, setup, params)
    (; dimension, Δu, Iu) = setup
    D = dimension()

    # Check maximum step size in each dimension
    minimum(1:D) do α
        Δαmin = minimum(view(Δu[α], Iu[α].indices[α]))
        Δαmin^2 / params.viscosity / 2D
    end
end

broadcastreduce(f, op, args...; kwargs...) =
    reduce(op, Broadcast.instantiate(Broadcast.broadcasted(f, args...); kwargs...))

function propose_timestep(::typeof(convection!), state, setup, params)
    (; dimension, Δu, Iu) = setup
    D = dimension()
    (; u) = state

    # Check maximum step size in each dimension
    minimum(1:D) do α
        uα = selectdim(u, D + 1, α)
        Δα = view(Δu[α], Iu[α].indices[α])
        Δα = reshape(Δα, ntuple(Returns(1), α - 1)..., :)
        uα = view(uα, Iu[α])
        broadcastreduce(min, Δα, uα) do Δα, uα
            Δα / abs(uα)
        end
    end
end

function propose_timestep(::typeof(convection_diffusion_temp!), state, setup, params)
    (; dimension, Δ, Ip) = setup
    D = dimension()

    # Check maximum step size in each dimension
    minimum(1:D) do α
        Δαmin = minimum(view(Δ[α], Ip.indices[α]))
        Δαmin^2 / params.conductivity / 2D
    end
end

# Fallback
propose_timestep(_, state, setup, params) =
    propose_timestep(navierstokes!, state, setup, params)

propose_timestep(::typeof(navierstokes), state, setup, params) =
    propose_timestep(navierstokes!, state, setup, params)
propose_timestep(::typeof(navierstokes!), state, setup, params) = min(
    propose_timestep(convection!, state, setup, params),
    propose_timestep(diffusion!, state, setup, params),
)
propose_timestep(::typeof(boussinesq), state, setup, params) =
    propose_timestep(boussinesq!, state, setup, params)
propose_timestep(::typeof(boussinesq!), state, setup, params) = min(
    propose_timestep(navierstokes!, state, setup, params),
    propose_timestep(convection_diffusion_temp!, state, setup, params),
)
