create_stepper(
    method::AdamsBashforthCrankNicolsonMethod;
    setup,
    psolver,
    bc_vectors,
    V,
    p,
    t,
    n = 0,

    # For the first step, these are not used
    Vₙ = copy(V),
    pₙ = copy(p),
    cₙ = copy(V),
    tₙ = t,
    Diff_fact = spzeros(eltype(V), 0, 0),
) = (; setup, psolver, bc_vectors, V, p, t, n, Vₙ, pₙ, cₙ, tₙ, Diff_fact)

function timestep(method::AdamsBashforthCrankNicolsonMethod, stepper, Δt)
    (; setup, psolver, bc_vectors, V, p, t, n, Vₙ, pₙ, cₙ, tₙ, Diff_fact) = stepper
    (; convection_model, viscosity_model, Re, force, grid, operators, boundary_conditions) =
        setup
    (; bc_unsteady) = boundary_conditions
    (; NV) = grid
    (; G, M) = operators
    (; Diff) = operators
    (; p_add_solve, α₁, α₂, θ, method_startup) = method

    T = typeof(Δt)

    # One-leg requires state at previous time step, which is not available at
    # the first iteration. Do one startup step instead
    if n == 0
        stepper_startup =
            create_stepper(method_startup; setup, psolver, bc_vectors, V, p, t)
        n += 1
        Vₙ = V
        pₙ = p
        tₙ = t

        # Initial convection term
        bc_unsteady && (bc_vectors = get_bc_vectors(setup, tₙ))
        cₙ, = convection(convection_model, Vₙ, Vₙ, setup; bc_vectors)

        if viscosity_model isa LaminarModel
            # Factorize implicit part at first time step
            Diff_fact = lu(I(NV) - θ * Δt / Re * Diagonal(1 ./ Ω) * Diff)
        end

        (; V, p, t) = timestep(method_startup, stepper_startup, Δt)
        return create_stepper(
            method;
            setup,
            psolver,
            bc_vectors,
            V,
            p,
            t,
            n,
            Vₙ,
            pₙ,
            cₙ,
            tₙ,
            Diff_fact,
        )
    end

    # Advance one step
    Δtₙ₋₁ = t - tₙ
    n += 1
    Vₙ = V
    pₙ = p
    tₙ = t
    Δtₙ = Δt
    cₙ₋₁ = cₙ

    # Adams-Bashforth requires fixed time step
    @assert Δtₙ ≈ Δtₙ₋₁

    # Unsteady BC at current time
    bc_unsteady && (bc_vectors = get_bc_vectors(setup, tₙ))
    (; yDiff) = bc_vectors

    yDiffₙ = yDiff

    # Evaluate boundary conditions and force at starting point
    bₙ = force

    # Convection of current solution
    cₙ, = convection(convection_model, Vₙ, Vₙ, setup; bc_vectors)

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    bc_unsteady && (bc_vectors = get_bc_vectors(setup, tₙ + Δt))
    (; yDiff, y_p) = bc_vectors

    bₙ₊₁ = force

    yDiffₙ₊₁ = yDiff

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    b = @. (1 - θ) * bₙ + θ * bₙ₊₁
    yDiff = @. (1 - θ) * yDiffₙ + θ * yDiffₙ₊₁

    Gpₙ = G * pₙ + y_p

    d = 1 / Re * (Diff * V)

    # Right hand side of the momentum equation update
    Rr = @. Vₙ + 1 / Ω * Δt * (-(α₁ * cₙ + α₂ * cₙ₋₁) + (1 - θ) * d + yDiff + b - Gpₙ)

    # Implicit time-stepping for diffusion
    if viscosity_model isa LaminarModel
        # Use precomputed LU decomposition
        V = Diff_fact \ Rr
    else
        # Get `∇d` since `Diff` is not constant
        d, ∇d = diffusion(V, t, setup; get_jacobian = true)
        V = ∇d \ Rr
    end

    # Make the velocity field `uₙ₊₁` at `tₙ₊₁` divergence-free (need BC at `tₙ₊₁`)
    bc_unsteady && (bc_vectors = get_bc_vectors(setup, tₙ + Δt))
    (; yM) = bc_vectors

    # Boundary condition for Δp between time steps (!= 0 if fluctuating outlet pressure)
    y_Δp = zeros(T, NV)

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * V + yM) / Δt - M * y_Δp

    # Solve the Poisson equation for the pressure
    Δp = poisson(psolver, f)

    # Update velocity field
    V -= Δt ./ Ω .* (G * Δp .+ y_Δp)

    # First order pressure:
    p = pₙ .+ Δp

    if p_add_solve
        p = pressure(psolver, V, p, tₙ + Δt, setup; bc_vectors)
    end

    t = tₙ + Δtₙ

    create_stepper(
        method;
        setup,
        psolver,
        bc_vectors,
        V,
        p,
        t,
        n,
        Vₙ,
        pₙ,
        cₙ,
        tₙ,
        Diff_fact,
    )
end

function timestep!(
    method::AdamsBashforthCrankNicolsonMethod,
    stepper,
    Δt;
    cache,
    momentum_cache,
)
    (; setup, psolver, bc_vectors, V, p, t, n, Vₙ, pₙ, cₙ, tₙ, Diff_fact) = stepper
    (; convection_model, viscosity_model, Re, force, grid, operators, boundary_conditions) =
        setup
    (; bc_unsteady) = boundary_conditions
    (; NV, Ω) = grid
    (; G, M) = operators
    (; Diff) = operators
    (; p_add_solve, α₁, α₂, θ, method_startup) = method
    (; cₙ₋₁, F, f, Δp, Rr, b, bₙ, bₙ₊₁, yDiffₙ, yDiffₙ₊₁, Gpₙ) = cache
    (; d, ∇d) = momentum_cache

    T = typeof(Δt)

    # One-leg requires state at previous time step, which is not available at
    # the first iteration. Do one startup step instead
    if n == 0
        stepper_startup =
            create_stepper(method_startup; setup, psolver, bc_vectors, V, p, t)
        n += 1
        Vₙ = V
        pₙ = p
        tₙ = t

        # Initial convection term
        bc_unsteady && (bc_vectors = get_bc_vectors(setup, tₙ))
        cₙ, = convection(convection_model, Vₙ, Vₙ, setup; bc_vectors)

        # Factorize implicit part at first time step
        Diff_fact = lu(I(NV) - θ * Δt / Re * Diagonal(1 ./ Ω) * Diff)

        # Note: We do one out-of-place step here, with a few allocations
        (; V, p, t) = timestep(method_startup, stepper_startup, Δt)
        return create_stepper(
            method;
            setup,
            psolver,
            bc_vectors,
            V,
            p,
            t,
            n,
            Vₙ,
            pₙ,
            cₙ,
            tₙ,
            Diff_fact,
        )
    end

    # Advance one step
    Δtₙ₋₁ = t - tₙ
    n += 1
    Vₙ .= V
    pₙ .= p
    tₙ = t
    Δtₙ = Δt
    cₙ₋₁ .= cₙ

    # Adams-Bashforth requires fixed time step
    @assert Δtₙ ≈ Δtₙ₋₁

    # Unsteady BC at current time
    bc_unsteady && (bc_vectors = get_bc_vectors(setup, tₙ))
    (; yDiff) = bc_vectors

    yDiffₙ .= yDiff

    # Evaluate boundary conditions and force at starting point
    bₙ = force

    # Convection of current solution
    convection!(convection_model, cₙ, nothing, Vₙ, Vₙ, setup, momentum_cache; bc_vectors)

    # Unsteady BC at next time (Vₙ is not used normally in bodyforce.jl)
    bc_unsteady && (bc_vectors = get_bc_vectors(setup, tₙ + Δt))
    (; yDiff, y_p) = bc_vectors
    bₙ₊₁ = force

    yDiffₙ₊₁ .= yDiff

    # Crank-Nicolson weighting for force and diffusion boundary conditions
    @. b = (1 - θ) * bₙ + θ * bₙ₊₁
    yDiff = @. (1 - θ) * yDiffₙ + θ * yDiffₙ₊₁

    mul!(Gpₙ, G, pₙ)
    Gpₙ .+= y_p

    mul!(d, Diff, V)
    d ./= Re

    # Right hand side of the momentum equation update
    @. Rr = Vₙ + 1 ./ Ω * Δt * (-(α₁ * cₙ + α₂ * cₙ₋₁) + (1 - θ) * d + yDiff + b - Gpₙ)

    # Implicit time-stepping for diffusion
    if viscosity_model isa LaminarModel
        # Use precomputed LU decomposition
        ldiv!(V, Diff_fact, Rr)
    else
        # Get `∇d` since `Diff` is not constant
        diffusion!(d, ∇d, V, t, setup; get_jacobian = true)
        V .= ∇d \ Rr
    end

    # Make the velocity field `uₙ₊₁` at `tₙ₊₁` divergence-free (need BC at `tₙ₊₁`)
    if isnothing(bc_vectors) || bc_unsteady
        bc_vectors = get_bc_vectors(setup, tₙ + Δt)
    end
    (; yM) = bc_vectors

    # Boundary condition for Δp between time steps (!= 0 if fluctuating outlet pressure)
    y_Δp = zeros(T, NV)

    # Divergence of `Ru` and `Rv` is directly calculated with `M`
    f = (M * V + yM) / Δt - M * y_Δp

    # Solve the Poisson equation for the pressure
    poisson!(psolver, Δp, f)

    # Update velocity field
    V .-= Δt ./ Ω .* (G * Δp .+ y_Δp)

    # First order pressure:
    p .= pₙ .+ Δp

    if p_add_solve
        pressure!(psolver, V, p, tₙ + Δt, setup, momentum_cache, F, f, Δp; bc_vectors)
    end

    t = tₙ + Δtₙ

    create_stepper(
        method;
        setup,
        psolver,
        bc_vectors,
        V,
        p,
        t,
        n,
        Vₙ,
        pₙ,
        cₙ,
        tₙ,
        Diff_fact,
    )
end
