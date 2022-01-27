# Run a typical simulation: Lid-Driven Cavity case (LDC)
@testset "Simulation 2D" begin
    # Floating point type for simulations
    T = Float64

    ## Viscosity model
    viscosity_model = LaminarModel{T}(; Re = 1000)
    # viscosity_model = KEpsilonModel{T}(; Re = 1000)
    # viscosity_model = MixingLengthModel{T}(; Re = 1000)
    # viscosity_model = SmagorinskyModel{T}(; Re = 1000)
    # viscosity_model = QRModel{T}(; Re = 1000)

    ## Convection model
    convection_model = NoRegConvectionModel{T}()
    # convection_model = C2ConvectionModel{T}()
    # convection_model = C4ConvectionModel{T}()
    # convection_model = LerayConvectionModel{T}()

    ## Grid parameters
    x = stretched_grid(0, 1, 25)
    y = stretched_grid(0, 1, 25)
    grid = create_grid(x, y; T)

    ## Solver settings
    solver_settings = SolverSettings{T}(;
        pressure_solver = DirectPressureSolver{T}(),    # Pressure solver
        # pressure_solver = CGPressureSolver{T}(),      # Pressure solver
        # pressure_solver = FourierPressureSolver{T}(), # Pressure solver
        p_add_solve = true,                             # Additional pressure solve for second order pressure
        abstol = 1e-10,                                 # Absolute accuracy
        reltol = 1e-14,                                 # Relative accuracy
        maxiter = 10,                                   # Maximum number of iterations
        # :no: Replace iteration matrix with I/Δt (no Jacobian)
        # :approximate: Build Jacobian once before iterations only
        # :full: Build Jacobian at each iteration
        newton_type = :approximate,
    )

    ## Boundary conditions
    lid_vel = 1.0 # Lid velocity
    u_bc(x, y, t, setup) = y ≈ setup.grid.ylims[2] ? lid_vel : 0.0
    v_bc(x, y, t, setup) = zero(x)
    bc = create_boundary_conditions(
        u_bc,
        v_bc;
        bc_unsteady = false,
        bc_type = (;
            u = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
            v = (; x = (:dirichlet, :dirichlet), y = (:dirichlet, :dirichlet)),
        ),
        T,
    )

    ## Forcing parameters
    bodyforce_u(x, y) = 0
    bodyforce_v(x, y) = 0
    force = SteadyBodyForce{T}(; bodyforce_u, bodyforce_v)

    ## Build setup and assemble operators
    setup =
        Setup{T,2}(; viscosity_model, convection_model, grid, force, solver_settings, bc)
    build_operators!(setup)

    ## Time interval
    t_start, t_end = tlims = (0.0, 0.5)

    ## Initial conditions
    initial_velocity_u(x, y) = 0
    initial_velocity_v(x, y) = 0
    initial_pressure(x, y) = 0
    V₀, p₀ = create_initial_conditions(
        setup,
        t_start;
        initial_velocity_u,
        initial_velocity_v,
        initial_pressure,
    )

    @testset "Steady state problem" begin
        problem = SteadyStateProblem(setup, V₀, p₀)
        V, p = solve(problem)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < lid_vel
    end

    ## Iteration processors
    logger = Logger()
    tracer = QuantityTracer()
    processors = [logger, tracer]

    @testset "Unsteady problem" begin
        problem = UnsteadyProblem(setup, V₀, p₀, tlims)
        V, p = solve(problem, RK44(); Δt = 0.01, processors)

        # Check that solution did not explode
        @test all(!isnan, V)
        @test all(!isnan, p)

        # Check that the average velocity is smaller than the lid velocity
        @test sum(abs, V) / length(V) < lid_vel

        # Check for steady state convergence
        @test tracer.umom[end] < 1e-10
        @test tracer.vmom[end] < 1e-10
    end
end