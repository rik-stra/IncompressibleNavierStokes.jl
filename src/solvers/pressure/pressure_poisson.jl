"""
    pressure_poisson(solver, f, t, setup, tol = 1e-14)

Solve the Poisson equation for the pressure with right hand side `f` at time `t`.

We should have `sum(f) = 0` for periodic and no-slip BC.
"""
function pressure_poisson end

function pressure_poisson(::DirectPressureSolver, f, t, setup, tol = 1e-14)
    # Assume the Laplace matrix is known (A) and is possibly factorized
    @unpack A_fact = setup.discretization

    # Use pre-determined decomposition
    Δp = A_fact \ f
end

function pressure_poisson(::CGPressureSolver, f, t, setup, tol = 1e-14)
    @unpack A = setup.discretization
    Δp = zeros(size(A, 1))
    cg!(Δp, A, f)
end

function pressure_poisson(::FFTPressureSolver,f,t, setup, tol = 1e-14)
    @unpack Npx, Npy = setup.grid
    @unpack Â = setup.solver_settings

    # Fourier transform of right hand side
    f̂ = fft(reshape(f, Npx, Npy));

    # Solve for coefficients in Fourier space
    p̂ = -f̂ ./ Â;

    # Transform back
    Δp = real.(ifft(p̂));

    # Convert 2D field to 1D column vector
    Δp = reshape(Δp, length(Δp));
end
