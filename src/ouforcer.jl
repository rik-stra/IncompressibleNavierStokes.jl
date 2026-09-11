
#= Implementation of random forcing using Ornstein-Uhlenbeck processes as in (Eswaran and Pope, 1988) also see (Chouippe and Uhlmann, 2015)
References:
Chouippe, A., & Uhlmann, M. (2015). Forcing homogeneous turbulence in direct numerical simulation of particulate flow with interface resolution and gravity. Physics of Fluids, 27, 123301.
Eswaran, V., & Pope, S. B. (1988). An examination of forcing in direct numerical simulations of turbulence. Computers & Fluids, 16(3), 257-278.
=#


"""
Create setup for OU forcing. The forcing is specified by
- `T_L` the charasteristic time scale of the OU-process
- `e_star` the energy injection rate
- `k_f` the cutoff wavenumber of the forcing
Returns a tuple with the setup of the OU forcing.
"""
function OU_setup(; T_L, 
                    e_star, 
                    k_f,
                    setup, 
                    rng_seed = 42,
                    freeze = 1
                    )

T = typeof(setup.Re)
rng = Xoshiro(rng_seed)
ArrayType = setup.ArrayType
num_dims = setup.dimension()
Var = e_star/T_L

# check if grid is equidistant
#@assert all([all(Δ ≈ Δ[1]) for Δ in setup.Δ])
#@assert all([ (Δ[1] ≈ setup.Δ[1][1]) for Δ in setup.Δ])
N = setup.Nu[1][1]

# Count the forced wavenumbers
k_f_int = floor(Int, k_f)
k = -k_f_int:k_f_int
N_f = 0 # Number of forced wavenumbers
N_d = 0 # forcing degrees of freedom
forced_range = repeat([2*k_f_int+1], num_dims)
mask = Array{Bool, num_dims}(undef, forced_range...)
mask[:] .= false

if num_dims == 2
    for k1 = k, k2 = k
        k_ = sqrt(k1^2 + k2^2)
        if (k_ <= k_f) & (k_ > 0)
            N_f += 1
            if k2 <= 0
                N_d += 1
                mask[k1+k_f_int+1, k2+k_f_int+1] = true
            end
        end
    end
elseif num_dims == 3
    for k1 = k, k2 = k, k3 = k
        k_ = sqrt(k1^2 + k2^2 + k3^2)
        if (k_ <= k_f) & (k_ > 0)
            N_f += 1
            if k3 <= 0
                N_d += 1
                mask[k1+k_f_int+1, k2+k_f_int+1, k3+k_f_int+1] = true
            end
        end
    end
else
    error("Number of dimensions must be 2 or 3. Got $num_dims")
end

state = ArrayType{ComplexF32, 2}(undef, N_d, num_dims) # contains the state of the OU process
state[:,:] .= 0
f_hat = [ArrayType{ComplexF32, num_dims}(undef, forced_range...) for a = 1:num_dims] # contains the Fourier coefficients of the forcing
for d in 1:num_dims
    f_hat[d][:] .= 0
end

f = [ArrayType{ComplexF32, num_dims}(undef, setup.Nu[a]...) for a = 1:num_dims] # contains the forcing in physical space

# create partial IFFT matrix
E = Array{ComplexF32,2}(undef, 2*k_f_int+1, N)
for j = 1:N, i = -k_f_int:k_f_int
    E[i+k_f_int+1, j] = exp(pi*2im*(i)*(j)/N)
end
z = ArrayType{T, 2}(undef, N_d, num_dims*2)
E = ArrayType(E)
mask = ArrayType(mask)

ou_setup = (;
    T_L,
    Var,
    k_f,
    N_f,
    rng,
    state,
    f,
    f_hat,
    E,
    mask,
    num_dims,
    z,
    freeze
)
end

"""
    OU_state_step!(; ou_setup, Δt)

Advance the OU chain's `state` by one step, and do nothing else.

Split out of [`OU_forcing_step!`](@ref) so the state can be advanced without paying for the forcing
field, which is the expensive half -- an O(N_f^3 N^3) partial inverse transform per step. It is what
[`OU_advance!`](@ref) calls to replay a chain over tens of thousands of steps.

🔑 The split has to be exact, and the reason it is: the state update is the **only** part of a
forcing step that touches `rng`, so `n` state steps consume the same random stream as `n` full
forcing steps and leave `state` bit-identical. `f_hat` and `f` are left stale by a replay, which is
safe because `solve_unsteady` calls `OU_forcing_step!` followed by `OU_get_force!` before every use
of the body force (`solver.jl:61-63,87-89,102-104`). `lib/RikFlow/test/test_ou.jl` asserts both
properties against the real `OU_forcing_step!` rather than trusting this comment.
"""
function OU_state_step!(; ou_setup, Δt)
    (; T_L, Var, rng, num_dims, z) = ou_setup

    # Generate random numbers
    randn!(rng, z)
    # Update the state  shape: N_d x num_dims
    ou_setup.state[:,:] .= ou_setup.state[:,:] .*(1-Δt/T_L) .+ sqrt(2 * Var * Δt/T_L) .* z[:,1:num_dims] .+ 1im * sqrt(2 * Var * Δt/T_L) .* z[:, num_dims+1:2*num_dims]

    return ou_setup
end

"""
    OU_advance!(; ou_setup, Δt, n)

Replay the OU chain forward by `n` steps of size `Δt`, without the solver and without building the
forcing field. Returns `ou_setup`.

`OU_setup` starts every chain at `state .= 0` with `rng = Xoshiro(rng_seed)`, and
`OU_state_step!` is a pure Markov update reading only that `rng`, so the state after `n` steps is a
function of `(rng_seed, n, Δt)` and of nothing else -- not of the flow, the grid or the solver.
That is what licenses replay at all.

🔴 Why this exists. A run launched from a snapshot taken at reference step `n_k` must start with the
chain the reference had at that step. Launching from a zero state instead puts every member's
forcing `n_k` steps out of phase with the field it was handed, which inflates skill without
inflating spread and biases a spread-skill ratio **downward** -- toward a false "over-confident"
verdict. The existing HIT drivers are correct only because they launch from `fields[1]`, where
`n_1 = 0` and the zero state is the right state (`meta_files/handoff_p2c_d6.md` section 3 step 2).
"""
function OU_advance!(; ou_setup, Δt, n::Integer)
    n >= 0 || error("OU_advance!: n must be non-negative, got n = $n")
    for _ in 1:n
        OU_state_step!(; ou_setup, Δt)
    end
    return ou_setup
end

function OU_forcing_step!(; ou_setup, Δt)
    (; mask, num_dims, E, f_hat) = ou_setup

    OU_state_step!(; ou_setup, Δt)

    if num_dims ==2
        for d in 1:num_dims
            f_hat[d][mask] .= ou_setup.state[:,d]
            f_hat[d][:,end÷2+2:end] .= conj.(reverse(f_hat[d][:,1:end÷2], dims=2)) # fill the positive frequencies in the last dimension
            #@tensor ou_setup.f[d][b,c] = E[j,b]*E[k,c]*f_hat[d][k,j]
        end
    elseif num_dims == 3
        for d in 1:num_dims
            f_hat[d][mask] .= ou_setup.state[:,d]
            f_hat[d][:,:,end÷2+2:end] .= conj.(reverse(f_hat[d][:,:,1:end÷2], dims=3)) # fill the positive frequencies in the last dimension
            # @tensor ou_setup.f[d][a,b,c] = E[i,a]*E[j,b]*E[k,c]*f_hat[d][k,j,i] # this is slow! (but computational cost is also high: O(N_f^3*N^3))
            t1=sum( reshape(E,        size(E,1), 1, 1, size(E,2)).* 
                    reshape(f_hat[d], size(f_hat[d],1), size(f_hat[d],2), size(f_hat[d],3), 1), dims=1)
            t2=sum(reshape(E, 1, size(E,1), 1, size(E,2), 1).*
                reshape(t1, size(t1,1), size(t1,2), size(t1,3), 1, size(t1,4)), dims=2)
            t3=sum(reshape(E, 1, 1, size(E,1), size(E,2),1,1).*
                    reshape(t2, size(t2,1), size(t2,2), size(t2,3), 1, size(t2,4), size(t2,5)), dims=3)
            ou_setup.f[d] = reshape(t3, size(t3,4), size(t3,5), size(t3,6))
        end
    end
end

"""
    OU_get_force!(ou_setup, bodyforce, setup)

Write the current OU forcing field into `bodyforce`.

⚠️ `bodyforce` used to be `setup.bodyforce`. Upstream's `setup` is a pure grid description with no
body-force slot — forcing is a property of the right-hand side now, not of the setup — so the
buffer is passed explicitly and lives in the force cache (see [`ou_force_cache`](@ref)).
"""
function OU_get_force!(ou_setup, bodyforce, setup)
    (;Iu, dimension) = setup
    D = dimension()
    for d in 1:D
        bodyforce[Iu[d],d] = real(ou_setup.f[d])
    end
    #apply_bc_u!(bodyforce, t, setup)
end

"""
    ou_force_cache(setup; T_L, e_star, k_f, rng_seed, freeze)

Build the `force_cache` for [`ou_navierstokes!`](@ref): the OU chain, the body-force buffer it
writes into, and the `freeze` interval `solve_unsteady` gates the advance on.

Pass the result to `solve_unsteady(; force! = ou_navierstokes!, force_cache = ..., ...)`. Under
the old API this was `Setup(; ou_bodyforce = (; T_L, e_star, k_f, freeze, rng_seed))`; the
parameters and the seeding are unchanged, only where the state is kept.

`setup` must be a RikFlow-extended setup — upstream's `Setup` output with `Re` and `ArrayType`
added — because `OU_setup` takes its element type from `Re` and its array type from `ArrayType`.
"""
ou_force_cache(setup; T_L, e_star, k_f, rng_seed = 42, freeze = 1) = (;
    ou_setup = OU_setup(; T_L, e_star, k_f, setup, rng_seed, freeze),
    bodyforce = vectorfield(setup),
    freeze,
)

"""
    ou_navierstokes!(force, state, t; setup, cache, viscosity)

Navier-Stokes momentum forcing plus the OU body force. The `force!` argument of `solve_unsteady`
for every OU-forced case.

🔴 This adds the *current* forcing field; it does **not** advance the chain. `force!` runs once per
Runge-Kutta stage, so advancing here would advance four times per step under RK44. The advance is
in `solve_unsteady`, once per step — see the long comment there and gotcha #33.
"""
function ou_navierstokes!(force, state, t; setup, cache, viscosity)
    navierstokes!(force, state, t; setup, cache, viscosity)
    force.u .+= cache.bodyforce
end

export OU_setup, OU_state_step!, OU_advance!, OU_forcing_step!, OU_get_force!
export ou_force_cache, ou_navierstokes!
