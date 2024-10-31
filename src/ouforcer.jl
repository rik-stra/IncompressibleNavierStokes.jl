
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
                    rng = Xoshiro(42),
                    )

T = typeof(setup.Re)
ArrayType = setup.ArrayType
num_dims = setup.grid.dimension()
Var = e_star/T_L

# check if grid is equidistant
#@assert all([all(Δ ≈ Δ[1]) for Δ in setup.grid.Δ])
#@assert all([ (Δ[1] ≈ setup.grid.Δ[1][1]) for Δ in setup.grid.Δ])
N = setup.grid.Nu[1][1]

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

f = [ArrayType{ComplexF32, num_dims}(undef, setup.grid.Nu[a]...) for a = 1:num_dims] # contains the forcing in physical space

# create partial IFFT matrix
E = Array{ComplexF32,2}(undef, 2*k_f_int+1, N)
for j = 1:N, i = -k_f_int:k_f_int
    E[i+k_f_int+1, j] = exp(pi*2im*(i)*(j-1)/N)
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
    z
)
end

function OU_forcing_step!(; ou_setup, Δt)
    (; T_L, Var, N_f, mask, rng, num_dims, E, z, f_hat) = ou_setup

    # Generate random numbers
    randn!(rng, z)
    # Update the state  shape: N_d x num_dims
    ou_setup.state[:,:] .= ou_setup.state[:,:] .*(1-Δt/T_L) .+ sqrt(2 * Var * Δt/T_L) .* z[:,1:num_dims] .+ 1im * sqrt(2 * Var * Δt/T_L) .* z[:, num_dims+1:2*num_dims]

        
    if num_dims ==2
        for d in 1:num_dims
            f_hat[d][mask] .= ou_setup.state[:,d]
            f_hat[d][:,end÷2+2:end] .= conj.(reverse(f_hat[d][:,1:end÷2], dims=2)) # fill the positive frequencies in the last dimension
            @tensor ou_setup.f[d][b,c] = E[j,b]*E[k,c]*f_hat[d][k,j]
        end
    elseif num_dims == 3
        for d in 1:num_dims
            f_hat[d][mask] .= ou_setup.state[:,d]
            f_hat[d][:,:,end÷2+2:end] .= conj.(reverse(f_hat[d][:,:,1:end÷2], dims=3)) # fill the positive frequencies in the last dimension
            @tensor ou_setup.f[d][a,b,c] = E[i,a]*E[j,b]*E[k,c]*f_hat[d][k,j,i] # this is slow! (but computational cost is also high: O(N_f^3*N^3))
        end
    end
end

function OU_get_force!(ou_setup, setup)
    (;Iu, dimension) = setup.grid
    D = dimension()
    for d in 1:D
        setup.bodyforce[d][Iu[d]] = real(ou_setup.f[d])
    end
    #apply_bc_u!(setup.bodyforce, t, setup)
end



export OU_setup, OU_forcing_step!, OU_get_force!
