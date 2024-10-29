# compute Re based on forcing parameters
T_L = 0.05  # correlation time of the forcing
e_star = 0.2 # energy injection rate

sigma = sqrt(e_star/T_L)

nu = 0.001 # kinematic viscosity
N_points = 512 # number of grid points

k_0 = 2*pi/1.0
k_f = 2.5*k_0
N_f = 80
β = 0.8

T_star_L = T_L * e_star^(1/3) * k_0^(2/3)
e_star_T = 4.0 * e_star * N_f / (1+T_star_L*N_f^(1/3)/β)
η_T = (nu^3/e_star_T)^(1/4) # Kolmogorov scale
dx = 1/N_points

L_c = pi / (k_0+k_f)
Re_lab = (20 * L_c * (T_L * e_star_T)^(1/2) / (3*nu))^(1/2)

Re_lab_old = 8.5/((η_T*k_0)^(5/6) * N_f^(2/9))

# programm OU process
using Random
using CairoMakie
T_L = 0.05  # correlation time of the forcing
δt = 0.001
e_star = 0.2 # energy injection rate
Sigma = e_star/T_L
rng = Xoshiro(123)

state = Array{Float64}(undef, 6)
state[:] .= 0.0
n_steps = 1000
vals = Array{Float64}(undef, (6,n_steps))
for i in 1:n_steps
    state = state.*(1-δt/T_L) .+ sqrt(2.0*Sigma * δt/T_L) .* randn(rng, 6)
    vals[:,i] .= state
end
let fig = Figure()
    ax = Axis(fig[1,1])
    for i in 1:2
        lines!(ax,vals[i,:])
    end
    fig
end