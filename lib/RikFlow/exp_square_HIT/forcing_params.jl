# compute Re based on forcing parameters
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate

sigma = sqrt(e_star/T_L)

nu = 1/2_000 # kinematic viscosity
N_points = 512 # number of grid points

k_0 = 1.0
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
