if false
    include("src/OUForcer.jl")
    include("../../src/IncompressibleNavierStokes.jl")
end

using OUForcer
using CairoMakie #!md
using IncompressibleNavierStokes
using CUDA
using Random
using FFTW
using Statistics


ArrayType = CuArray

T = Float64


# ## Setup
#
# Define a uniform grid with a steady body force field.

n = 32
axis = range(0.0, 1., n + 1)
setup = Setup(;
    x = (axis, axis, axis),
    Re = 5e3,
    bodyforce = (dim, x, y, z, t) -> (dim == 1) * 0.5 * sinpi(2*y),
    issteadybodyforce = true,
    ArrayType = ArrayType,
);

e_star = 0.05
T_L = 0.1
k_f = sqrt(2)

ou_setup = OU_setup(; T_L, 
                    e_star, 
                    k_f,
                    setup, 
                    rng = Xoshiro(42),
                    );

forces = Array{ComplexF32}(undef, n, n, 10000)
for i in 1:10000                    
    OU_forcing_step!(; ou_setup, Δt=0.1)
    forces[:,:,i].=Array(ou_setup.f[1])[:,:,5]
end
fig, ax, hm = heatmap(real.(forces[:,:,620]))
Colorbar(fig[:,end+1], hm, label="Force")
fig

mean_force = mean(forces, dims=3)
fig, ax, hm = heatmap(real.(mean_force[:,:,1]))
Colorbar(fig[:,end+1], hm, label="Mean Force")
fig
