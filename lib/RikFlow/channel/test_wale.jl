using IncompressibleNavierStokes
#using CairoMakie
using CUDA
using CUDSS
#using AMGX
using RikFlow
using JLD2
using CairoMakie

# Precision
T = Float32
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

nx_les = 64
ny_les = 64
nz_les = 32
ArrayType = CuArray
kwargs = (;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re = 180f,
    bodyforce = (dim, x, y, z, t) -> 1 * (dim == 1),
    issteadybodyforce = true,
    backend = CUDABackend(),
    ArrayType = ArrayType,
)

setup = Setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz_les + 1),
    ),
    kwargs...,
);

psolver = default_psolver(setup);

ustart = ArrayType(
    load(
        @__DIR__() * "/output/HF_channel_mirror_256_256_128_to_64_64_32_tsim10.0.jld2",
    )["f"].data[1].u[1],
);

(; dimension, x, N) = setup.grid
D = dimension()
T = eltype(x[1])
G = (;
    xx = scalarfield(setup),
    yx = scalarfield(setup),
    zx = scalarfield(setup),
    xy = scalarfield(setup),
    yy = scalarfield(setup),
    zy = scalarfield(setup),
    xz = scalarfield(setup),
    yz = scalarfield(setup),
    zz = scalarfield(setup),
)

c = vectorfield(setup)

#strain_natural!(S, u, setup)
u = ustart;
@show minimum(u), maximum(u)
@show minimum(G.xx)

heatmap(visc[:, :, 10] |> Array)

heatmap(visc[:, :, 10] |> Array)



visc = scalarfield(setup)

IncompressibleNavierStokes.gradient_tensor!(G, u, setup);
# IncompressibleNavierStokes.apply_bc_tensor!(G, 0f, setup)

IncompressibleNavierStokes.wale_viscosity!(visc, G, 0.6, setup)
visc[:,2,10]
IncompressibleNavierStokes.apply_bc_p!(visc, 0f, setup)

IncompressibleNavierStokes.strain_from_gradient!(G)
IncompressibleNavierStokes.apply_eddy_viscosity!(G, visc, setup)

G.xy[:, :, 10] |> Array |> heatmap
G.xx[1, :, 10]


IncompressibleNavierStokes.divoftensor_natural!(c, G, setup)
c[:, :, 10, 3] |> Array |> heatmap
c[:, 2, 10, 1]


visc[:, 2, 10]
