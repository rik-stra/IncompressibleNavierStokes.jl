using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2
#using LoggingExtras


# Precision
T = Float64
f = one(T)

#### small test
xlims = 0f, 2f*pi
ylims = 0f, 2f*pi
zlims = 0f, 2f*pi

# Grid
nx = 512
ny = 512
nz = 512

nx_les = 64
ny_les = 64
nz_les = 64

Re = 1_600f

kwargs = (;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re,
    backend = CUDABackend(),
)

setup = Setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
);

les_setup = Setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1),
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);
@info "Grid size HF: $(nx) x $(ny) x $(nz)"
@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"


qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],
        ["Z",11,17],["E", 11, 17]];
ArrayType = CuArray


U(dim, x, y, z) =
    if dim == 1
        sin(x) * cos(y) * sin(z)
    elseif dim == 2
        -cos(x) * sin(y) * sin(z)
    else
        zero(x)
    end
ustart = velocityfield(setup, U, psolver = nothing, doproject=false);

u_les = vectorfield(les_setup)
ϕ = FaceAverage()
ϕ(u_les, ustart, les_setup, Int(nx/nx_les));
IncompressibleNavierStokes.apply_bc_u!(u_les, 0, les_setup)

psolver = psolver_spectral(les_setup);
u_les_projected = project(u_les, les_setup, psolver=psolver);

using CairoMakie
heatmap(Array(u_les)[:,:,5,1])
heatmap(Array(u_les_projected)[:,:,5,1])

file_name = @__DIR__() * "/output/filtered_initial_field.jld2"
u_start = Array(u_les)
jldsave(file_name; u_start)