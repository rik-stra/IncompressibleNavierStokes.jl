## Turbulent channel flow

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
#using CairoMakie
using CUDA
#using CUDSS
using AMGX
using RikFlow
using JLD2
using LoggingExtras


# Precision
T = Float32
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 10f
# Grid
nx = 256      #-> highest wave number 128/4pi = 10.2
ny = 256      #-> highest wave number 128/2 = 64
nz = 128      #-> highest wave number 64/(4/3*pi) = 15.3
Δt = 0.001f

nx_les = 64
ny_les = 64
nz_les = 32

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
)

setup = Setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
);

les_setup = Setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);
@info "Grid size HF: $(nx) x $(ny) x $(nz)"
@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"
amgx_objects = amgx_setup();
psolver = psolver_cg_AMGX(setup; stuff=amgx_objects);

qois = [["Z",0,6],["E", 0, 6],["Z",7,16],["E", 7, 16]];
ArrayType = CuArray

ustart = ArrayType(load(@__DIR__()*"/output/u_start_256_256_128_tspin10.0.jld2", "u_start"));

to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = les_setup,
    mirror_y = true,);

#determine checkpoints
n_checkpoints = 3
nt = round(Int, tsim / Δt)
checkpoints= 0:round(nt/(n_checkpoints+1)):nt
checkpoints = checkpoints[2:end-1]
checkpoints_dir = @__DIR__() *"/output/checkpoints"
outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)
ispath(checkpoints_dir) || mkpath(checkpoints_dir)


@info "Solving DNS"
# Solve DNS and store filtered quantities
(; u, t), outputs = solve_unsteady(;
    setup,
    ustart,
    docopy = false,
    tlims = (0f, tsim),
    Δt,
    processors = (;
        f = RikFlow.filtersaver(
            setup,
            [les_setup,],
            (FaceAverage(),),
            [4,],
            [to_setup_les,];
            nupdate = 1,
            n_plot = 1000,
            checkpoints,
            checkpoint_name = checkpoints_dir,
        ),
        log = timelogger(; nupdate = 100),
    ),
    psolver,
);
close_amgx(amgx_objects)
# Save filtered DNS data
filename = "$outdir/HF_channel_mirror_1framerate_$(nx)_$(ny)_$(nz)_to_$(nx_les)_$(ny_les)_$(nz_les)_tsim$(tsim).jld2"
jldsave(filename; outputs.f)

exit()

a = load(filename)
keys(a["f"].data[1])
a["f"].data[1].qoi_hist

# u_start low fidelity
u_lf = a["f"].data[1].u[1]
u_hf = load(@__DIR__()*"/output/u_start_256_256_128_tspin10.0.jld2", "u_start")

using CairoMakie
heatmap(u_lf[:,:,1,1])
heatmap(u_hf[:,:,1,1])

total_kinetic_energy(ArrayType(u_hf), setup)