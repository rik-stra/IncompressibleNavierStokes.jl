## Turbulent channel flow

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
#using CairoMakie
using CUDA
using CUDSS
#using AMGX
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
Δt = 0.01f

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
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);


@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"
#amgx_objects = amgx_setup();
psolver = default_psolver(setup);

qois = [["Z",0,6],["E", 0, 6],["Z",7,16],["E", 7, 16]];


ustart = ArrayType(load(@__DIR__()*"/output/HF_channel_mirror_256_256_128_to_64_64_32_tsim10.0.jld2")["f"].data[1].u[1]);


to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = setup,
    mirror_y = true,);

#determine checkpoints
outdir = @__DIR__() *"/output"
ispath(outdir) || mkpath(outdir)



@info "Solving LES"
# Solve DNS and store filtered quantities
(; u, t), outputs = solve_unsteady(;
    # setup,
    setup = (; setup..., closure_model = IncompressibleNavierStokes.smagorinsky_closure),
    θ = T(0.11), 
    ustart,
    tlims = (0f, tsim),
    Δt,
    processors = (;
        log = timelogger(; nupdate = 100),
        fields = fieldsaver(; setup, nupdate = 100),  # by calling this BEFORE qoisaver, we also save the field at t=0!
        qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1, nan_limit = 1f7),
    ),
    psolver,
);

#close_amgx(amgx_objects)

q = stack(outputs.qoihist)


# Save filtered DNS data
filename = "$outdir/LF_smag_mirror_channel_to_$(nx_les)_$(ny_les)_$(nz_les)_tsim$(tsim).jld2"
jldsave(filename; outputs.fields, outputs.qoihist)

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