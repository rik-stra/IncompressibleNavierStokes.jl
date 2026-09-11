## Run one long simulation with no closure model

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2

# High-fidelity file for initial condition
hf_file = @__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2"
qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],["Z",11,17],["E", 11, 17]];
# Precision
T = Float64
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 5f # 100
Δt = 0.005f

nx_les = 64
ny_les = 64
nz_les = 32
ArrayType = CuArray
# Steady streamwise driving force for the channel. Was `Setup(; bodyforce, issteadybodyforce)`;
# upstream removed both, along with `applybodyforce!`, so it is a force cache now.
# ⚠️ The trailing `t` argument is gone: upstream builds the field with `velocityfield`, whose
# `ufunc` takes `(dim, x...)` only. A steady force never used it.
channel_bodyforce(dim, x, y, z) = 1 * (dim == 1)

kwargs = (;
    boundary_conditions = (; u = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    )),
    Re = 180f,
    backend = CUDABackend(),
    ArrayType = ArrayType,
)

setup = rf_setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);

@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"
psolver = psolver_transform(setup);

ustart = ArrayType(load(hf_file)["f"].data[1].u[1]);

to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = setup,
    mirror_y = true,);

@info "Solving LES"
(; u, t), outputs = solve_unsteady(;
    # Steady driving force, formerly Setup(; bodyforce, issteadybodyforce).
    # Without it the channel is unforced and decays to rest, silently.
    force! = rf_bodyforce_navierstokes!,
    force_cache = rf_steady_force_cache(to_setup_les, channel_bodyforce),
    setup,
    # LMWray3 by Rik's decision of 2026-09-11: stated, never inherited from the library default.
    method = LMWray3(; T = eltype(ustart)),
    start = (; u = ustart),
    params = rf_params(to_setup_les),
    docopy = false,
    tlims = (0f, tsim),
    Δt,
    processors = (;
        log = timelogger(; nupdate = 200),
        fields = fieldsaver(; setup, nupdate = 200),  # by calling this BEFORE qoisaver, we also save the field at t=0!
        qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1, nan_limit = 1f8),
    ),
    psolver,
);

outdir = @__DIR__()*"/output/nomodel"
ispath(outdir) || mkpath(outdir)
filename = "$outdir/LF_nomodel_channel_tsim$(tsim).jld2"
jldsave(filename; outputs.fields, outputs.qoihist)
