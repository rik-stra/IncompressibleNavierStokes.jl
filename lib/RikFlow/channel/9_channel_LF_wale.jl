## Run 1 long simulations with WALE closure model and optimal WALE constant

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2

# WALE constant
c = 0.53
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

to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = setup,
    mirror_y = true,);

@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"
psolver = psolver_transform(setup);
ustart = ArrayType(load(hf_file)["f"].data[1].u[1]);

@info "Solving LES"
# Solve DNS and store filtered quantities
(; u, t), outputs = solve_unsteady(;
    # setup,
    setup = (; setup..., closure_model = IncompressibleNavierStokes.wale_closure),
    θ = T(c), 
    ustart,
    tlims = (0f, tsim),
    Δt,
    processors = (;
        log = timelogger(; nupdate = 100),
        fields = fieldsaver(; setup, nupdate = 200),  # by calling this BEFORE qoisaver, we also save the field at t=0!
        qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1, nan_limit = 1f8),
    ),
    psolver,
);

# Save filtered DNS data
outdir = @__DIR__()*"/output/WALE"
filename = "$outdir/LF_wale_channel_c$(c)_tsim$(tsim).jld2"
jldsave(filename; outputs.fields, outputs.qoihist)
