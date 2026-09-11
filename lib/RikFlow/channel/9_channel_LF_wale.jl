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
    setup = setup,
    # Closure moved from setup.closure_model + theta into the right-hand side.
    # Upstream's kernels, not this fork's (map section 9, Q2).
    force! = rf_eddyvisc_navierstokes!,
    force_cache = rf_eddyvisc_force_cache(setup; model = WALE(T(c)), bodyforce = channel_bodyforce),
    # LMWray3 by Rik's decision of 2026-09-11: stated, never inherited from the library default.
    method = LMWray3(; T = eltype(ustart)),
    start = (; u = ustart),
    params = rf_params(setup),
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
