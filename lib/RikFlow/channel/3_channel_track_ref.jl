## Turbulent channel flow

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2

# Precision
T = Float64
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi
nx_les = 64
ny_les = 64
nz_les = 32
tsim = 10f
Δt = 0.005f
qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],
        ["Z",11,17],["E", 11, 17]];
hf_file = @__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2"

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

psolver = psolver_transform(setup)


ustart = ArrayType(load(hf_file)["f"].data[1].u[1]);
qoi_ref = stack(load(hf_file)["f"].data[1].qoi_hist[1:10001]);
sample_rate = 5 # In the HF simulation we saved every second time step, now we take 10 times bigger time steps
qoi_ref = qoi_ref[:,1:sample_rate:end]
ref_reader = Reference_reader(qoi_ref);

nt = round(Int, tsim / Δt)

to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :TRACK_REF, 
    ArrayType, 
    setup,
    nstep=nt,
    time_series_method = ref_reader,
    mirror_y = true,);

outdir = @__DIR__() *"/output/track"
ispath(outdir) || mkpath(outdir)


@info "Solving LES"
(; u, t), outputs = solve_unsteady(;
    setup,
    ustart,
    docopy = false,
    method = TOMethod(; to_setup = to_setup_les),
    tlims = (0f, tsim),
    Δt,
    processors = (;
        log = timelogger(; nupdate = Int(100/sample_rate)),
        fields = fieldsaver(; setup, nupdate = Int(1000/sample_rate)),  # by calling this BEFORE qoisaver, we also save the field at t=0!
        qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1, nan_limit = 1f7),
    ),
    psolver,
);

q = stack(outputs.qoihist)
dQ = to_setup_les.outputs.dQ
tau = to_setup_les.outputs.tau
q_star = to_setup_les.outputs.q_star
fields = outputs.fields
data_train = (;dQ, tau, q, q_star, fields)

# Save filtered DNS data
filename = "$outdir/LF_6qoi_track_channel_to_$(nx_les)_$(ny_les)_$(nz_les)_dt$(Δt)_tsim$(tsim).jld2"
jldsave(filename; data_train)
