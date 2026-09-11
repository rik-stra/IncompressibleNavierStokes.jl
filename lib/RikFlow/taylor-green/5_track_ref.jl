## Taylor-Green LF sim

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

#### small test
xlims = 0f, 2f*pi
ylims = 0f, 2f*pi
zlims = 0f, 2f*pi

Re = 800f
tsim = 20f
# Grid
nx_les = 64
ny_les = 64
nz_les = 64
Δt = 0.05f

qois = [["Z",0,1],["E", 0, 1],["Z",2,3],["E", 2, 3],["Z",4,5],["E", 4, 5]];

hf_file = @__DIR__() *"/output/HF/HF_TG_512_to_64_Re_800.0_tsim20.0.jld2"
qoi_ref = stack(load(hf_file)["f"].data[1].qoi_hist[:]);
sample_rate = 10 # In the HF simulation we saved every time step, now we take 10 times bigger time steps
qoi_ref = qoi_ref[:,1:sample_rate:end]
ref_reader = Reference_reader(qoi_ref);
ArrayType = CuArray
kwargs = (;
    boundary_conditions = (; u = (
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
    )),
    Re,
    backend = CUDABackend(),
    ArrayType,
)
les_setup = rf_setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1),
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);

@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"

psolver = psolver_spectral(les_setup);

u_start_file_name = @__DIR__() *"/output/filtered_initial_field.jld2"
ustart = ArrayType(load(u_start_file_name, "u_start"));

nt = round(Int, tsim / Δt)
to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :TRACK_REF, 
    ArrayType, 
    setup = les_setup,
    nstep=nt,
    time_series_method = ref_reader,
    );


@info "Solving LES"
# Solve DNS and store filtered quantities
(; u, t), outputs = solve_unsteady(;
    setup = les_setup,
    start = (; u = ustart),
    params = rf_params(les_setup),
    docopy = true,
    method = TOMethod(; to_setup = to_setup_les),
    tlims = (0f, tsim),
    Δt,
    processors = (;
        log = timelogger(; nupdate = 10),
        fields = fieldsaver(; setup=les_setup, nupdate = 10),  # by calling this BEFORE qoisaver, we also save the field at t=0!
        qoihist = RikFlow.qoisaver(; setup=les_setup, to_setup=to_setup_les, nupdate = 1),
    ),
    psolver,
);

q = stack(outputs.qoihist)
dQ = to_setup_les.outputs.dQ
tau = to_setup_les.outputs.tau
q_star = to_setup_les.outputs.q_star
data_train = (;dQ, tau, q, q_star)
# Save filtered DNS data
outdir = @__DIR__() *"/output/LF/track"
ispath(outdir) || mkpath(outdir)
filename = "$outdir/track_TG_$(nx_les)_Re_$(Re)_tsim$(tsim).jld2"

jldsave(filename; data_train, outputs.fields)