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



kwargs = (;
    boundary_conditions = (; u = (
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
    )),
    Re,
    backend = CUDABackend(),
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

qois = [["Z",0,1],["E", 0, 1],["Z",2,3],["E", 2, 3],["Z",4,5],["E", 4, 5]];


ArrayType = CuArray


u_start_file_name = @__DIR__() *"/output/filtered_initial_field.jld2"
ArrayType = CuArray
ustart = ArrayType(load(u_start_file_name, "u_start"));



to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = les_setup,
    );

#determine checkpoints


@info "Solving LES"
# Solve DNS and store filtered quantities
(; u, t), outputs = solve_unsteady(;
    setup = les_setup,
    # Upstream changed the default from RKMethods.RK44 to LMWray3; pinned.
    method = RKMethods.RK44(; T = eltype(ustart)),
    start = (; u = ustart),
    params = rf_params(les_setup),
    docopy = false,
    tlims = (0f, tsim),
    Δt,
    processors = (;
        log = timelogger(; nupdate = 10),
        fields = fieldsaver(; setup=les_setup, nupdate = 10),  # by calling this BEFORE qoisaver, we also save the field at t=0!
        qoihist = RikFlow.qoisaver(; setup=les_setup, to_setup=to_setup_les, nupdate = 1),
    ),
    psolver,
);


# Save filtered DNS data
outdir = @__DIR__() *"/output/LF"
ispath(outdir) || mkpath(outdir)
filename = "$outdir/LF_TG_$(nx_les)_Re_$(Re)_tsim$(tsim).jld2"

jldsave(filename; outputs.qoihist, outputs.fields)