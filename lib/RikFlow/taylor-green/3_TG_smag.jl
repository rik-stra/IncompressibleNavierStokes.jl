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



kwargs = (;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re,
    backend = CUDABackend(),
)
les_setup = Setup(;
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

c_vals = 0.01:0.01:0.15

for c_s in c_vals
    closure_model = IncompressibleNavierStokes.smagorinsky_closure_natural;
    les_setup = (; les_setup..., closure_model);

    to_setup_les = 
        RikFlow.TO_Setup(; qois, 
        to_mode = :CREATE_REF, 
        ArrayType, 
        setup = les_setup,
        );


    @info "Solving LES"
    # Solve DNS and store filtered quantities
    (; u, t), outputs = solve_unsteady(;
        setup = les_setup,
        θ = T(c_s),
        ustart,
        docopy = true,
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
    outdir = @__DIR__() *"/output/LF/smag"
    ispath(outdir) || mkpath(outdir)
    filename = "$outdir/smag_TG_$(c_s)_$(nx_les)_Re_$(Re)_tsim$(tsim).jld2"

    jldsave(filename; outputs.qoihist, outputs.fields)
end