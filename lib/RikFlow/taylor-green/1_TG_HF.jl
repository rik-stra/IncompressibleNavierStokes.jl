## Taylor-Green HF sim

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2
using LoggingExtras

jobid = ENV["SLURM_JOB_ID"]

logfile = joinpath(@__DIR__, "log_$(jobid).out")
filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
logger = TeeLogger(global_logger(), filelogger)
global_logger(logger)

# Precision
T = Float64
f = one(T)

#### small test
xlims = 0f, 2f*pi
ylims = 0f, 2f*pi
zlims = 0f, 2f*pi

Re = 1_000f
tsim = 20f
# Grid
nx = 512      
ny = 512     
nz = 512     
Δt = 0.005f
nx_les = 64
ny_les = 64
nz_les = 64


kwargs = (;
    boundary_conditions = (; u = (
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
    )),
    Re,
    backend = CUDABackend(),
)

setup = rf_setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
);

les_setup = rf_setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1),
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);
@info "Grid size HF: $(nx) x $(ny) x $(nz)"
@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"

psolver = psolver_spectral(setup);

qois = [["Z",0,1],["E", 0, 1],["Z",2,3],["E", 2, 3],["Z",4,5],["E", 4, 5]];
ArrayType = CuArray


U(dim, x, y, z) =
    if dim == 1
        sin(x) * cos(y) * sin(z)
    elseif dim == 2
        -cos(x) * sin(y) * sin(z)
    else
        zero(x)
    end
ustart = velocityfield(setup, U, psolver);


to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = les_setup,
    );

#determine checkpoints
n_checkpoints = 0
nt = round(Int, tsim / Δt)
checkpoints= 0:round(nt/(n_checkpoints+1)):nt
checkpoints = checkpoints[2:end-1]
checkpoints_dir = @__DIR__() *"/output/HF/checkpoints"
outdir = @__DIR__() *"/output/HF"
ispath(outdir) || mkpath(outdir)
ispath(checkpoints_dir) || mkpath(checkpoints_dir)


@info "Solving DNS"
# Solve DNS and store filtered quantities
(; u, t), outputs = solve_unsteady(;
    setup,
    # LMWray3 by Rik's decision of 2026-09-11: stated, never inherited from the library default.
    method = LMWray3(; T = eltype(ustart)),
    start = (; u = ustart),
    params = rf_params(setup),
    docopy = false,
    tlims = (0f, tsim),
    Δt,
    processors = (;
        f = RikFlow.filtersaver(
            setup,
            [les_setup,],
            (FaceAverage(),),
            [Int(nx/nx_les),],
            [to_setup_les,];
            nupdate = 1,
            n_plot = 100,
            checkpoints,
            checkpoint_name = checkpoints_dir,
        ),
        log = timelogger(; nupdate = 10),
        fields = fieldsaver(; nupdate = round(Int,nt/2), setup),
        ),
    psolver,
);


# Save filtered DNS data
filename = "$outdir/HF_TG_$(nx)_to_$(nx_les)_Re_$(Re)_tsim$(tsim).jld2"

jldsave(filename; outputs.f, outputs.fields)



