## Taylor-Green HF sim

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2
#using LoggingExtras

# jobid = ENV["SLURM_JOB_ID"]

# logfile = joinpath(@__DIR__, "log_$(jobid).out")
# filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
# logger = TeeLogger(global_logger(), filelogger)
# global_logger(logger)

# Precision
T = Float64
f = one(T)

# Domain
xlims = 0f, 1f
ylims = 0f, 1f
zlims = 0f, 1f

tsim = 10f
# Grid
nx = 64      
ny = 64      
nz = 64      
Δt = 0.0005f

nx_les = 32
ny_les = 32
nz_les = 32

#### small test
xlims = 0f, 1f
ylims = 0f, 1f
zlims = 0f, 1f

tsim = 10f
# Grid
nx = 128      
ny = 128     
nz = 128     
Δt = 0.0025f
nx_les = 64
ny_les = 64
nz_les = 64

kwargs = (;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re = 1000f,
    backend = CUDABackend(),
)

setup = Setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
);

les_setup = Setup(;
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

qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],
        ["Z",11,17],["E", 11, 17]];
ArrayType = CuArray


U(dim, x, y, z) =
    if dim == 1
        sinpi(2x) * cospi(2y) * sinpi(2z)
    elseif dim == 2
        -cospi(2x) * sinpi(2y) * sinpi(2z)
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
    ustart,
    docopy = false,
    tlims = (0f, tsim),
    Δt,
    processors = (;
        f = RikFlow.filtersaver(
            setup,
            [les_setup,],
            (FaceAverage(),),
            [2,],
            [to_setup_les,];
            nupdate = 1,
            n_plot = 40,
            checkpoints,
            checkpoint_name = checkpoints_dir,
        ),
        log = timelogger(; nupdate = 10),
        fields = fieldsaver(; nupdate = round(Int,nt/3), setup),
        ),
    psolver,
);


# Save filtered DNS data
filename = "$outdir/HF_TG_$(nx)_to_$(nx_les)_tsim$(tsim).jld2"

jldsave(filename; outputs.f, outputs.fields)



