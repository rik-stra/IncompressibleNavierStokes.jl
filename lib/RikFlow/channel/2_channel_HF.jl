## Turbulent channel flow

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
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 10f
tspin = 15f
# Grid
nx = 512      #-> highest wave number 128/4pi = 10.2
ny = 512      #-> highest wave number 128/2 = 64
nz = 256      #-> highest wave number 64/(4/3*pi) = 15.3
Δt = 0.0005f

nx_les = 64
ny_les = 64
nz_les = 32

#### small test
# tsim = 5f
# tspin = 5f
# # Grid
# nx = 64   
# ny = 64     
# nz = 32      
# Δt = 0.005f

# nx_les = 8
# ny_les = 8
# nz_les = 4


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
)

setup = rf_setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
);

les_setup = rf_setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);
@info "Grid size HF: $(nx) x $(ny) x $(nz)"
@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"

psolver = psolver_transform(setup);

qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],
        ["Z",11,17],["E", 11, 17]];
ArrayType = CuArray

ustart = ArrayType(load(@__DIR__()*"/output/HF/u_start_T$(Int(tspin))_$(nx)_$(ny)_$(nz).jld2", "u_start"));

to_setup_les = 
    RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = les_setup,
    mirror_y = true,);

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
    # Steady driving force, formerly Setup(; bodyforce, issteadybodyforce).
    # Without it the channel is unforced and decays to rest, silently.
    force! = rf_bodyforce_navierstokes!,
    force_cache = rf_steady_force_cache(setup, channel_bodyforce),
    setup,
    # Upstream changed the default from RKMethods.RK44 to LMWray3; pinned.
    method = RKMethods.RK44(; T = eltype(ustart)),
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
            [8,],
            [to_setup_les,];
            nupdate = 2,
            n_plot = 2000,
            checkpoints,
            checkpoint_name = checkpoints_dir,
        ),
        log = timelogger(; nupdate = 400),
        fields = fieldsaver(; nupdate = round(Int,nt/3), setup),  # 1.6 GB per snapshot!
        ),
    psolver,
);


# Save filtered DNS data
filename = "$outdir/HF_channel_$(nx)_$(ny)_$(nz)_to_$(nx_les)_$(ny_les)_$(nz_les)_tsim$(tsim).jld2"

jldsave(filename; outputs.f)