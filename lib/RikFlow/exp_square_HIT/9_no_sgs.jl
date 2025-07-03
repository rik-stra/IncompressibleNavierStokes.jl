if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
end

using Random
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

no_model_folder = @__DIR__()*"/output/no_model"
track_file = @__DIR__()*"/output/data_track_tsim10.0.jld2" #we will take some parameters and the initial field from here
ispath(no_model_folder) || mkpath(no_model_folder)

# simulation parameters
Re = Float32(2_000);
Δt = Float32(2.5e-3);
tsim = Float32(100);
# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
freeze = 1 # number of time steps to freeze the forcing

# For running on a CUDA compatible GPU
T = Float32
ArrayType = CuArray
backend = CUDABackend()


seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)


# load reference data
data_track = load(track_file, "data_track");
params_track = load(track_file, "params_track");

# get initial condition
if data_track.fields[1].u isa Tuple
    ustart = stack(ArrayType.(data_track.fields[1].u));
elseif data_track.fields[1].u isa Array{<:Number,4}
    ustart = ArrayType(data_track.fields[1].u);
end

params = (;
    params_track...,
    tsim,
    Δt,
    ArrayType,
    ustart, 
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou),
    savefreq = 1000);

# Build setup and assemble operators
setup = Setup(;
    x = ntuple(α -> LinRange(params.lims[α]..., params.nles[1][α] + 1), params.D),
    Re=params.Re,
    ArrayType,
    backend = CUDABackend(),
    params.ou_bodyforce,
);

# Number of time steps to save
nt = round(Int, params.tsim / params.Δt)

to_setup_les = RikFlow.TO_Setup(; 
         params.qois, 
         to_mode = :CREATE_REF,  # allows us to save the scale-aware QoIs during the simulation
         params.ArrayType, 
         setup,
         nstep=nt);

psolver = psolver_spectral(setup);

# Solve
@info "Solving LF sim (no SGS)"
(; u, t), outputs = solve_unsteady(;
    # method = LMWray3(; T),
    setup, 
    ustart,
    tlims = (T(0), params.tsim),
    params.Δt,
    processors = (;
        log = timelogger(; nupdate = 100),
        fields = fieldsaver(; setup, nupdate = params.savefreq),  # by calling this BEFORE qoisaver, we also save the field at t=0!
        qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1),
    ),
    psolver,
);

q = stack(outputs.qoihist);
data_online = (;q, fields = outputs.fields);
# Save tracking data
jldsave(no_model_folder*"/data_no_sgs_tsim$(tsim).jld2"; data_online, params);