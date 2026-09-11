if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
end

using Random
using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA

smag_folder = @__DIR__()*"/output/smag"
track_file = @__DIR__()*"/output/data_track_tsim10.0.jld2" #we will take some parameters and the initial field from here
ispath(smag_folder) || mkpath(smag_folder)

smag_vals = [0.071]
# simulation parameters
T = Float64
Re = T(2_000);
Δt = T(2.5e-3);
tsim = T(100);
# forcing
T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
freeze = 1 # number of time steps to freeze the forcing

# For running on a CUDA compatible GPU
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
    ustart = stack(ArrayType{T}.(data_track.fields[1].u));
elseif data_track.fields[1].u isa Array{<:Number,4}
    ustart = ArrayType{T}(data_track.fields[1].u);
end

params = (;
    params_track...,
    # 🔴 Override the archived Re. The splat above carries the archive's Float32 parameters, and a
    # later key wins — without this the setup is built at Float32 while the script declares
    # Float64, and `typeof(setup.Re)` silently drives every QoI buffer back to single precision.
    # `rf_setup` now refuses that mismatch outright, so this is what keeps the script runnable.
    Re = T(2_000),
    tsim,
    Δt,
    ArrayType,
    ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou),
    savefreq = 1000);


for c_s in smag_vals
    # Build setup and assemble operators
    setup = rf_setup(;
        x = ntuple(α -> LinRange(params.lims[α]..., params.nles[1][α] + 1), params.D),
        Re=params.Re,
        ArrayType,
        backend,
    );

    # The closure moved from `setup.closure_model` + `θ` into the right-hand side and its cache.
    # 🔴 These are upstream's Smagorinsky kernels, not the `smagorinsky_closure_natural` this
    # script used before the merge (map section 9, Q2) - a different implementation of the same
    # model, so this baseline is not numerically the one paper 2 reports.
    force_cache = rf_smag_force_cache(setup; c_s = T(c_s), params.ou_bodyforce);

    # Number of time steps to save
    nt = round(Int, params.tsim / params.Δt)

    to_setup_les = RikFlow.TO_Setup(; 
            params.qois, 
            to_mode = :CREATE_REF,
            params.ArrayType, 
            setup,
            nstep=nt);

    psolver = psolver_spectral(setup);

    # Solve
    @info "Solving LF sim (SMAG)"
    (; u, t), outputs = solve_unsteady(; 
            # LMWray3 by Rik's decision of 2026-09-11: stated, never inherited from the library
            # default. Reproducing an archived run means passing RKMethods.RK44 explicitly.
            method = LMWray3(; T = eltype(ustart)),
            setup,
            start = (; u = ustart),
            force! = rf_smag_navierstokes!,
            force_cache,
            params = rf_params(setup),
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
    jldsave(smag_folder*"/data_smag_$(c_s)_tsim$(tsim).jld2"; data_online, params);
end