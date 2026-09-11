using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA
using Random

T = Float64
Re = T(2_000);
Δt = T(2.5e-3);
tsim = T(1);
nles = 64

T_L = 0.01  # correlation time of the forcing
e_star = 0.1 # energy injection rate
k_f = sqrt(2) # forcing wavenumber  
freeze = 1 # number of time steps to freeze the forcing


track_file = @__DIR__()*"/../exp_square_HIT/output/data_track_tsim10.0.jld2"

ArrayType = CuArray
backend = CUDABackend()
seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)

# get initial condition
data_track = load(track_file, "data_track");
if data_track.fields[1].u isa Tuple
    ustart = stack(ArrayType{T}.(data_track.fields[1].u));
elseif data_track.fields[1].u isa Array{<:Number,4}
    ustart = ArrayType{T}(data_track.fields[1].u);
end
lims = ( (T(0) , T(1)) , (T(0) , T(1)), (T(0),T(1)) )
ou_bodyforce = (;T_L, e_star, k_f, freeze, rng_seed = seeds.ou )

setup = rf_setup(;
        x = ntuple(α -> LinRange(lims[α]..., nles + 1), 3),
        Re=Re,
        ArrayType,
        backend,
    );

psolver = psolver_spectral(setup);


@info "Solving LF sim (SMAG)"
(; u, t), outputs = solve_unsteady(; 
        setup = setup,
        # Closure moved from setup.closure_model + theta into the right-hand side.
        # Upstream's kernels, not this fork's (map section 9, Q2).
        force! = rf_eddyvisc_navierstokes!,
        # 🔴 `ou_bodyforce` used to ride along in the setup, so these timings were for a forced
        # solve. Upstream's setup has no forcing slot; dropping it here would have quietly turned
        # every run in this file into an unforced one and made the numbers incomparable.
        force_cache = rf_eddyvisc_force_cache(setup; model = Smagorinsky(T(0.071)), ou_bodyforce),
        # LMWray3 + Float64 is the production configuration (Rik, 2026-09-11).
        method = LMWray3(; T = eltype(ustart)),
        start = (; u = ustart),
        params = rf_params(setup),
        docopy = true,
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 300),
            timer = RikFlow.solver_timer(; n_steps = 100, n_warmup = 110),
        ),
        psolver,
);

elapsed = outputs.timer[2]-outputs.timer[1]
println("elapsed time:", elapsed, " s")

io = open(@__DIR__()*"/time_results.txt", "a")
write(io, "HIT_LF_smag_natural nles: $nles, elapsed_time: $elapsed  \n")
close(io)

@info "Solving LF sim (SMAG)"
(; u, t), outputs = solve_unsteady(; 
        setup = setup,
        # Closure moved from setup.closure_model + theta into the right-hand side.
        # Upstream's kernels, not this fork's (map section 9, Q2).
        force! = rf_eddyvisc_navierstokes!,
        # 🔴 `ou_bodyforce` used to ride along in the setup, so these timings were for a forced
        # solve. Upstream's setup has no forcing slot; dropping it here would have quietly turned
        # every run in this file into an unforced one and made the numbers incomparable.
        force_cache = rf_eddyvisc_force_cache(setup; model = Smagorinsky(T(0.071)), ou_bodyforce),
        # LMWray3 + Float64 is the production configuration (Rik, 2026-09-11).
        method = LMWray3(; T = eltype(ustart)),
        start = (; u = ustart),
        params = rf_params(setup),
        docopy = true,
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 300),
            timer = RikFlow.solver_timer(; n_steps = 100, n_warmup = 110),
        ),
        psolver,
);

elapsed = outputs.timer[2]-outputs.timer[1]
println("elapsed time:", elapsed, " s")

io = open(@__DIR__()*"/time_results.txt", "a")
write(io, "HIT_LF_smag_new nles: $nles, elapsed_time: $elapsed  \n")
close(io)

@info "Solving LF sim (no_model)"
(; u, t), outputs = solve_unsteady(; 
        setup, 
        force! = ou_navierstokes!,
        force_cache = ou_force_cache(setup; ou_bodyforce...),
        # LMWray3 + Float64 is the production configuration (Rik, 2026-09-11).
        method = LMWray3(; T = eltype(ustart)),
        start = (; u = ustart),
        params = rf_params(setup),
        docopy = true,
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 300),
            timer = RikFlow.solver_timer(; n_steps = 100, n_warmup = 110),
        ),
        psolver,
);

elapsed = outputs.timer[2]-outputs.timer[1]
println("elapsed time:", elapsed, " s")

io = open(@__DIR__()*"/time_results.txt", "a")
write(io, "HIT_LF_nomodel nles: $nles, elapsed_time: $elapsed  \n")
close(io)

#####
# track ref
#####

nt = round(Int, tsim / Δt)
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
qoi_ref = stack(data_track.q[:,1:Int(tsim/Δt)+1]);
ref_reader = Reference_reader(qoi_ref);

to_setup_les = RikFlow.TO_Setup(; 
            qois, 
            to_mode = :TRACK_REF, 
            ArrayType, 
            setup, 
            nstep=nt,
            time_series_method = ref_reader);

@info "Solving LF sim (track ref)"
(; u, t), outputs = solve_unsteady(; 
        setup, 
        start = (; u = ustart),
        params = rf_params(setup),
        force! = ou_navierstokes!,
        force_cache = ou_force_cache(setup; ou_bodyforce...),
        # TOMethod wraps an inner scheme; LMWray3 is the production one (Rik, 2026-09-11).
        method = TOMethod(;
            rk_method = LMWray3(; T = eltype(ustart)),
            to_setup = to_setup_les),
        docopy = true,
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 300),
            timer = RikFlow.solver_timer(; n_steps = 100, n_warmup = 110),
        ),
        psolver,
);

elapsed = outputs.timer[2]-outputs.timer[1]
println("elapsed time:", elapsed, " s")

io = open(@__DIR__()*"/time_results.txt", "a")
write(io, "HIT_LF_track nles: $nles, elapsed_time: $elapsed  \n")
close(io)

#####
# TO online
#####
inputs_file_name = "/inputs_example.jld2"
TO_folder = @__DIR__()*"/../exp_square_HIT/output/TO_LRS"
model_index = 1
inputs = load(TO_folder*inputs_file_name, "inputs")
(; name, hist_len, n_replicas, hist_var,tracking_noise) = inputs[model_index]
out_dir = TO_folder*"/$(name)/"

dQ_data = data_track.dQ[:,1:100];

LinReg_file_name = out_dir*"LinReg.jld2"

q_hist = ArrayType{T}(zeros(T,size(qois,1),hist_len)) 
q_hist = cat(q_hist, q_hist, dims=1)

time_series_sampler = RikFlow.LinReg(LinReg_file_name, Xoshiro(seeds.to+1+2), ArrayType, q_hist = q_hist, spinnup_data = ArrayType{T}(dQ_data));

nt = round(Int, tsim / Δt)
to_setup_les = RikFlow.TO_Setup(; 
        qois,
        to_mode = :ONLINE,
        time_series_method = time_series_sampler,
        ArrayType, 
        setup, 
        nstep=nt);

@info "Solving LF sim (TO online)"
(; u, t), outputs = solve_unsteady(; 
        setup, 
        start = (; u = ustart),
        params = rf_params(setup),
        force! = ou_navierstokes!,
        force_cache = ou_force_cache(setup; ou_bodyforce...),
        # TOMethod wraps an inner scheme; LMWray3 is the production one (Rik, 2026-09-11).
        method = TOMethod(;
            rk_method = LMWray3(; T = eltype(ustart)),
            to_setup = to_setup_les),
        docopy = true,
        tlims = (T(0), tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 300),
            timer = RikFlow.solver_timer(; n_steps = 100, n_warmup = 110),
        ),
        psolver,
);

elapsed = outputs.timer[2]-outputs.timer[1]
println("elapsed time:", elapsed, " s")

io = open(@__DIR__()*"/time_results.txt", "a")
write(io, "HIT_LF_TOonline nles: $nles, elapsed_time: $elapsed  \n")
close(io)