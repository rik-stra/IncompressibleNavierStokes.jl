using JLD2
using RikFlow
using IncompressibleNavierStokes
using CUDA
using Random

T = Float64
f = one(T)

nx = 64
ny = 64
nz = 32

xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

Re = 180f
Δt = 0.005f
tsim = 2f


hf_file = @__DIR__()*"/../channel/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2" # for initial condition
qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],["Z",11,17],["E", 11, 17]];

ArrayType = CuArray
backend = CUDABackend()
seeds = (;
    dns = 123, # DNS initial condition
    ou = 333, # OU process
    to = 234, # TO method online sampling
)

# get initial condition
ustart = ArrayType(load(hf_file)["f"].data[1].u[1]);

setup = Setup(;
        boundary_conditions = (
            (PeriodicBC(), PeriodicBC()),
            (DirichletBC(), DirichletBC()),
            (PeriodicBC(), PeriodicBC()),
        ),
        x = (
            range(xlims..., nx + 1),
            range(ylims..., ny + 1),
            range(zlims..., nz + 1)
        ),
        Re,
        bodyforce = (dim, x, y, z, t) -> 1 * (dim == 1),
        issteadybodyforce = true,
        backend,
        ArrayType,
    );

psolver = psolver_transform(setup);


@info "Solving LF sim (SMAG)"
(; u, t), outputs = solve_unsteady(; 
        setup = (;setup..., closure_model = IncompressibleNavierStokes.smagorinsky_closure),
        θ = T(0.071), 
        ustart,
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
write(io, "Channel_LF_smag nles: $nx $ny $nz, elapsed_time: $elapsed  \n")
close(io)

@info "Solving LF sim (WALE)"
(; u, t), outputs = solve_unsteady(; 
        setup = (;setup..., closure_model = IncompressibleNavierStokes.wale_closure),
        θ = T(0.53), 
        ustart,
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
write(io, "Channel_LF_wale nles: $nx $ny $nz, elapsed_time: $elapsed  \n")
close(io)


@info "Solving LF sim (no_model)"
(; u, t), outputs = solve_unsteady(; 
        setup, 
        ustart,
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
write(io, "Channel_LF_nomodel nles: $nx $ny $nz, elapsed_time: $elapsed  \n")
close(io)

#####
# track ref
#####

nt = round(Int, tsim / Δt)
sample_rate = 5 
qoi_ref = stack(load(hf_file)["f"].data[1].qoi_hist[1:nt*sample_rate+1]);
# In the HF simulation we saved every second time step, now we take 10 times bigger time steps
qoi_ref = qoi_ref[:,1:sample_rate:end]
ref_reader = Reference_reader(qoi_ref);

to_setup_les = RikFlow.TO_Setup(; 
            qois, 
            to_mode = :TRACK_REF, 
            ArrayType, 
            setup, 
            nstep=nt,
            time_series_method = ref_reader,
            mirror_y = true,);

@info "Solving LF sim (track ref)"
(; u, t), outputs = solve_unsteady(; 
        setup, 
        ustart,
        method = TOMethod(; to_setup = to_setup_les),
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
write(io, "Channel_LF_track nles: $nx $ny $nz, elapsed_time: $elapsed  \n")
close(io)

#####
# TO online
#####
inputs_file_name = "/inputs_example.jld2"
TO_folder = @__DIR__()*"/../channel/output/TO_LRS"
model_index = 2
inputs = load(TO_folder*inputs_file_name, "inputs")
(; name, hist_len, n_replicas, hist_var,tracking_noise) = inputs[model_index]
out_dir = TO_folder*"/$(name)/"

track_file = @__DIR__()*"/../channel/output/paper_data_channel/track/LF_6qoi_track_channel_to_64_64_32_dt0.005_tsim10.0.jld2"
data_track = load(track_file, "data_train");
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
        nstep=nt,
        mirror_y = true,);

@info "Solving LF sim (TO online)"
(; u, t), outputs = solve_unsteady(; 
        setup, 
        ustart,
        method = TOMethod(; to_setup = to_setup_les),
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
write(io, "Channel_LF_TOonline nles: $nx $ny $nz, elapsed_time: $elapsed  \n")
close(io)