if false                                               #src
    include("../src/RikFlow.jl")                  #src
    include("../../../src/IncompressibleNavierStokes.jl") #src
end

using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2
using Random

# parse input ARGS
model_index = parse(Int, ARGS[1])
# or set model_index manually
# model_index = 1

inputs_file_name = "/inputs_example.jld2"
TO_folder = @__DIR__()*"/output/TO_LRS"
track_file = @__DIR__()*"/output/paper_data_channel/track/LF_6qoi_track_channel_to_64_64_32_dt0.005_tsim10.0.jld2"

# simulation parameters

qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],
        ["Z",11,17],["E", 11, 17]];
# Precision
T = Float64
f = one(T)
ArrayType = CuArray
# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 2f
Δt = 0.005f

nx_les = 64
ny_les = 64
nz_les = 32

inputs = load(TO_folder*inputs_file_name, "inputs")
(; name, hist_len, n_replicas, hist_var,tracking_noise) = inputs[model_index]

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
    ArrayType = ArrayType,
)

setup = rf_setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1),
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);

out_dir = TO_folder*"/$(name)/"
data_track = load(track_file, "data_train");
ustart = ArrayType(data_track.fields[1].u);
dQ_data = data_track.dQ[:,1:100]; # first 100 time steps are not predicted but taken from training data.

@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"
psolver = psolver_transform(setup);

nt = round(Int, tsim / Δt)

for i in 1:n_replicas
    LinReg_file_name = out_dir*"LinReg.jld2"
    if hist_len == 0
        q_hist = nothing
    else
        q_hist = ArrayType{T}(zeros(T,size(qois,1),hist_len)) 
        if hist_var == :q_star_q
            q_hist = cat(q_hist, q_hist, dims=1)
        end
    end
    time_series_sampler = RikFlow.LinReg(LinReg_file_name, Xoshiro(i), ArrayType, q_hist = q_hist, spinnup_data = ArrayType{T}(dQ_data));
    

    to_setup_les = 
        RikFlow.TO_Setup(; qois, 
        to_mode = :ONLINE, 
        ArrayType, 
        setup,
        nstep=nt,
        time_series_method = time_series_sampler,
        mirror_y = true,);

    @info "Solving LES"
    # Solve DNS and store filtered quantities
    (; u, t), outputs = solve_unsteady(;
        # Steady driving force, formerly Setup(; bodyforce, issteadybodyforce).
        # Without it the channel is unforced and decays to rest, silently.
        force! = rf_bodyforce_navierstokes!,
        force_cache = rf_steady_force_cache(to_setup_les, channel_bodyforce),
        setup,
        start = (; u = ustart),
        params = rf_params(to_setup_les),
        docopy = true,
        method = TOMethod(; to_setup = to_setup_les),
        tlims = (0f, tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 200),
            fields = fieldsaver(; setup, nupdate = 200),  # by calling this BEFORE qoisaver, we also save the field at t=0!
            qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1, nan_limit = 1e8),
        ),
        psolver,
    );


    q = stack(outputs.qoihist)
    dQ = to_setup_les.outputs.dQ
    tau = to_setup_les.outputs.tau
    fields = outputs.fields
    data = (;dQ, tau, q, fields)

    # Save filtered DNS data
    filename = "$out_dir/LF_online_channel_to_$(nx_les)_$(ny_les)_$(nz_les)_tsim$(tsim)_repl_$(i).jld2"
    jldsave(filename; data)
end
