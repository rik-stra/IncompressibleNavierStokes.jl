## Turbulent channel flow

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2
using Random

model_index = parse(Int, ARGS[1])
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
(; name, hist_len, n_replicas, hist_var,tracking_noise) = inputs[model_index]

# Precision
T = Float64
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 100f
Δt = 0.005f

nx_les = 64
ny_les = 64
nz_les = 32
ArrayType = CuArray
kwargs = (;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re = 180f,
    bodyforce = (dim, x, y, z, t) -> 1 * (dim == 1),
    issteadybodyforce = true,
    backend = CUDABackend(),
    ArrayType = ArrayType,
)

setup = Setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);

@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"

psolver = psolver_transform(setup);

qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],
        ["Z",11,17],["E", 11, 17]];

track_file = @__DIR__()*"/output/track/LF_6qoinew_mirror_track_channel_to_64_64_32_dt0.005_tsim10.0.jld2"
data_track = load(track_file, "data_train");
ustart = ArrayType(data_track.fields[1].u);

dQ_data = data_track.dQ[:,1:100];

nt = round(Int, tsim / Δt)
outdir = @__DIR__() *"/output/online_TOpaper/$(name)/"

for i in 1:n_replicas
    
    LinReg_file_name = outdir*"LinReg.jld2"
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
        setup,
        ustart,
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
    filename = "$outdir/LF_online_channel_to_$(nx_les)_$(ny_les)_$(nz_les)_tsim$(tsim)_repl_$(i).jld2"
    jldsave(filename; data)
end
