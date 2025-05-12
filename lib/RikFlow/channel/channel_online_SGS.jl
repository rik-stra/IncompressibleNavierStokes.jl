## Turbulent channel flow

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
#using CairoMakie
using CUDA
using CUDSS
#using AMGX
using RikFlow
using JLD2
using Random

model_index = 1
inputs = load(@__DIR__()*"/inputs.jld2", "inputs")
(; name, hist_len, n_replicas, hist_var,tracking_noise) = inputs[model_index]

# Precision
T = Float32
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 50f
Δt = 0.01f

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
#amgx_objects = amgx_setup();
#psolver = psolver_cg_AMGX(setup; stuff=amgx_objects);
psolver = default_psolver(setup)

qois = [["Z",0,6],["E", 0, 6],["Z",7,16],["E", 7, 16]];

ustart = ArrayType(load(@__DIR__()*"/output/HF_channel_mirror_256_256_128_to_64_64_32_tsim10.0.jld2")["f"].data[1].u[1]);
track_file = @__DIR__()*"/output/LF_mirror_track_channel_to_64_64_32_tsim10.0.jld2"
data_track = load(track_file, "data_train");
dQ_data = data_track.dQ[:,1:100];

nt = round(Int, tsim / Δt)
outdir = @__DIR__() *"/output/online_mirror/$(name)/"
ispath(outdir) || mkpath(outdir)

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
    time_series_sampler = RikFlow.LinReg(LinReg_file_name, Xoshiro(i), q_hist = q_hist, spinnup_data = ArrayType{T}(dQ_data));
    

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
        docopy = false,
        method = TOMethod(; to_setup = to_setup_les),
        tlims = (0f, tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 10),
            fields = fieldsaver(; setup, nupdate = 100),  # by calling this BEFORE qoisaver, we also save the field at t=0!
            qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1, nan_limit = 1f7),
        ),
        psolver,
    );


    #close_amgx(amgx_objects)
    q = stack(outputs.qoihist)
    dQ = to_setup_les.outputs.dQ
    tau = to_setup_les.outputs.tau
    fields = outputs.fields
    data_train = (;dQ, tau, q, fields)

    # Save filtered DNS data
    filename = "$outdir/LF_online_channel_to_$(nx_les)_$(ny_les)_$(nz_les)_tsim$(tsim)_repl_$(i).jld2"
    jldsave(filename; data_train)
end
exit()
q = stack(outputs.qoihist)
a = load(filename)
keys(a["f"].data[1])
a["f"].data[1].qoi_hist

# u_start low fidelity
u_lf = a["f"].data[1].u[1]
u_hf = load(@__DIR__()*"/output/u_start_256_256_128_tspin10.0.jld2", "u_start")

using CairoMakie
_, _, cb = heatmap(outputs.fields[1].u[:,:,16,1], colorrange = (-20, 20))

heatmap(u_lf[:,:,1,1])
heatmap(u_hf[:,:,64,1], colorrange = (-20, 20))

total_kinetic_energy(ArrayType(u_hf), setup)