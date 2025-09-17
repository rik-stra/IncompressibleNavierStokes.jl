## Taylor-Green LF sim

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2
using Random

# parse input ARGS
model_index = parse(Int, ARGS[1])
# or set model_index manually
# model_index = 2

inputs_file_name = "/inputs.jld2"
TO_folder = @__DIR__()*"/output/TO_LRS"
track_file = @__DIR__()*"/output/LF/track/track_TG_64_Re_800.0_tsim20.0.jld2"

inputs = load(TO_folder*inputs_file_name, "inputs")
(; name, hist_len, n_replicas, hist_var,tracking_noise) = inputs[model_index]

# Precision
T = Float64
f = one(T)

#### small test
xlims = 0f, 2f*pi
ylims = 0f, 2f*pi
zlims = 0f, 2f*pi

Re = 800f
tsim = 20f
# Grid
nx_les = 64
ny_les = 64
nz_les = 64
Δt = 0.05f

qois = [["Z",0,1],["E", 0, 1],["Z",2,3],["E", 2, 3],["Z",4,5],["E", 4, 5]];

ArrayType = CuArray
kwargs = (;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re,
    backend = CUDABackend(),
    ArrayType,
)


@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"




data_track = load(track_file, "data_train");
dQ_data = data_track.dQ[:,1:5];

u_start_file_name = @__DIR__() *"/output/filtered_initial_field.jld2"
ustart = ArrayType(load(u_start_file_name, "u_start"));

nt = round(Int, tsim / Δt)
out_dir = TO_folder*"/$(name)/"

for i in 1:n_replicas

    setup = Setup(;
        x = (
            range(xlims..., nx_les + 1),
            range(ylims..., ny_les + 1),
            range(zlims..., nz_les + 1)
        ),
        kwargs...,
    );
    psolver = psolver_spectral(setup);
    LinReg_file_name = out_dir*"LinReg.jld2"
    if hist_len == 0
        q_hist = nothing
    else
        q_hist = ArrayType{T}(zeros(T,size(qois,1),hist_len)) 
        if hist_var == :q_star_q
            q_hist = cat(q_hist, q_hist, dims=1)
        end
    end
    time_series_sampler = RikFlow.LinReg(LinReg_file_name, Xoshiro(i), ArrayType, q_hist = q_hist, 
    spinnup_data = ArrayType{T}(dQ_data)
    );
    

    to_setup_les = 
        RikFlow.TO_Setup(; qois, 
        to_mode = :ONLINE, 
        ArrayType, 
        setup,
        nstep=nt,
        time_series_method = time_series_sampler,
        mirror_y = false,);

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
            log = timelogger(; nupdate = 10),
            fields = fieldsaver(; setup, nupdate = 10),  # by calling this BEFORE qoisaver, we also save the field at t=0!
            qoihist = RikFlow.qoisaver(; setup, to_setup=to_setup_les, nupdate = 1, nan_limit = 1e4),
        ),
        psolver,
    );


    q = stack(outputs.qoihist)
    dQ = to_setup_les.outputs.dQ
    tau = to_setup_les.outputs.tau
    fields = outputs.fields
    data = (;dQ, tau, q, fields)

    # Save filtered DNS data
    filename = "$out_dir/TO_online_TG_to_$(nx_les)_tsim$(tsim)_repl_$(i).jld2"
    jldsave(filename; data)
end


# using CairoMakie
# g = Figure();
# ax = [Axis(g[i ÷ 2, i%2], 
#         title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
#         for i in 0:6-1];
# for i in 1:6
#     lines!(ax[i], q[i,:]);
# end
# display(g)