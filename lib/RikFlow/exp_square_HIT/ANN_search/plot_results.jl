using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics

# figs folder
figsfolder = @__DIR__()*"/figures"
# load reference data
filename = @__DIR__()*"/../output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim10.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
q_ref = stack(ref_data.data[1].qoi_hist)
t_sim = 10
time_index = 0:t_sim/(size(q_ref, 2)-1):t_sim

# load no-model sim
filename = @__DIR__()*"/../output/new/data_no_sgs_dns512_les64_Re2000.0_tsim10.0.jld2"
no_sgs_data = stack(load(filename, "data_online").q);

# load replica simulations
n_replicas = 1
q_rep = stack([load(@__DIR__()*"/output/ANN1/data_online_dns512_les64_Re2000.0_tsim10.0_replica$(i).jld2", "data_online").q for i in 1:n_replicas])

# plot trajectories
let
lines = []
g = Figure(size = (1000, 800))
axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
    for i in 0:size(q_ref, 1)-1]
for i in 1:size(q_ref, 1)
    for j in 1:n_replicas
        l=lines!(axs[i],time_index, q_rep[i,:,j], color = (:blue, 0.2))
        if j ==1 push!(lines, l) end
    end
    l = lines!(axs[i], time_index, q_ref[i,:], color = :black)
    push!(lines, l)
    l = lines!(axs[i], time_index, no_sgs_data[i,:], color = (:red, 0.6))
    push!(lines, l)
end
Label(g[-1, :], text = "ANN tanh", fontsize = 20)
Legend(g[0,2], [lines[2], lines[3], lines[1]], ["Reference", "No model", "Online"], fontsize = 12)
display(g)
save(figsfolder*"/trajectories_online_ANN_tanh_5layer_hist0_lamb01.png", g)
end

# plot dQ trajectories
dQ_rep = stack([load(@__DIR__()*"/data_online_dns512_les64_Re2000.0_tsim10.0_replica$(i).jld2", "data_online").dQ for i in 1:n_replicas])
filename = @__DIR__()*"/../data_track_qstar2_dns512_les64_Re2000.0_tsim10.0.jld2"
ref_data = load(filename, "data_track");
dQ_ref = stack(ref_data.dQ)
let
    lines = []
    g = Figure(size = (1000, 800))
    axs = [Axis(g[i ÷ 2, i%2], 
               title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(q_ref, 1)-1]
    for i in 1:size(q_ref, 1)
        for j in 1:n_replicas
            l=lines!(axs[i],time_index[1:end-1], dQ_rep[i,:,j], color = (:blue, 0.2))
            if j ==1 push!(lines, l) end
        end
        l = lines!(axs[i], time_index[1:end-1], dQ_ref[i,:], color = :black)
        push!(lines, l)
        
    end
    Label(g[-1, :], text = "ANN tanh", fontsize = 20)
    Legend(g[0,2], [lines[2], lines[1]], ["Reference", "Online"], fontsize = 12)
    display(g)
    save(figsfolder*"/dQ_trajectories_online_ANN_tanh_5layer_hist0_lamb01.png", g)
    end

# plot Training loss
n_replicas = 5
let
    losses = [load("/export/scratch2/rik/IncompressibleNavierStokes.jl/lib/RikFlow/deep_learning/output/trained_models/ANN_tanh_5layer_regularized0.1_hist0_repl$(i).jld2", "losses") for i in 1:n_replicas-1]
    losses = [losses..., 
    load("/export/scratch2/rik/IncompressibleNavierStokes.jl/lib/RikFlow/deep_learning/output/trained_models/ANN_tanh_5layer_regularized0.1_hist3_repl$(n_replicas).jld2", "losses"),
    load("/export/scratch2/rik/IncompressibleNavierStokes.jl/lib/RikFlow/deep_learning/output/trained_models/ANN_tanh_5layer_regularized0.1_hist10_repl$(n_replicas).jld2", "losses")]
    g = Figure(size=(800, 600))
    ax = Axis(g[1,1], title = "Training loss", xlabel = "Epoch", ylabel = "Loss")
    l1,l2 = nothing, nothing
    for i in 1:n_replicas+1
        l1 = lines!(g[1,1], 10:3000, losses[i].train[10:end], color = (:blue, 0.2))
        l2 = lines!(g[1,1], 100:100:3000, losses[i].val[2:end], color = (:red, 0.2))
    end
    Legend(g[1,2], [l1, l2], ["Training", "Validation"], fontsize = 12)
    display(g)

    save(figsfolder*"/training_loss_online_ANN_tanh_5layer_hist0_lamb01.png", g)
end