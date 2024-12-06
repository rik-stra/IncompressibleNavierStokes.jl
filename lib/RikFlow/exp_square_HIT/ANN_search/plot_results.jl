using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics

ANN_names = ["ANN13"]
# figs folder
figsfolder = @__DIR__()*"/figures"
ispath(figsfolder) || mkpath(figsfolder)
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
for ANN_name in ANN_names
n_replicas = 10
q_rep = stack([load(@__DIR__()*"/output/$(ANN_name)/data_online_dns512_les64_Re2000.0_tsim10.0_replica$(i).jld2", "data_online").q for i in 1:n_replicas])
ANN_parameters = load(@__DIR__()*"/output/$(ANN_name)/parameters.jld2", "parameters")
hist_var = ANN_parameters.hist_var
# plot trajectories
best_runs = []
for i in 1:n_replicas
    if q_rep[1,end,i]>500 && q_rep[1,end-50,i]<1650
        push!(best_runs, i)
    end
end
let
lines = []
best= nothing
c1, c2 = 0, 0
g = Figure(size = (1000, 800))
axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
    for i in 0:size(q_ref, 1)-1]
for i in 1:size(q_ref, 1)
    for j in 1:n_replicas
        if j in best_runs
            best=lines!(axs[i],time_index, q_rep[i,:,j], color = (:green, 0.4))
        else
            l=lines!(axs[i],time_index, q_rep[i,:,j], color = (:blue, 0.2))
            if c2 ==0 push!(lines, l); c2=1 end
        end
    end
    l = lines!(axs[i], time_index, q_ref[i,:], color = :black)
    push!(lines, l)
    l = lines!(axs[i], time_index, no_sgs_data[i,:], color = (:red, 0.6))
    push!(lines, l)
end
Label(g[-1, :], text = "ANN tanh \n hist = $(ANN_parameters.hist_len), lamda_reg = $(ANN_parameters.lambda)", fontsize = 20)
Legend(g[0,2], [lines[2], lines[3], lines[1], best], ["Reference", "No model", "Online", "best models"], fontsize = 12)
display(g)
save(figsfolder*"/trajectories_online_$(ANN_name)_hist$(ANN_parameters.hist_len)_$(hist_var)_lamb$(ANN_parameters.lambda).png", g)
end

# only best runs

let
    lines = []
    best = nothing
    g = Figure(size = (1000, 800))
    axs = [Axis(g[i ÷ 2, i%2], 
               title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(q_ref, 1)-1]
    for i in 1:size(q_ref, 1)
        for j in best_runs
            best=lines!(axs[i],time_index, q_rep[i,:,j], color = (:green, 0.3))
        end
        l = lines!(axs[i], time_index, q_ref[i,:], color = :black)
        push!(lines, l)
        l = lines!(axs[i], time_index, no_sgs_data[i,:], color = (:red, 0.6))
        push!(lines, l)
    end
    Label(g[-1, :], text = "ANN tanh \n hist = $(ANN_parameters.hist_len), lamda_reg = $(ANN_parameters.lambda)", fontsize = 20)
    Legend(g[0,2], [lines[1], lines[2], best], ["Reference", "No model", "Online best"], fontsize = 12)
    display(g)
    save(figsfolder*"/best_trajectories_online_$(ANN_name)_hist$(ANN_parameters.hist_len)_$(hist_var)_lamb$(ANN_parameters.lambda).png", g)
end


# plot dQ trajectories
dQ_rep = stack([load(@__DIR__()*"/output/$(ANN_name)/data_online_dns512_les64_Re2000.0_tsim10.0_replica$(i).jld2", "data_online").dQ for i in 1:n_replicas])
filename = @__DIR__()*"/data_track.jld2"
ref_data = load(filename, "data_track");
dQ_ref = stack(ref_data.dQ)
let
    lines = []
    c2 = 0
    best = nothing
    g = Figure(size = (1000, 800))
    axs = [Axis(g[i ÷ 2, i%2], 
               title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(q_ref, 1)-1]
    for i in 1:size(q_ref, 1)
        for j in 1:n_replicas
            if j in best_runs
                best=lines!(axs[i],time_index[1:end-1], dQ_rep[i,:,j], color = (:green, 0.4))
            else
                l=lines!(axs[i],time_index[1:end-1], dQ_rep[i,:,j], color = (:blue, 0.2))
                if c2 ==0 push!(lines, l); c2=1 end
            end
        end
        l = lines!(axs[i], time_index[1:end-1], dQ_ref[i,:], color = :black)
        push!(lines, l)
        
    end
    Label(g[-1, :], text = "ANN tanh \n hist = $(ANN_parameters.hist_len), lamda_reg = $(ANN_parameters.lambda)", fontsize = 20)
    Legend(g[0,2], [lines[2], lines[1], best], ["Reference", "Online", "best models"], fontsize = 12)
    display(g)
    save(figsfolder*"/dQ_trajectories_online_$(ANN_name)_hist$(ANN_parameters.hist_len)_$(hist_var)_lamb$(ANN_parameters.lambda).png", g)
    end

# plot Training loss
n_replicas = 10
let
    losses = [load(@__DIR__()*"/output/$(ANN_name)/ANN_repl$(i).jld2", "losses") for i in 1:n_replicas]
    g = Figure(size=(800, 600))
    ax = Axis(g[1,1], xlabel = "Epoch", ylabel = "Loss")
    l1,l2,lb1,lb2 = nothing, nothing, nothing, nothing
    for i in 1:n_replicas
        if i in best_runs
            lb1 = lines!(g[1,1], 10:3000, losses[i].train[10:end], color = (:green, 0.4))
            lb2 = lines!(g[1,1], 100:100:3000, losses[i].val[2:end], color = (:red, 0.2))
        else
            l1 = lines!(g[1,1], 10:3000, losses[i].train[10:end], color = (:blue, 0.2))
            l2 = lines!(g[1,1], 100:100:3000, losses[i].val[2:end], color = (:red, 0.2), linestyle = :dash)
        end
        
    end
    Legend(g[1,2], [l1, l2, lb1, lb2], ["Training", "Validation", "Training best models", "Validation best models"], fontsize = 12)
    
    Label(g[0, :], text = "ANN tanh \n hist = $(ANN_parameters.hist_len), lamda_reg = $(ANN_parameters.lambda)", fontsize = 20)
    display(g)
    save(figsfolder*"/training_loss_$(ANN_name)_hist$(ANN_parameters.hist_len)_$(hist_var)_lamb$(ANN_parameters.lambda).png", g)
end
end