using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics

ANN_names = ["LinReg5"]
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
    n_replicas = 2
    #ANN_name = ANN_names[5]
    q_rep = [load(@__DIR__()*"/output/$(ANN_name)/data_online_dns512_les64_Re2000.0_tsim10.0_replica$(i).jld2", "data_online").q for i in 1:n_replicas]
    ANN_parameters = load(@__DIR__()*"/output/$(ANN_name)/parameters.jld2", "parameters")
    hist_var = ANN_parameters.hist_var
    # plot trajectories

    let
    lines = []
    best, ref, no_model, model= nothing, nothing, nothing, nothing

    g = Figure(size = (1000, 800))
    axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(q_ref, 1)-1]
    xlim_right = maximum(size.(q_rep,2))
    for i in 1:size(q_ref, 1)
        for j in 1:n_replicas
            
                model=lines!(axs[i],time_index[1:size(q_rep[j],2)], q_rep[j][i,:], color = (:blue, 0.2))
            
        end
        ref = lines!(axs[i], time_index[1:xlim_right], q_ref[i,1:xlim_right], color = :black)
        no_model = lines!(axs[i], time_index[1:xlim_right], no_sgs_data[i,1:xlim_right], color = (:red, 0.6))
        ylims!(axs[i],(0, maximum(q_ref[i,:])*2)) 
    end
    Label(g[-1, :], text = "LinReg \n hist = $(ANN_parameters.hist_len), $(hist_var)", fontsize = 20)
    Legend(g[0,2], [ref, no_model, model], ["Reference", "No model", "Online"], fontsize = 12)
    display(g)
    save(figsfolder*"/trajectories_online_$(ANN_name)_hist$(ANN_parameters.hist_len)_$(hist_var).png", g)
    end



    # plot dQ trajectories
    dQ_rep = stack([load(@__DIR__()*"/output/$(ANN_name)/data_online_dns512_les64_Re2000.0_tsim10.0_replica$(i).jld2", "data_online").dQ for i in 1:n_replicas])
    filename = @__DIR__()*"/data_track.jld2"
    ref_data = load(filename, "data_track");
    dQ_ref = stack(ref_data.dQ)
    let
        lines = []
        c2 = 0
        ref, model = nothing, nothing
        g = Figure(size = (1000, 800))
        axs = [Axis(g[i ÷ 2, i%2], 
                title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(q_ref, 1)-1]
        for i in 1:size(q_ref, 1)
            for j in 1:n_replicas
                    model=lines!(axs[i],time_index[1:end-1], dQ_rep[i,:,j], color = (:blue, 0.2))
                
            end
            ref = lines!(axs[i], time_index[1:end-1], dQ_ref[i,:], color = :black)
            
        end
        Label(g[-1, :], text = "LinReg \n hist = $(ANN_parameters.hist_len), $(hist_var)", fontsize = 20)
        Legend(g[0,2], [ref, model], ["Reference", "Online"], fontsize = 12)
        display(g)
        save(figsfolder*"/dQ_trajectories_online_$(ANN_name)_hist$(ANN_parameters.hist_len)_$(hist_var).png", g)
    end

end