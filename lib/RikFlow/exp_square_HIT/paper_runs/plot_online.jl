using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics
using DataFrames

ANN_names = ["LinReg$i" for i in 1:10]
# figs folder
figsfolder = @__DIR__()*"/../output/figures_paper/online"
if !isdir(figsfolder)
    mkpath(figsfolder)
end

# load reference data
filename = @__DIR__()*"/../output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
q_ref = stack(ref_data.data[1].qoi_hist)

track_file = "/../output/new/data_track2_dns512_les64_Re2000.0_tsim100.0.jld2"
dQ_ref = stack(load(@__DIR__()*track_file, "data_track").dQ)
# = stack(ref_data.dQ[1])
t_sim = 30
time_index = 0:0.0025:t_sim

# load no-model sim
filename = @__DIR__()*"/../output/new/data_no_sgs2_dns512_les64_Re2000.0_tsim100.0.jld2"
no_sgs_data = stack(load(filename, "data_online").q);

# load replica simulations
for ANN_name in ANN_names
    #ANN_name = "LinReg19"
    n_replicas = 5
    q_rep = [load(@__DIR__()*"/output/online/$(ANN_name)/data_online_dns512_les64_Re2000.0_tsim100.0_replica$(i)_rand_initial_dQ.jld2", "data_online").q for i in 1:n_replicas]
    linreg_params_table = DataFrame(load(@__DIR__()*"/inputs.jld2", "inputs"))
    linreg_params = linreg_params_table[linreg_params_table.name .== ANN_name,:]
    if linreg_params.model_noise[1] == :no_noise
        linreg_params.model_noise_str .= "no noise"
    else
        linreg_params.model_noise_str = linreg_params.model_noise
    end
    # plot trajectories

    let
    lines = []
    best, ref, no_model, model= nothing, nothing, nothing, nothing

    g = Figure(size = (2000, 1600))
    axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(q_ref, 1)-1]
    
    for i in 1:size(q_ref, 1)
        for j in 1:n_replicas
                xlim_right = min(size(q_rep[j],2), size(time_index,1))
                model=lines!(axs[i],time_index[1:xlim_right], q_rep[j][i,1:xlim_right], color = (:blue, 0.2))
            
        end
        xlim_right = min(maximum(size.(q_rep,2)), size(time_index,1))
        ref = lines!(axs[i], time_index[1:xlim_right], q_ref[i,1:xlim_right], color = :black)
        no_model = lines!(axs[i], time_index[1:min(xlim_right,size(no_sgs_data,2))], no_sgs_data[i,1:min(xlim_right,size(no_sgs_data,2))], color = (:red, 0.6))
        ylims!(axs[i],(0, maximum(q_ref[i,:])*2)) 
    end
    Label(g[-1, :], text = L"$\sigma_\epsilon =$ %$(linreg_params.tracking_noise[1]), $\eta =$ %$(linreg_params.model_noise_str[1]), hist $=$ %$(linreg_params.hist_len[1]), $\lambda =$ %$(linreg_params.lambda[1])", fontsize = 20)
    Legend(g[0,2], [ref, no_model, model], ["Reference", "No model", "Online"], fontsize = 12)
    #display(g)
    save(figsfolder*"/trajectories_online_$(ANN_name)_noise$(linreg_params.tracking_noise[1])_$(linreg_params.model_noise[1])_hist$(linreg_params.hist_len[1])_lambd$(linreg_params.lambda[1]).png", g)
    end



    # plot dQ trajectories
    dQ_rep = [load(@__DIR__()*"/output/online/$(ANN_name)/data_online_dns512_les64_Re2000.0_tsim100.0_replica$(i)_rand_initial_dQ.jld2", "data_online").dQ for i in 1:n_replicas]
    
    let
        # lims = [(nothing, (-25,25)), 
        #     (nothing, (-0.06, 0.06)), 
        #     (nothing, (-50, 50)), 
        #     (nothing, (-0.005, 0.005)), 
        #     (nothing, (-300, 0)), 
        #     (nothing, (-0.007, 0))]
        lines = []
        c2 = 0
        ref, model = nothing, nothing
        g = Figure(size = (1000, 800))
        axs = [Axis(g[i ÷ 2, i%2], 
                #limits = lims[i+1],
                title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(q_ref, 1)-1]
        
        for i in 1:size(q_ref, 1)
            for j in 1:n_replicas
                    xlim_right = min(size(dQ_rep[j],2), size(time_index,1))
                    model=lines!(axs[i],time_index[1:xlim_right], dQ_rep[j][i,1:xlim_right], color = (:blue, 0.2))
                
            end
            xlim_right = min(maximum(size.(dQ_rep,2)), size(time_index,1)-1)
            ref = lines!(axs[i], time_index[1:xlim_right], dQ_ref[i,1:xlim_right], color = :black)
            
        end
        Label(g[-1, :], text = L"$\sigma_\epsilon =$ %$(linreg_params.tracking_noise[1]), $\eta =$ \text{%$(linreg_params.model_noise_str[1])}, hist $=$ %$(linreg_params.hist_len[1]), $\lambda =$ %$(linreg_params.lambda[1])", fontsize = 20)
        Legend(g[0,2], [ref, model], ["ref","Online"], fontsize = 12)
        #display(g)
        save(figsfolder*"/dQ_trajectories_online_$(ANN_name)_noise$(linreg_params.tracking_noise[1])_$(linreg_params.model_noise[1])_hist$(linreg_params.hist_len[1])_lambd$(linreg_params.lambda[1]).png", g)
    end

end

for ANN_name in ANN_names
let # all replicas together
    
    linreg_params_table = DataFrame(load(@__DIR__()*"/inputs.jld2", "inputs"))
    linreg_params = linreg_params_table[linreg_params_table.name .== ANN_name,:]
    if linreg_params.model_noise[1] == :no_noise
        linreg_params.model_noise_str .= "no noise"
    else
        linreg_params.model_noise_str = linreg_params.model_noise
    end
    n_replicas = 5
    q_rep = [load(@__DIR__()*"/output/online/$(ANN_name)/data_online_dns512_les64_Re2000.0_tsim100.0_replica$(i)_rand_initial_dQ.jld2", "data_online").q for i in 1:n_replicas]
    q_rep = q_rep[size.(q_rep,2) .>= 40000]
    qs = cat(q_rep..., dims = 2)
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
        title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(qois, 1)-1]
    for i in 1:size(qois, 1)
        density!(axs[i], q_ref[i, 100:end], label = "ref", color = (:black, 0.3),
        strokecolor = :black, strokewidth = 3, strokearound = false)
        
        density!(axs[i], qs[i,100:end], label = "TO", color = (:red, 0.3),
            strokecolor = :red, strokewidth = 3, strokearound = false)
        
        if i == size(qois, 1) axislegend(axs[i], position = :rt) end
    end
    Label(g[-1, :], text = L"$\sigma_\epsilon =$ %$(linreg_params.tracking_noise[1]), $\eta =$ \text{%$(linreg_params.model_noise_str[1])}, hist $=$ %$(linreg_params.hist_len[1]), $\lambda =$ %$(linreg_params.lambda[1])", fontsize = 20)
    #display(g)
    save(figsfolder*"/lt_distr_q_noise$(linreg_params.tracking_noise[1])_$(linreg_params.model_noise[1])_hist$(linreg_params.hist_len[1])_lambd$(linreg_params.lambda[1]).png", g)
    
end
end


# # plot c

#     ANN_name = "LinReg1"
#     ANN_parameters = load(@__DIR__()*"/output/online/$(ANN_name)/parameters.jld2", "parameters")
#     c = load(@__DIR__()*"/output/online/$(ANN_name)/LinReg.jld2", "c")
#     heatmap( c, colormap = :balance, colorrange = (-1,1))

#     #save(figsfolder*"/c_heatmap_$(ANN_name)_hist$(ANN_parameters.hist_len)_$(ANN_parameters.hist_var).png", g)

#     stoch_distr = load(@__DIR__()*"/output/online/$(ANN_name)/LinReg.jld2", "stoch_distr")