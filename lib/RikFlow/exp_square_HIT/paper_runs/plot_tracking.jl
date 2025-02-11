using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics
using RikFlow


# Load data
filename = @__DIR__()*"/../output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
q_ref = stack(ref_data.data[1].qoi_hist)
t_sim = 50
time_index = 0:2.5e-3:t_sim

## plot 1 noise level all replicas
noise_level = 0.01
n_replicas = 2
track_data = []
for i in 1:n_replicas
    fname = @__DIR__()*"/output/tracking/data_track_trackingnoise_std_$(noise_level)_Re2000.0_tsim10.0_replica$(i).jld2"
    push!(track_data, load(fname, "data_track"))
end


#trajectories dQ
let
    
    t_sim = 10
    time_index = 0:2.5e-3:t_sim
    g = Figure()
    range = 1:4000
    axs = [Axis(g[i ÷ 2, i%2], 
        title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(track_data[1].dQ, 1)-1]
    for i in 1:size(track_data[1].dQ, 1)
        for n in 1:n_replicas
            lines!(axs[i], time_index[1:end-1], track_data[n].dQ[i, range])
        end
            #lines!(axs[i], track_data2.dQ[i, range], linestyle = :dash ,color = :red)
    end
    Label(g[-1, :], text = L"$\sigma_\epsilon =$ %$(noise_level)", fontsize = 20)
    display(g)
    
    #save(fig_folder*"/dQ_dns512_les64_Re2000.0_tsim10.png", g)
end

let
    
    t_sim = 10
    time_index = 0:2.5e-3:t_sim
    g = Figure()
    range = 1:4000
    axs = [Axis(g[i ÷ 2, i%2], 
        title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(track_data[1].dQ, 1)-1]
    for i in 1:size(track_data[1].dQ, 1)
        for n in 1:n_replicas
            lines!(axs[i], time_index[1:end-1], track_data[n].q[i, range])
        end
        lines!(axs[i], time_index[1:end-1], q_ref[i,range], linestyle = :dash ,color = :black)
    end
    Label(g[-1, :], text = "tracking noise $(noise_level)", fontsize = 20)
    display(g)
    
    #save(fig_folder*"/dQ_dns512_les64_Re2000.0_tsim10.png", g)
end

