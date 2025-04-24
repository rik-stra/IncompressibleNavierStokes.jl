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


noise_levels = [0.0, 0.001]
track_data = []
for n in noise_levels
    if n == 0.0
        fname = @__DIR__()*"/../output/new/data_track2_dns512_les64_trackingnoise_0.1_Re2000.0_tsim10.0.jld2"
    else 
        fname = @__DIR__()*"/../output/new/data_track2_dns512_les64_trackingnoise_mu_"*string(n)*"_Re2000.0_tsim10.0.jld2"
    end
    push!(track_data, load(fname, "data_track"))
end

# print Statistics of q in reference data and dQ in tracking data

let
    println("Q:");
    println("means: $(mean(q_ref, dims=2))");
    println("std: $(std(q_ref, dims=2))");
    println("means: $(mean(q_ref[:,1:4000], dims=2)./100)");
    println("std: $(std(q_ref[:,1:4000], dims=2)./100)");
    println("dQ:");
    println("means: $(mean(track_data[1].dQ, dims=2))");
    println("std: $(std(track_data[1].dQ, dims=2))");
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
        lines!(axs[i], time_index[1:end-1], track_data[1].dQ[i, range])
            #lines!(axs[i], track_data2.dQ[i, range], linestyle = :dash ,color = :red)
    end
    display(g)
    #save(fig_folder*"/dQ_dns512_les64_Re2000.0_tsim10.png", g)
end

# trajectories Q
let
    t_sim = 0.05
    time_index = 0:2.5e-3:t_sim
    g = Figure()
    
    axs = [Axis(g[i ÷ 2, i%2], 
        title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(track_data[1].q, 1)-1]
    for i in 1:size(track_data[1].q, 1)
        for n in 1:size(noise_levels,1)
            lines!(axs[i], time_index[1:end-1], track_data[n].q[i, 1:size(time_index,1)-1])
            #lines!(axs[i], track_data2.Q[i, range], linestyle = :dash ,color = :red)
        end
    end
    display(g)
    #save(fig_folder*"/Q_dns512_les64_Re2000.0_tsim10.png", g)
end