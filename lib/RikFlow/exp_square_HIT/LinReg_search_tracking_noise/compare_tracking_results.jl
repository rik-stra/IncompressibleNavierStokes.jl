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

input_index = [11]
inputs= load(@__DIR__()*"/inputs.jld2", "inputs")
track_data = []
for r in input_index
    
    (;tracking_noise) = inputs[r]
    fname = @__DIR__()*"/output/LinReg$(r)/data_track_trackingnoise_mu_$(tracking_noise)_Re2000.0_tsim10.0.jld2"
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

let
    for i in 1:size(track_data,1)
    println("dQ $i:");
    println("means: $(mean(track_data[i].dQ, dims=2))");
    println("std: $(std(track_data[i].dQ, dims=2))");
    end
end


#trajectories dQ
let
    for r in 1:size(input_index,1)
    t_sim = 10
    time_index = 0:2.5e-3:t_sim
    g = Figure()
    range = 1:4000
    axs = [Axis(g[i ÷ 2, i%2], 
        title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(track_data[1].dQ, 1)-1]
    for i in 1:size(track_data[1].dQ, 1)
        lines!(axs[i], time_index[1:end-1], track_data[r].dQ[i, range])
            #lines!(axs[i], track_data2.dQ[i, range], linestyle = :dash ,color = :red)
    end
    Label(g[-1, :], text = "tracking noise $(inputs[input_index[r]].tracking_noise)", fontsize = 20)
    display(g)
    end
    #save(fig_folder*"/dQ_dns512_les64_Re2000.0_tsim10.png", g)
end

println(track_data[1].dQ[2, 2])
println(track_data[2].dQ[2, 2])
println(track_data[3].dQ[2, 2])
println(track_data[4].dQ[2, 2])
# trajectories Q
let
    for r in 1:size(input_index,1)
        t_sim = 10
        time_index = 0:2.5e-3:t_sim
        g = Figure()
        range = 1:4000
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(track_data[1].dQ, 1)-1]
        for i in 1:size(track_data[1].dQ, 1)
            lines!(axs[i], time_index[1:end-1], track_data[r].q[i, range])
                #lines!(axs[i], track_data2.dQ[i, range], linestyle = :dash ,color = :red)
        end
        Label(g[-1, :], text = "tracking noise $(inputs[input_index[r]].tracking_noise)", fontsize = 20)
        display(g)
    end
end


# plot linregs
input_index = [9,10]
inputs= load(@__DIR__()*"/inputs.jld2", "inputs")
track_data = []
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
for r in input_index
    
    (;tracking_noise) = inputs[r]
    fname = @__DIR__()*"/output/LinReg$(r)/LinReg.jld2"
    c,stoch_distr =load(fname, "c", "stoch_distr")
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
    title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
    for i in 0:size(qois, 1)-1]
    for i in 1:size(qois,1)
        c_plot = reshape(c[i,7:end-1], 12,5)
        heatmap!(axs[i], c_plot, colormap = :viridis)
    end
    display(g)
end

# plot linregs
input_index = [13]
inputs= load(@__DIR__()*"/inputs.jld2", "inputs")
track_data = []
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
for r in input_index
    
    (;tracking_noise) = inputs[r]
    fname = @__DIR__()*"/output/LinReg$(r)/LinReg.jld2"
    c,stoch_distr =load(fname, "c", "stoch_distr")
    g = Figure()
    ax,hm = heatmap(g[1,1], c, 
    #colormap = :grays, colorrange = (-5, 5), highclip = :red, lowclip = :blue)
    colormap = :balance, colorrange = (-25,25))
    Colorbar(g[1, 2], hm)
    Label(g[0,:], text = "tracking noise $(inputs[r].tracking_noise)", fontsize = 20)
    display(g)
end

for r in input_index
    
    (;tracking_noise) = inputs[r]
    fname = @__DIR__()*"/output/LinReg$(r)/LinReg.jld2"
    c,stoch_distr =load(fname, "c", "stoch_distr")
    g = Figure()
    ax,hm = heatmap(g[1,1], stoch_distr.Σ, 
    #colormap = :grays, colorrange = (-5, 5), highclip = :red, lowclip = :blue)
    colormap = :balance, colorrange = (-1.2,1.2))
    Colorbar(g[1, 2], hm)
    Label(g[0,:], text = "Σ, tracking noise $(inputs[r].tracking_noise)", fontsize = 20)
    display(g)
end