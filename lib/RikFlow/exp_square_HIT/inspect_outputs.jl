using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics

# create folder for figures
fig_folder = @__DIR__()*"/output/figures"
if !isdir(fig_folder)
    mkdir(fig_folder)
end

# We first run spinnup.jl to generate an initial condition for the DNS simulation.

# load initial field
filename =  @__DIR__()*"/output/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2"
u_start = load(filename, "u_start");

heatmap(u_start[1][end-1, :, :])
# plot energy spectrum
let
    n = 512
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
        x = (axis_x, axis_x, axis_x),
        Re = Float32(2e3),);
    state = (;u = u_start, t=0.);
    fig = energy_spectrum_plot(state; setup, npoint = 100)
    save(fig_folder*"/energy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.png", fig)
end
# plot enstrophy spectrum
let
    n = 512
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
        x = (axis_x, axis_x, axis_x),
        Re = Float32(2e3),);
    state = (;u = u_start, t=0.);
    fig = enstrophy_spectrum_plot(state; setup, npoint = 100)
    save(fig_folder*"/enstrophy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.png", fig)
end

######################################################################
# Using HF_ref.jl, we collect the reference trajectories of the qois
# load reference data
######################################################################
filename = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim10.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
keys(ref_data.data[1])
q_ref = stack(ref_data.data[1].qoi_hist)
t_sim = 10
time_index = 0:t_sim/(size(q_ref, 2)-1):t_sim

let # plot reference data
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(q_ref, 1)-1]
    for i in 1:size(q_ref, 1)
        lines!(axs[i], time_index, q_ref[i,:])
    end
    display(g)
    save(fig_folder*"/q_ref_dns512_les64_Re2000.0_freeze_10_tsim10.png", g)
end

u_lf = ref_data.data[1].u;
heatmap(u_lf[1][1][end-1, :, :]) # initial coarse field
## plot specrum of filtered field
let
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
        x = (axis_x, axis_x, axis_x),
        Re = Float32(2e3),);
    state = (;u = u_lf[1], t=0.);
    energy_spectrum_plot(state; setup, npoint = 100)
end

### Track ref ###################################################################
# We now run track_ref.jl to track the reference trajectories of the qois
#################################################################################
fname = @__DIR__()*"/output/new/data_track_dns512_les64_Re2000.0_tsim10.0.jld2"
track_data = load(fname, "data_track");
# plot dQ data
#trajectories
let 
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(track_data.dQ, 1)-1]
    for i in 1:size(track_data.dQ, 1)
        lines!(axs[i], track_data.dQ[i, :])
    end
    display(g)
    save(fig_folder*"/dQ_dns512_les64_Re2000.0_tsim10.png", g)
end

let #trajectories with -q_ref
    g = Figure()
    t1 = i -> rich("d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])] ")
    t2 = i -> rich("-$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]", color = :red)
    axs1 = [Axis(g[i ÷ 2, i%2], 
           title = rich(t1(i), t2(i)))
        for i in 0:size(track_data.dQ, 1)-1]
    axs2 = [Axis(g[i ÷ 2, i%2], yticklabelcolor = :red, yaxisposition = :right)
        for i in 0:size(track_data.dQ, 1)-1]
    for i in 1:size(track_data.dQ, 1)
        lines!(axs1[i], track_data.dQ[i, :])
        lines!(axs2[i], -q_ref[i,:], color = :red)
    end
    display(g)
end

# plot histograms
let
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(track_data.dQ, 1)-1]
    for i in 1:size(track_data.dQ, 1)
        density!(axs[i], track_data.dQ[i, :])
    end
    display(g)
end

# plot fancy histograms
let
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]",
           xticks = ([0.6, 1.2, 1.8, 2.4], ["1/4", "1/2", "3/4", "1"]),
           )
        for i in 0:size(track_data.dQ, 1)-1]
    dQ_scaled = track_data.dQ ./ (std(track_data.dQ, dims = 2))
    for i in 1:size(track_data.dQ, 1)
        for t in 1:4
            density!(axs[i], dQ_scaled[i, 1:t*1000], offset = t*(0.6), color = (:slategray, 0.5), direction=:y)
        end
    end
    display(g)
    save(fig_folder*"/dQ_hist_overtime_dns512_les64_Re2000.0_tsim10.png", g)
end

## plot corrected trajectories
let 
    interval = 550:660
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(track_data.q, 1)-1]
    for i in 1:size(track_data.q, 1)
        lines!(axs[i], q_ref[i, interval], label = "ref", color = :black)
        lines!(axs[i], track_data.q_star[i, interval], label = "*")
        lines!(axs[i], track_data.q[i, interval], label = "corrected")
    end
    axislegend(position = :rt)
    display(g)
    
end

### no SGS ###
fname = @__DIR__()*"/output/new/data_no_sgs_dns512_les64_Re2000.0_tsim10.0.jld2"
no_sgs_data = load(fname, "data_online");

let 
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(no_sgs_data.q, 1)-1]
    for i in 1:size(no_sgs_data.q, 1)
        lines!(axs[i], q_ref[i, :], label = "ref")
        lines!(axs[i], no_sgs_data.q[i, :], label = "no sgs")
        if i == size(no_sgs_data.q, 1) axislegend(axs[i], position = :rt) end
    end
    display(g)
    save(fig_folder*"/q_no_sgs_dns512_les64_Re2000.0_tsim10.png", g)
end

## plot final field
heatmap(no_sgs_data.fields[end].u[1][1,:,:])

### Online SGS ###
fname = @__DIR__()*"/output/new/data_online_samplingMVG_sampler_dns512_les64_Re2000.0_tsim10.0.jld2"
online_data = load(fname, "data_online")
# plot dQ data
let 
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(online_data.dQ, 1)-1]
    for i in 1:size(online_data.dQ, 1)
        plot!(axs[i], online_data.dQ[i, :])
    end
    display(g)
end
# plot q data
let 
    g = Figure(size = (1200, 800))
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(online_data.dQ, 1)-1]
    for i in 1:size(online_data.q, 1)
        
        lines!(axs[i], no_sgs_data.q[i, :], label = "no sgs")
        lines!(axs[i], online_data.q[i, :], label = "online")
        lines!(axs[i], q_ref[i, :], label = "ref", color = :black)
        if i==1 axislegend(axs[i], position = :rt) end
    end
    display(g)
end
heatmap(online_data.fields[end].u[1][1,:,:])

### Online SGS Resample ###
fname = @__DIR__()*"/output/new/data_online_samplingRikFlow.Resampler_dns512_les64_Re2000.0_tsim10.0.jld2"
online_data = load(fname, "data_online")
# plot dQ data
let 
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(online_data.dQ, 1)-1]
    for i in 1:size(online_data.dQ, 1)
        plot!(axs[i], online_data.dQ[i, :])
    end
    display(g)
end
# plot q data
let 
    g = Figure(size = (1200, 800))
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(online_data.dQ, 1)-1]
    for i in 1:size(online_data.q, 1)
        
        lines!(axs[i], no_sgs_data.q[i, :], label = "no sgs")
        lines!(axs[i], online_data.q[i, :], label = "online")
        lines!(axs[i], q_ref[i, :], label = "ref", color = :black)
        if i==1 axislegend(axs[i], position = :rt) end
    end
    display(g)
end
heatmap(online_data.fields[end].u[1][1,:,:])

### Online SGS ANN tanh ###
fname = @__DIR__()*"/output/new/data_online_samplingANN_tanh_dns512_les64_Re2000.0_tsim10.0.jld2"
online_data = load(fname, "data_online")
# plot dQ data
let 
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(online_data.dQ, 1)-1]
    for i in 1:size(online_data.dQ, 1)
        lines!(axs[i], track_data.dQ[i, 1:1000], color = :black)
        lines!(axs[i], online_data.dQ[i, 1:1000])
        
    end
    display(g)
end
# plot q data
let 
    g = Figure(size = (1200, 800))
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(online_data.dQ, 1)-1]
    for i in 1:size(online_data.q, 1)
        
        lines!(axs[i], no_sgs_data.q[i, :], label = "no sgs")
        lines!(axs[i], online_data.q[i, :], label = "online")
        lines!(axs[i], q_ref[i, :], label = "ref", color = :black)
        if i==1 axislegend(axs[i], position = :rt) end
    end
    display(g)
end
heatmap(online_data.fields[end].u[1][1,:,:])

### Online SGS ANN tanh regularized###
fname = @__DIR__()*"/output/new/data_online_samplingANN_tanh_regularized_dns512_les64_Re2000.0_tsim10.0.jld2"
online_data = load(fname, "data_online")
# plot dQ data
let 
    g = Figure(size = (1200, 800))
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(online_data.dQ, 1)-1]
    for i in 1:size(online_data.dQ, 1)
        lines!(axs[i], track_data.dQ[i, :], color = :black)
        lines!(axs[i], online_data.dQ[i, :])
        
    end
    display(g)
end
# plot q data
let 
    g = Figure(size = (1200, 800))
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(online_data.dQ, 1)-1]
    for i in 1:size(online_data.q, 1)
        
        lines!(axs[i], no_sgs_data.q[i, 1:4000], label = "no sgs")
        lines!(axs[i], online_data.q[i, 1:4000], label = "online")
        lines!(axs[i], q_ref[i, 1:4000], label = "ref", color = :black)
        if i==1 axislegend(axs[i], position = :rt) end
    end
    display(g)
end
heatmap(online_data.fields[end].u[1][1,:,:])