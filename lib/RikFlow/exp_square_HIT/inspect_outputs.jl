using JLD2
using CairoMakie
using IncompressibleNavierStokes

# We first run spinnup.jl to generate an initial condition for the DNS simulation.

# load initial field
filename =  @__DIR__()*"/output/u_start_spinnup_512_Re2000.0_tsim4.0.jld2"
u_start = load(filename, "u_start");

heatmap(u_start[1][:, :, 1])
# plot spectrum
n = 512
axis_x = range(0.0, 1., n + 1)
setup = Setup(;
    x = (axis_x, axis_x, axis_x),
    Re = Float32(2e3),);
state = (;u = u_start, t=0.);
energy_spectrum_plot(state; setup, npoint = 100)

# Using HF_ref.jl, we collect the reference trajectories of the qois
# load reference data
filename = @__DIR__()*"/output/data_train_dns256_les64_Re3000.0_tsim20.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
keys(ref_data.data[1])
q_ref = stack(ref_data.data[1].qoi_hist)

let # plot reference data
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(q_ref, 1)-1]
    for i in 1:size(q_ref, 1)
        plot!(axs[i], q_ref[i, :])
    end
    display(g)
end

u_lf = ref_data.data[1].u
heatmap(u_lf[1][1][1, :, :]) # initial coarse field

### Track ref ####
# We now run track_ref.jl to track the reference trajectories of the qois
fname = @__DIR__()*"/output/data_track_dns256_les64_Re3000.0_tsim20.0.jld2"
track_data = load(fname, "data_track")
# plot dQ data
let 
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(track_data.dQ, 1)-1]
    for i in 1:size(track_data.dQ, 1)
        plot!(axs[i], track_data.dQ[i, :])
    end
    display(g)
end

### no SGS ###
fname = @__DIR__()*"/output/data_no_sgs_dns256_les64_Re3000.0_tsim40.0.jld2"
no_sgs_data = load(fname, "data_online")

let 
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
           title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
        for i in 0:size(no_sgs_data.q, 1)-1]
    for i in 1:size(no_sgs_data.q, 1)
        plot!(axs[i], q_ref[i, :], label = "ref")
        plot!(axs[i], no_sgs_data.q[i, :], label = "no sgs")
        axislegend(axs[i], position = :rt)
    end
    display(g)
end

## plot final field
heatmap(no_sgs_data.fields[end].u[1][1,:,:])

### Online SGS ###
fname = @__DIR__()*"/output/data_online_samplingmvg_dns256_les64_Re3000.0_tsim100.0.jld2"
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
    g = Figure()
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