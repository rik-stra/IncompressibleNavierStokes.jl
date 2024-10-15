using JLD2
using CairoMakie

# We first run spinnup.jl to generate an initial condition for the DNS simulation.

# load initial field
filename =  @__DIR__()*"/output/u_start_spinnup_256_Re3000.0_tsim20.0.jld2"
u_start = load(filename, "u_start");

heatmap(u_start[1][1, :, :])

# Using HF_ref.jl, we collect the reference trajectories of the qois
# load reference data
filename = @__DIR__()*"/output/data_train_dns256_les64_Re3000.0_tsim20.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,15],["E", 0, 15],["Z",16,31],["E", 16, 31]]
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