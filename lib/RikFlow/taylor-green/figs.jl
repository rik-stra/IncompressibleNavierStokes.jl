using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics
using RikFlow

fig_folder = @__DIR__()*"/figures"
if !isdir(fig_folder)
    mkdir(fig_folder)
end

## load reference QoI data
filename = @__DIR__()*"/output/HF/HF_TG_128_to_64_Re_2000.0_tsim4.0.jld2"
ref_data = load(filename, "f");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
q_ref = stack(ref_data.data[1].qoi_hist)


let # plot reference QoI trajectories
    time_axis = 0:2.5e-3:4
    g = Figure()
    ax = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(q_ref, 1)-1]
    for i in 1:size(q_ref, 1)
        lines!(ax[i], time_axis[:], q_ref[i,1:length(time_axis)], color=:black, label = "Ref")
    end
    axislegend(ax[6], position=:rc)
    ax[5].xlabel="t"
    ax[6].xlabel="t"
    display(g)
    #save(fig_folder*"/Qoi_trajectories_nomodel_smag.pdf", g)
end

# plot fields
size(ref_data.data[1].u)
for i in 1:40
    fig = Figure()
    ax = Axis(fig[1,1], title = "Coarse DNS u at i=$(i)")
    heatmap!(ref_data.data[1].u[i][:,:,9,2])
    display(fig)
end

hf_fields = load(filename, "fields");

for j in 1:size(hf_fields, 1)
    fig = Figure()
    ax = Axis(fig[1,1], title = "DNS u at t=$(hf_fields[j].t)")
    heatmap!(hf_fields[j].u[:,:,9,3])
    display(fig)
end

# save vtk:

n = 128
Δx = 1/n
axis_x = range(0.0, 1., n + 1)
setup = Setup(;
                x = (axis_x, axis_x, axis_x),
                Re = Float32(1e3),);
state = (;u = hf_fields[3].u, t=hf_fields[3].t, temp=0);
    # save to vtk
save_vtk(state; setup, filename = @__DIR__()*"/output/vtks/HF3", fieldnames = (:velocity, :Qfield))
