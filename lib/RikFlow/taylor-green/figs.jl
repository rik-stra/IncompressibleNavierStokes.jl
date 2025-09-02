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
filename = @__DIR__()*"/output/HF/HF_TG_512_to_64_Re_1600.0_tsim20.0.jld2"
ref_data = load(filename, "f");
qois = [["Z",0,1],["E", 0, 1],["Z",2,3],["E", 2, 3],["Z",4,5],["E", 4, 5]]
q_ref = stack(ref_data.data[1].qoi_hist)


let # plot reference QoI trajectories
    time_axis = 0:5e-3:20
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

## load LF data
LF_file_name = @__DIR__()*"/output/LF/LF_TG_64_Re_1600.0_tsim20.0.jld2"
lf_data = load(LF_file_name);
q_lf = stack(lf_data["qoihist"])

smag_file_name = @__DIR__()*"/output/LF/smag/smag_TG_0.07_64_Re_1600.0_tsim20.0.jld2"
smag_data = load(smag_file_name);
q_smag = stack(smag_data["qoihist"])

wale_file_name = @__DIR__()*"/output/LF/wale/wale_TG_0.5_64_Re_1600.0_tsim20.0.jld2"
wale_data = load(wale_file_name);
q_wale = stack(wale_data["qoihist"])

track_file_name = @__DIR__()*"/output/LF/track/track_TG_64_Re_1600.0_tsim20.0.jld2"
track_data = load(track_file_name); 
q_track = track_data["data_train"].q;

TO_file_name = @__DIR__()*"/output/TO_LRS/LinReg1/TO_online_TG_to_64_tsim20.0_repl_4.jld2"
TO_data = load(TO_file_name);
q_TO = TO_data["data"].q;

let # plot lf QoI trajectories
    time_axis_lf = 0:0.05:20
    time_axis_hf = 0:5e-3:20
    g = Figure(size=(1000,1000))
    ax = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(q_lf, 1)-1]
    for i in 1:size(q_lf, 1)
        lines!(ax[i], time_axis_hf[:], q_ref[i,1:length(time_axis_hf)], color=:black, label = "HF")
        lines!(ax[i], time_axis_lf[:], q_lf[i,1:length(time_axis_lf)], color=:blue, label = "LF")
        lines!(ax[i], time_axis_lf[:], q_smag[i,1:length(time_axis_lf)], color=:red, label = "Smag")
        lines!(ax[i], time_axis_lf[:], q_wale[i,1:length(time_axis_lf)], color=:green, label = "WALE")
        lines!(ax[i], time_axis_lf[:], q_track[i,1:length(time_axis_lf)], color=:orange, linestyle = :dash, label = "Track")
        lines!(ax[i], time_axis_lf[:], q_TO[i,1:length(time_axis_lf)], color=:purple, linestyle = :dash, label = "TO")
    end
    axislegend(ax[6], position=:rc)
    ax[5].xlabel="t"
    ax[6].xlabel="t"
    display(g)
    #save(fig_folder*"/Qoi_trajectories_nomodel_smag.pdf", g)
end

# plot trajectories of TO sims
TO_file_name = @__DIR__()*"/output/TO_LRS/LinReg1/TO_online_TG_to_64_tsim20.0_repl_2.jld2"
TO_data = load(TO_file_name);
q_TO = TO_data["data"].q;


# plot fields
size(ref_data.data[1].u)
for i in 1:41
    fig = Figure()
    ax = Axis(fig[1,1], title = "Coarse DNS u at i=$(i)")
    ax2 = Axis(fig[1,2], title = "LES u at i=$(i)")
    ax3 = Axis(fig[2,1], title = "Smag u at i=$(i)")
    ax4 = Axis(fig[2,2], title = "WALE u at i=$(i)")
    ax5 = Axis(fig[2,3], title = "Track u at i=$(i)")
    heatmap!(ax,ref_data.data[1].u[i][:,:,9,3])
    heatmap!(ax2,lf_data["fields"][i].u[:,:,9,3])
    heatmap!(ax3,smag_data["fields"][i].u[:,:,9,3])
    heatmap!(ax4,wale_data["fields"][i].u[:,:,9,3])
    heatmap!(ax5,track_data["fields"][i].u[:,:,9,3])
    display(fig)
end

# plot dQ

track_data
let # plot lf QoI trajectories
    time_axis_lf = 0.05:0.05:20
    g = Figure()
    ax = [Axis(g[i ÷ 2, i%2], 
        title = L"d%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(q_lf, 1)-1]
    for i in 1:size(q_lf, 1)
        lines!(ax[i], time_axis_lf[:], track_data["dQ"][i,1:length(time_axis_lf)], color=:black, label = "dQ")
    end
    axislegend(ax[6], position=:rc)
    ax[5].xlabel="t"
    ax[6].xlabel="t"
    display(g)
    #save(fig_folder*"/Qoi_trajectories_nomodel_smag.pdf", g)
end





hf_fields = load(filename, "fields");

for j in 1:size(hf_fields, 1)
    fig = Figure()
    ax = Axis(fig[1,1], title = "DNS u at t=$(hf_fields[j].t)")
    heatmap!(hf_fields[j].u[:,:,9,1])
    display(fig)
end

# save vtk:

n = 512
Δx = 1/n
axis_x = range(0.0, 2*pi, n + 1)
setup = Setup(;
                x = (axis_x, axis_x, axis_x),
                Re = 1e3,);
state = (;u = hf_fields[2].u, t=hf_fields[2].t, temp=0);
    # save to vtk
save_vtk(state; setup, filename = @__DIR__()*"/output/vtks/HF512", fieldnames = (:velocity, :Qfield))

n = 64
Δx = 1/n
axis_x = range(0.0, 1., n + 1)
setup = Setup(;
                x = (axis_x, axis_x, axis_x),
                Re = Float64(1e3),);
for i in 1:4:80
    state = (;u = ref_data.data[1].u[i], t=0, temp=0);
        # save to vtk
    save_vtk(state; setup, filename = @__DIR__()*"/output/vtks/LF$(i)", fieldnames = (:velocity, :Qfield))
end