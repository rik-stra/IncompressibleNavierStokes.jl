using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics
using RikFlow

fig_folder = @__DIR__()*"/output/figures_poster"

## q trajectories
to_data = [load(
        @__DIR__()*"/paper_runs/output/online/LinReg1/data_online_dns512_les64_Re2000.0_tsim100.0_replica$(i)_rand_initial_dQ.jld2",
        "data_online").q for i in 1:5]
fname = @__DIR__()*"/output/new/data_no_sgs_dns512_les64_Re2000.0_tsim100.0.jld2"
nomodel_data = load(fname, "data_online").q;
smag_data = load(
            @__DIR__()*"/output/new/smag/data_smag_0.071_dns512_les64_Re2000.0_tsim100.0.jld2",
            "data_online").q;
fname = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = stack(load(fname, "data_train").data[1].qoi_hist);
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]

#plot q trajectories
time_axis = 0:0.025:100;

let
    fig = Figure(size=(600, 1500));
    ga = fig[1, 1] = GridLayout()
    axs = [Axis(ga[1, i], yreversed=true) for i in 1:6];
    colgap!(ga, 0)
    rowgap!(ga, 0)   
    to, smag, nomodel, DNS = nothing, nothing, nothing, nothing 
    for q in 1:6
        for i in 1:5
            to = lines!(axs[q], to_data[i][q, 1:10:end], time_axis, color=(:teal, 0.5), linewidth = 1)
        end
        DNS = lines!(axs[q], ref_data[q, 1:10:end], time_axis, color=:black, linewidth = 1, label="DNS");
        smag = lines!(axs[q], smag_data[q, 1:10:end], time_axis, color=:orange, linewidth = 1, label="Smagorinsky");
        nomodel = lines!(axs[q], nomodel_data[q, 1:10:end], time_axis, color=:red, linewidth = 1, label="No model");
        ylims!(axs[q], 100, 0)
        hidedecorations!(axs[q])
        hidespines!(axs[q])
        hlines!(axs[q], [0, 10], color=:black, linewidth=1) 
    end
    Legend(fig[1,2], [DNS, to, smag, nomodel],["DNS", "TO", "Smagorinsky", "No model"])
    display(fig)
    name = fig_folder*"/q_trajectoryHIT.pdf"
    save(name, fig)
    #run(`magick $name -trim $name`)

end

# plot energy spectrum
filename = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
u_start_lf = load(filename, "data_train").data[1].u[1];
n = 64
Δx = 1/n
axis_x = range(0.0, 1., n + 1)
setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
state = (;u = u_start_lf, t=0., temp=0);
scales_LF = get_scale_numbers(u_start_lf, setup)
scales = (;ϵ = 3.7794485)
fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [2,16], v_lines = [6.5,15.5,32], slopeoffset = 1.8, scale_numbers = scales, figure_size = (350,250))
v_labels = ["[0,6]", "[7,15]", "[16,32]"]
v = [3, 11, 24]
for i in 1:3
    text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1*0.5), align = (:center, :center), color = :black)
end
display(fig)
save(fig_folder*"/energy_spectrum_afterspinup.pdf", fig)

# Long term distributions TO, Smag, DNS

smag_data = load(
            @__DIR__()*"/output/new/smag/data_smag_0.071_dns512_les64_Re2000.0_tsim100.0.jld2",
            "data_online").q;
fname = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
q_ref = stack(load(fname, "data_train").data[1].qoi_hist);
linreg_data = [load(
        @__DIR__()*"/paper_runs/output/online/LinReg1/data_online_dns512_les64_Re2000.0_tsim100.0_replica$(i)_rand_initial_dQ.jld2",
        "data_online").q for i in 1:5]
qs = cat(linreg_data..., dims = 2)
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
let 
        g = Figure(size = (800, 300))
        axs = [Axis(g[i % 2, i÷2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
            for i in 0:size(smag_data, 1)-1]
        ref, to, smag = nothing, nothing, nothing
        for i in 1:size(smag_data, 1)
            ref = density!(axs[i], q_ref[i, :], label = "ref", color = (:black, 0.3),
            strokecolor = :black, strokewidth = 3, strokearound = false)
            
            to = density!(axs[i], qs[i, :], label = "TO", color = (:teal, 0.3),
                strokecolor = :teal, strokewidth = 3, linestyle=:dot, strokearound = false)

            smag = density!(axs[i], smag_data[i, :], label = "Smag", color = (:orange, 0.3),
            strokecolor = :orange, strokewidth = 3, linestyle = :dot, strokearound = false)

            hideydecorations!(axs[i], label = false, ticks = false, grid = false)
            #if i == size(smag_data, 1) axislegend(axs[i], position = :rt) end
            #Legend(g[:,3], [ref, to, smag], ["Reference", "TO", "Smagorinsky"])
            if i in [1,2]
                axs[i].ylabel="Density"
            end
        end

        display(g)
        save(fig_folder*"/lt_distr_q_poster.pdf", g)
end