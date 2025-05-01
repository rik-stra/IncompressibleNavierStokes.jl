using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics
using RikFlow

fig_folder = @__DIR__()*"/output/figures_paper"
if !isdir(fig_folder)
    mkdir(fig_folder)
end

filename =  @__DIR__()*"/output/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2"
u_start = stack(load(filename, "u_start"));

n = 512
Δx = 1/n
axis_x = range(0.0, 1., n + 1)
setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
state = (;u = u_start, t=0., temp=0);
        
# save to vtk
#save_vtk(state; setup, filename = @__DIR__()*"/output/vtks", fieldnames = (:velocity, :Qfield))
scales = get_scale_numbers(u_start, setup)
fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [2,16], v_lines = [scales.λ, scales.η, Δx], slopeoffset = 1.8, scale_numbers = scales, plot_wavelength = true)
display(fig)
v = [scales.λ, scales.η, 1/n]
v_labels = ["λ", "η", "Δx"]
for i in 1:3
    text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1e-12*1.2), align = (:left, :bottom), color = :black)
end
display(fig)
save(fig_folder*"/energy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.png", fig)

## energy spectrum coarse grained
filename = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = load(filename, "data_train");

u_start_lf = ref_data.data[1].u[1];
heatmap(u_start_lf[end-1, :, :, 1]) # initial coarse field
heatmap(u_start[end-1, :, :, 1])
n = 64
Δx = 1/n
axis_x = range(0.0, 1., n + 1)
setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
state = (;u = u_start_lf, t=0., temp=0);
scales_LF = get_scale_numbers(u_start_lf, setup)
scales = (;ϵ = 3.7794485)
fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [2,16], v_lines = [6.5,15.5,32], slopeoffset = 1.8, scale_numbers = scales)
v_labels = ["[0,6]", "[7,15]", "[16,32]"]
v = [3, 11, 24]
for i in 1:3
    text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1*0.5), align = (:center, :center), color = :black)
end
display(fig)
save(fig_folder*"/energy_spectrum_afterspinup_coarse_grained_Re2000.0_freeze_10_tsim4.png", fig)

######################################################################
# Using HF_ref.jl, we collect the reference trajectories of the qois
# load reference data
######################################################################
filename = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
q_ref = stack(ref_data.data[1].qoi_hist)
t_sim = 20
time_index = 0:2.5e-3:t_sim

# begin some plots
    let # plot reference data
        g = Figure()
        axs = [Axis(g[i ÷ 2, i%2], 
            title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}",
            )
            for i in 0:size(q_ref, 1)-1]
        for i in 1:size(q_ref, 1)
            lines!(axs[i], time_index, q_ref[i,1:size(time_index,1)])
            xlims!(axs[i], -0.4, t_sim) 
        end
        for i in [1, 2, 3, 4]
                hidexdecorations!(axs[i], ticks = false, grid = false)
        end
        for i in [5,6]
                axs[i].xlabel = "t"
        end
        display(g)
        save(fig_folder*"/Qoi_trajectories_HF.png", g)
    end
# end

## Compare spectra of final fields
#begin
    ## LinReg1 at t=100
    n_fields = 10
    fname = @__DIR__()*"/paper_runs/output/online/LinReg1/data_online_dns512_les64_Re2000.0_tsim100.0_replica1_rand_initial_dQ.jld2"
    u_LinReg = load(fname)["data_online"].fields[end:-1:end-n_fields+1]; #41

    fname = @__DIR__()*"/output/new/data_no_sgs_dns512_les64_Re2000.0_tsim100.0.jld2"
    u_no_sgs = load(fname, "data_online").fields[end:-10:end-10*n_fields+1]; #401

    u_smag = load(@__DIR__()*"/output/new/smag/data_smag_0.071_dns512_les64_Re2000.0_tsim100.0.jld2", "data_online").fields[end:-1:end-n_fields+1]; #41

    fname = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
    u_ref = load(fname, "data_train").data[1].u[end:-10:end-10*n_fields+1]; #401

    n = 64
    Δx = 1/n
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
                x = (axis_x, axis_x, axis_x),
                Re = Float32(2e3),);
    states = [ u_ref,
            u_no_sgs,
            u_smag,
            u_LinReg ];
    #scales = get_scale_numbers(u_ref[1], setup)
    scales = (;ϵ = 3.7794485) # taken from HF_ref
    fig = energy_spectra_comparison(
            states,
            ["Ref", "No model", "Smagorinsky", "TO LRS h=5"];
            setup,
            sloperange = [2, 16],
            slopeoffset = 3,
            scale_numbers = scales,
        )
    save(fig_folder*"/energy_spectrum_compare_online_nsnaps_$(n_fields).png", fig)

#end


## plot short term correlation
#load data
let
fig_folder = @__DIR__()*"/output/figures_paper"
to_data = [load(@__DIR__()*"/paper_runs/output/online/LinReg1/data_online_dns512_les64_Re2000.0_tsim100.0_replica$(i)_rand_initial_dQ.jld2")["data_online"].q for i in 1:5]
fname = @__DIR__()*"/output/new/data_no_sgs_dns512_les64_Re2000.0_tsim100.0.jld2"
nomodel_data = load(fname, "data_online").q;
smag_data = load(
            @__DIR__()*"/output/new/smag/data_smag_0.071_dns512_les64_Re2000.0_tsim100.0.jld2",
            "data_online").q;
fname = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = stack(load(fname, "data_train").data[1].qoi_hist);
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]

time_axis = 0:2.5e-3:100

g = Figure()
ax = [Axis(g[i ÷ 2, i%2], 
    title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
    for i in 0:size(ref_data, 1)-1]
for i in 1:size(ref_data, 1)
    lines!(ax[i], time_axis[2000:6000], ref_data[i,2000:6000], color=:black, label = "ref")
    lines!(ax[i], time_axis[2000:6000], nomodel_data[i,2000:6000], label = "no model")
    lines!(ax[i], time_axis[2000:6000], smag_data[i,2000:6000], label = "smag")
    #lines!(ax[i], to_data[1][i,1:4000], label = "TO model")
end
axislegend(ax[6], position=:rc)
ax[5].xlabel="t"
ax[6].xlabel="t"
display(g)
save(fig_folder*"/Qoi_trajectories_nomodel_smag.pdf", g)

g = Figure()
ax = [Axis(g[i ÷ 2, i%2], 
    title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
    for i in 0:size(ref_data, 1)-1]
l,r = nothing, nothing
for i in 1:size(ref_data, 1)
    
    for r in 1:5
        l=lines!(ax[i], time_axis[2000:6000], to_data[r][i,2000:6000], color=:blue, alpha = 0.3)
    end
    r=lines!(ax[i], time_axis[2000:6000], ref_data[i,2000:6000], color=:black, label = "ref")
    
end
axislegend(ax[6],[r,l],["ref", "TO LRS"], position=:rc)
ax[5].xlabel="t"
ax[6].xlabel="t"
display(g)
save(fig_folder*"/Qoi_trajectories_TOh5.pdf", g)
end

function rolling_correlation(ref_data, model_data; window_size=40)
    n_qois, n_total = size(ref_data)
    n_windows = n_total-window_size
    timepoints = collect(1:n_windows)
    corr_over_time = zeros(n_windows)

    for w in 1:n_windows
        start_idx = w
        stop_idx = w + window_size
        corrs = zeros(n_qois)
        for q in 1:n_qois
            ref_segment = ref_data[q, start_idx:stop_idx]
            model_segment = model_data[q, start_idx:stop_idx]
            corrs[q] = cor(ref_segment, model_segment)
        end
        corr_over_time[w] = mean(corrs)
    end

    return timepoints, corr_over_time
end

# Compute rolling correlations
t_nomodel, corr_nomodel = rolling_correlation(ref_data[:,1:1000], nomodel_data[:,1:1000])
t_smag, corr_smag = rolling_correlation(ref_data[:,1:1000], smag_data[:,1:1000])
t_to, corr_to = rolling_correlation(ref_data[:,1:1000], to_data[:,1:1000])

# Plot
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax,t_nomodel, corr_nomodel, label = "No model")
lines!(ax,t_smag, corr_smag, label = "Smagorinsky")
lines!(ax,t_to, corr_to, label = "TO model")
axislegend(ax, position=:rb)
display(fig)


## save VTK files
# begin
    ## LinReg1 at t=100
    filename = @__DIR__()*"/paper_runs/output/online/LinReg1/data_online_dns512_les64_Re2000.0_tsim100.0_replica1_rand_initial_dQ.jld2"
    u_final = load(filename)["data_online"].fields[end].u;

    n = 64
    Δx = 1/n
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
                x = (axis_x, axis_x, axis_x),
                Re = Float32(2e3),);
    state = (;u = u_final, t=0., temp=0);        
    # save to vtk
    save_vtk(state; setup, filename = @__DIR__()*"/paper_runs/output/vtks/LinReg1_r1_T100", fieldnames = (:velocity, :Qfield))

    fname = @__DIR__()*"/output/new/data_no_sgs2_dns512_les64_Re2000.0_tsim100.0.jld2"
    no_sgs_data = load(fname, "data_online");
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = no_sgs_data.fields[end].u, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/paper_runs/output/vtks/LF_no_model_T100", fieldnames = (:velocity, :Qfield))

    smag = load(
            @__DIR__()*"/output/new/smag/data_smag_0.071_dns512_les64_Re2000.0_tsim100.0.jld2",
            "data_online")
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = smag.fields[end].u, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/paper_runs/output/vtks/Smag_0071_T100", fieldnames = (:velocity, :Qfield))

    fname = @__DIR__()*"/output/new/data_track2_dns512_les64_Re2000.0_tsim100.0.jld2"
    track_fields = load(fname, "data_track").fields;
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = track_fields[end].u, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/paper_runs/output/vtks/Track_T100", fieldnames = (:velocity, :Qfield))

    fname = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
    train_field = load(fname, "data_train").data[1].u[end];
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = train_field, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/paper_runs/output/vtks/Train_T100", fieldnames = (:velocity, :Qfield))
# end