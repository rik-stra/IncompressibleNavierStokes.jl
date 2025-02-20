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
    n_fields = 1
    filename = @__DIR__()*"/paper_runs/output/online/LinReg1/data_online_dns512_les64_Re2000.0_tsim100.0_replica1_rand_initial_dQ.jld2"
    u_LinReg = load(filename)["data_online"].fields[end:-1:end-n_fields+1]; #41

    fname = @__DIR__()*"/output/new/data_no_sgs2_dns512_les64_Re2000.0_tsim100.0.jld2"
    u_no_sgs = load(fname, "data_online").fields[end:-10:end-10*n_fields+1]; #401

    u_smag = load(@__DIR__()*"/output/new/smag/data_smag_0.07_dns512_les64_Re2000.0_tsim100.0.jld2", "data_online").fields[end:-1:end-n_fields+1]; #41

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
            ["Ref", "No model", "Smagorinsky", "TO"];
            setup,
            sloperange = [2, 16],
            slopeoffset = 3,
            scale_numbers = scales,
        )
    save(fig_folder*"/energy_spectrum_compare_online_nsnaps_$(n_fields).png", fig)

#end


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
            @__DIR__()*"/output/new/smag/data_smag_0.07_dns512_les64_Re2000.0_tsim100.0.jld2",
            "fields")
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = no_sgs_data.fields[end].u, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/paper_runs/output/vtks/Smag_007_T100", fieldnames = (:velocity, :Qfield))

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