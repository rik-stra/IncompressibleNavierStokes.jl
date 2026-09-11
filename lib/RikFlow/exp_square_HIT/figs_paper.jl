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
filename = @__DIR__()*"/output/paper_data_HIT/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
q_ref = stack(ref_data.data[1].qoi_hist)


###############################################
# Energy spectra initial turbulent field (after spinnup)
###############################################
let # energy spectrum HF
    filename =  @__DIR__()*"/output/paper_data_HIT/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2"
    u_start = stack(load(filename, "u_start"));
    n = 512
    Δx = 1/n
    axis_x = range(0.0, 1., n + 1)
    setup = rf_setup(;
                x = (axis_x, axis_x, axis_x),
                Re = Float32(2e3),);
    state = (;u = u_start, t=0., temp=0);
            
    scales = turbulence_statistics(u_start, setup, 1/setup.Re)
    fig = rf_energy_spectrum_plot(state; setup, npoint = 100, sloperange = [2,16], v_lines = [scales.l_tay, scales.l_kol, Δx], slopeoffset = 1.8, scale_numbers = scales, plot_wavelength = true)
    display(fig)
    v = [scales.l_tay, scales.l_kol, 1/n]
    v_labels = ["λ", "η", "Δx"]
    for i in 1:3
        text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1e-12*1.2), align = (:left, :bottom), color = :black)
    end
    display(fig)
    save(fig_folder*"/energy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.pdf", fig)
end

let ## energy spectrum coarse grained
    u_start_lf = ref_data.data[1].u[1];
    heatmap(u_start_lf[end-1, :, :, 1]) # initial coarse field
    n = 64
    Δx = 1/n
    axis_x = range(0.0, 1., n + 1)
    setup = rf_setup(;
                x = (axis_x, axis_x, axis_x),
                Re = Float32(2e3),);
    state = (;u = u_start_lf, t=0., temp=0);
    scales_LF = turbulence_statistics(u_start_lf, setup, 1/setup.Re)
    scales = (; diss = 3.7794485f0)
    fig = rf_energy_spectrum_plot(state; setup, npoint = 100, sloperange = [2,16], v_lines = [6.5,15.5,32], slopeoffset = 1.8, scale_numbers = scales)
    v_labels = ["[0,6]", "[7,15]", "[16,32]"]
    v = [3, 11, 24]
    for i in 1:3
        text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1*0.5), align = (:center, :center), color = :black)
    end
    display(fig)
    save(fig_folder*"/energy_spectrum_afterspinup_coarse_grained_Re2000.0_freeze_10_tsim4.pdf", fig)
end


##################################
## Compare energy-spectra of final fields
##################################
let
    n_fields = 10
    fname = @__DIR__()*"/output/paper_data_HIT/TO_LRS/LinReg1/data_online_tsim100.0_replica1.jld2"
    u_LinReg = load(fname,"data_online").fields[end:-1:end-n_fields+1]; #41

    fname = @__DIR__()*"/output/paper_data_HIT/no_model/data_no_sgs_tsim100.0.jld2"
    u_no_sgs = load(fname, "data_online").fields[end:-1:end-n_fields+1]; #41

    u_smag = load(@__DIR__()*"/output/paper_data_HIT/smag/data_smag_0.071_dns512_les64_Re2000.0_tsim100.0.jld2", "data_online").fields[end:-1:end-n_fields+1]; #41

    fname = @__DIR__()*"/output/paper_data_HIT/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
    u_ref = load(fname, "data_train").data[1].u[end:-10:end-10*n_fields+1]; #401

    n = 64
    Δx = 1/n
    axis_x = range(0.0, 1., n + 1)
    setup = rf_setup(;
                x = (axis_x, axis_x, axis_x),
                Re = Float32(2e3),);
    states = [ u_ref,
            u_LinReg,
            u_no_sgs,
            u_smag,];
    #scales = turbulence_statistics(u_ref[1], setup, 1/setup.Re)
    scales = (; diss = 3.7794485f0) # taken from HF_ref
    fig = energy_spectra_comparison(
            states,
            ["Ref",  "TO LRS h=5", "No model", "Smagorinsky",];
            setup,
            sloperange = [2, 16],
            slopeoffset = 3,
            scale_numbers = scales,
            linestyles = [:solid, :solid, :dash, :dashdot, ],
        )
    save(fig_folder*"/energy_spectrum_compare_online_nsnaps_$(n_fields).pdf", fig)
    display(fig)
end

###################################
## plot short trajectories
###################################
let # no model and smagorinsky

    fname = @__DIR__()*"/output/paper_data_HIT/no_model/data_no_sgs_tsim100.0.jld2"
    nomodel_data = load(fname, "data_online").q;
    smag_data = load(@__DIR__()*"/output/paper_data_HIT/smag/data_smag_0.071_dns512_les64_Re2000.0_tsim100.0.jld2", "data_online").q;

    time_axis = 0:2.5e-3:100

    g = Figure()
    ax = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(q_ref, 1)-1]
    for i in 1:size(q_ref, 1)
        lines!(ax[i], time_axis[2000:6000], q_ref[i,2000:6000], color=:black, label = "Ref")
        lines!(ax[i], time_axis[2000:6000], nomodel_data[i,2000:6000], label = "No model", linestyle=:dash)
        lines!(ax[i], time_axis[2000:6000], smag_data[i,2000:6000], label = "Smag", linestyle=:dashdot)
        #lines!(ax[i], to_data[1][i,1:4000], label = "TO model")
    end
    axislegend(ax[6], position=:rc)
    ax[5].xlabel="t"
    ax[6].xlabel="t"
    display(g)
    save(fig_folder*"/Qoi_trajectories_nomodel_smag.pdf", g)
end

let # TO LRS
    to_data = [load(@__DIR__()*"/output/paper_data_HIT/TO_LRS/LinReg1/data_online_tsim100.0_replica$(i).jld2","data_online").q for i in 1:5]
    time_axis = 0:2.5e-3:100
    g = Figure()
    ax = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(q_ref, 1)-1]
    l,r = nothing, nothing
    for i in 1:size(q_ref, 1)
        
        for r in 1:5
            l=lines!(ax[i], time_axis[2000:6000], to_data[r][i,2000:6000], color=:blue, alpha = 0.3)
        end
        r=lines!(ax[i], time_axis[2000:6000], q_ref[i,2000:6000], color=:black, label = "Ref")
        
    end
    axislegend(ax[6],[r,l],["Ref", "TO LRS"], position=:rc)
    ax[5].xlabel="t"
    ax[6].xlabel="t"
    display(g)
    save(fig_folder*"/Qoi_trajectories_TOh5.pdf", g)
end

##################################
## plot long-term distributions
##################################
function plot_long_term_distr(data, q_ref, label, qois)
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(data, 1)-1]
    for i in 1:size(data, 1)
        density!(axs[i], q_ref[i, :], label = "Ref", color = (:black, 0.3),
        strokecolor = :black, strokewidth = 3, strokearound = false)
        density!(axs[i], data[i, :], label = label, color = (:red, 0.3),
        strokecolor = :red, strokewidth = 3, linestyle=:dot, strokearound = false)
        hideydecorations!(axs[i], label = false, ticks = false, grid = false)
        if i == size(data, 1) axislegend(axs[i], position = :rt) end
        if i in [1,3,5]
            axs[i].ylabel="Density"
        end
    end
    return g
end

let  # no model
    fname = @__DIR__()*"/output/paper_data_HIT/no_model/data_no_sgs_tsim100.0.jld2"
    no_sgs_data = load(fname, "data_online");
    g = plot_long_term_distr(no_sgs_data.q, q_ref, "No model", qois)
    display(g)
    save(fig_folder*"/lt_distr_q_nomodel_dns512_les64_Re2000.0_tsim100.pdf", g)
end


let # smagorisky 0.71
    smag_val = 0.071
    smag_data = load(
        @__DIR__()*"/output/paper_data_HIT/smag/data_smag_$(smag_val)_dns512_les64_Re2000.0_tsim100.0.jld2",
        "data_online").q
    g = plot_long_term_distr(smag_data, q_ref, "Smag $smag_val", qois)
    display(g)
    save(fig_folder*"/lt_distr_q_smag_dns512_les64_Re2000.0_tsim100.pdf", g)
end

let 
    linreg_id = 1
    linreg_data = [load(
        @__DIR__()*"/output/paper_data_HIT/TO_LRS/LinReg$(linreg_id)/data_online_tsim100.0_replica$(i).jld2",
        "data_online").q for i in 1:5]
    qs = cat(linreg_data..., dims = 2)
    g = plot_long_term_distr(qs, q_ref, "TO LRS", qois)
    display(g)
    save(fig_folder*"/lt_distr_q_TO_ensemble.pdf", g)
end

let 
    linreg_id = 63
    linreg_data = [load(
        @__DIR__()*"/output/paper_data_HIT/TO_LRS/LinReg$(linreg_id)/data_online_tsim100.0_replica$(i).jld2",
        "data_online").q for i in 1:5]
    qs = cat(linreg_data..., dims = 2)
    g = plot_long_term_distr(qs, q_ref, "TO LRS", qois)
    display(g)
    save(fig_folder*"/lt_distr_q_TO_ensemble_h10_l001.pdf", g)
end

let 
    linreg_id = 64
    linreg_data = [load(
        @__DIR__()*"/output/paper_data_HIT/TO_LRS/LinReg$(linreg_id)/data_online_tsim100.0_replica$(i).jld2",
        "data_online").q for i in 1:5]
    qs = cat(linreg_data..., dims = 2)
    g = plot_long_term_distr(qs, q_ref, "TO LRS", qois)
    display(g)
    save(fig_folder*"/lt_distr_q_TO_ensemble_h10_l0.pdf", g)
end


###################################
## plot smag tuning
###################################

let     
    smag_vals = [0.05, 0.055, 0.06, 0.065, 0.07, 0.071, 0.072, 0.073, 0.075, 0.077, 0.08, 0.085, 0.09, 0.095, 0.1]
    smag_data = [load(
        @__DIR__()*"/output/paper_data_HIT/smag/data_smag_$(c)_dns512_les64_Re2000.0_tsim100.0.jld2",
        "data_online").q for c in smag_vals];
    ks_dists = []
    for j in 1:size(smag_vals,1)
        ks = [ks_dist(q_ref[i,:], smag_data[j][i,:])[1] for i in 1:size(q_ref, 1)]
        push!(ks_dists, ks)
    end
    ks_dists = stack(ks_dists)
    g = Figure()
    axs = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(smag_data[1], 1)-1]

    for i in 1:size(smag_data[1], 1)
        scatterlines!(axs[i], smag_vals, ks_dists[i,:])
        
        #if i == size(smag_data[1], 1) axislegend(axs[i], position = :rt) end
        if i in [1,3,5]
            axs[i].ylabel="KS-distance"
        end
        if i in [5,6]
            axs[i].xlabel=L"C_s"
        end
        for i in [1, 2, 3, 4]
            hidexdecorations!(axs[i], ticks = false, grid = false)
        end
    end

    display(g)
    save(fig_folder*"/KSdists_smag_dns512_les64_Re2000.0_tsim100.pdf", g)
end


####################################
## save VTK files final fields
####################################
begin
    ## LinReg1 at t=100
    filename = @__DIR__()*"/output/paper_data_HIT/TO_LRS/LinReg1/data_online_tsim100.0_replica1.jld2"
    u_final = load(filename,"data_online").fields[end].u;
    n = 64
    Δx = 1/n
    axis_x = range(0.0, 1., n + 1)
    setup = rf_setup(;
                x = (axis_x, axis_x, axis_x),
                Re = Float32(2e3),);
    state = (;u = u_final, t=0., temp=0);        
    # save to vtk
    save_vtk(state; setup, filename = @__DIR__()*"/figures/vtks/LinReg1_r1_T100", fieldnames = (:velocity, :qcrit))

    fname = @__DIR__()*"/output/paper_data_HIT/no_model/2data_no_sgs_tsim100.0.jld2"
    no_sgs_data = load(fname, "data_online");
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = rf_setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = no_sgs_data.fields[end].u, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/figures/vtks/LF_no_model_T100", fieldnames = (:velocity, :qcrit))

    smag = load(
            @__DIR__()*"/output/paper_data_HIT/smag/data_smag_0.071_dns512_les64_Re2000.0_tsim100.0.jld2",
            "data_online");
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = rf_setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = smag.fields[end].u, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/figures/vtks/Smag_0071_T100", fieldnames = (:velocity, :qcrit))

    fname = @__DIR__()*"/output/paper_data_HIT/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
    train_field = load(fname, "data_train").data[1].u[end];
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = rf_setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = train_field, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/figures/vtks/Train_T100", fieldnames = (:velocity, :qcrit))
end