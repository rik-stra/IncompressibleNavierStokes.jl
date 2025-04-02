using JLD2
using CairoMakie
using IncompressibleNavierStokes
using Statistics
using RikFlow

# create folder for figures
fig_folder = @__DIR__()*"/output/figures_long"
if !isdir(fig_folder)
    mkdir(fig_folder)
end

# We first run spinnup.jl to generate an initial condition for the DNS simulation.

#begin # load initial field
    filename =  @__DIR__()*"/output/u_start_spinnup_512_Re2000.0_freeze_10_tsim4.0.jld2"
    u_start = stack(load(filename, "u_start"));

    heatmap(u_start[end-1, :, :,1])
    # plot energy spectrum
    
        n = 512
        axis_x = range(0.0, 1., n + 1)
        setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
        state = (;u = u_start, t=0., temp=0);
        
        # save to vtk

        save_vtk(state; setup, filename = @__DIR__()*"/output/vtks", fieldnames = (:velocity, :Qfield))

        scales = get_scale_numbers(u_start, setup)
        fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [2,16], scale_numbers = scales, slopeoffset = 1.8)
        display(fig)
        v = [scales.λ, scales.η, 1/n]
        v_labels = ["λ", "η", "Δx"]
        for i in 1:3
            text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1e-15*1.2), align = (:left, :bottom), color = :black)
        end
        display(fig)
        save(fig_folder*"/energy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.png", fig)
        println(scales)

    # plot enstrophy spectrum
    let
        n = 512
        axis_x = range(0.0, 1., n + 1)
        setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
        state = (;u = u_start, t=0.);
        fig = enstrophy_spectrum_plot(state; setup, npoint = 100)  # not compatible with last merge
        save(fig_folder*"/enstrophy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.png", fig)
    end
#end

######################################################################
# Using HF_ref.jl, we collect the reference trajectories of the qois
# load reference data
######################################################################
filename = @__DIR__()*"/output/new/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0.jld2"
ref_data = load(filename, "data_train");
qois = [["Z",0,6],["E", 0, 6],["Z",7,15],["E", 7, 15],["Z",16,32],["E", 16, 32]]
q_ref = stack(ref_data.data[1].qoi_hist)
t_sim = 50
time_index = 0:2.5e-3:t_sim

# begin some plots
    let # plot reference data
        g = Figure()
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(q_ref, 1)-1]
        for i in 1:size(q_ref, 1)
            lines!(axs[i], time_index, q_ref[i,1:size(time_index,1)])
            xlims!(axs[i], -0.4, t_sim) 
        end
        display(g)
        save(fig_folder*"/q_ref_dns512_les64_Re2000.0_freeze_10_tsim100.png", g)
    end

    # plot fancy histograms
    let
        g = Figure()
        xvals = ["5", "10", "20", "100"]
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]",
            xticks = ([0.6, 1.2, 1.8, 2.4], xvals),
            )
            for i in 0:size(q_ref, 1)-1]
        Q_scaled = q_ref ./ (std(q_ref, dims = 2))
        n = size(q_ref, 2)
        for i in 1:size(q_ref, 1)
            for t in 1:4
                density!(axs[i], Q_scaled[i, 1:parse(Int,xvals[t])*400], offset = t*(0.6), color = (:slategray, 0.5), direction=:y)
            end
        end
        display(g)
        save(fig_folder*"/Q_hist_overtime_dns512_les64_Re2000.0_tsim100.png", g)
    end

    u_lf = ref_data.data[1].u;
    heatmap(u_lf[7][end-1, :, :, 1]) # initial coarse field
    
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = u_lf[1], t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/output/vtk_files/LF_initial", fieldnames = (:velocity, :Qfield))
    state = (;u = u_lf[10], t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/output/vtk_files/LF_10", fieldnames = (:velocity, :Qfield))
    state = (;u = u_lf[100], t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/output/vtk_files/LF_100", fieldnames = (:velocity, :Qfield))
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
# end

### Track ref ###################################################################
# We now run track_ref.jl to track the reference trajectories of the qois
#################################################################################
# begin
    fname = @__DIR__()*"/output/new/data_track2_dns512_les64_Re2000.0_tsim100.0.jld2"
    track_data = load(fname, "data_track");
    #fname = @__DIR__()*"/output/new/data_track2_dns512_les64_Re2000.0_tsim10.0.jld2"
    #track_data2 = load(fname, "data_track");
    # plot dQ data
    #trajectories
    let 
        t_sim = 10
        time_index = 0:2.5e-3:t_sim
        g = Figure()
        range = 1:4000
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(track_data.dQ, 1)-1]
        for i in 1:size(track_data.dQ, 1)
            lines!(axs[i], time_index[1:end-1], track_data.dQ[i, range])
            #lines!(axs[i], track_data2.dQ[i, range], linestyle = :dash ,color = :red)
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
        xvals = ["5", "10", "20", "100"]
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "d$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]",
            xticks = ([0.6, 1.2, 1.8, 2.4], xvals),
            )
            for i in 0:size(track_data.dQ, 1)-1]
        dQ_scaled = track_data.dQ ./ (std(track_data.dQ, dims = 2))
        n = size(track_data.dQ, 2)
        for i in 1:size(track_data.dQ, 1)
            for t in 1:4
                density!(axs[i], dQ_scaled[i, 1:parse(Int,xvals[t])*400], offset = t*(0.6), color = (:slategray, 0.5), direction=:y)
            end
        end
        display(g)
        save(fig_folder*"/dQ_hist_overtime_dns512_les64_Re2000.0_tsim10.png", g)
    end

    ## plot corrected trajectories
    let 
        interval = 2900:3000
        time_index = interval*2.5e-3
        g = Figure()
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(track_data.q, 1)-1]
        for i in 1:size(track_data.q, 1)
            lines!(axs[i], time_index, q_ref[i, interval], label = "ref", color = :black)
            lines!(axs[i], time_index, track_data.q_star[i, interval], label = "*")
            lines!(axs[i], time_index, track_data.q[i, interval], linestyle = :dash, label = "corrected")
        end
        axislegend(position = :rt)
        display(g)
        save(fig_folder*"/corrected_trajectories.png", g)
    end

    ##
    let 
        interval = 4850:4953
        time_index = interval*2.5e-3
        g = Figure(size = (400,300))
        axs = Axis(g[1,1], 
            title = "$(qois[2][1])_[$(qois[2][2]), $(qois[2][3])]",
            xlabel = "t",
            limits = ((12.15, 12.5),(1.67, 1.9)),
            )
            
        
        lines!(axs, time_index, q_ref[2, interval], label = "HF", linewidth = 2)
        #lines!(axs, time_index, track_data.q_star[i, interval], label = "*")
        lines!(axs, time_index, track_data.q[2, interval], linestyle = :dash, label = "LF", linewidth = 2)
        
        axislegend(position = :rt)
        display(g)
        save(fig_folder*"/pred_corr_tracking_for_presentation.png", g)
    end

# end

### no SGS ###
##############
# begin
    fname = @__DIR__()*"/output/new/data_no_sgs2_dns512_les64_Re2000.0_tsim100.0.jld2"
    no_sgs_data = load(fname, "data_online");


    let 
        g = Figure()
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(no_sgs_data.q, 1)-1]
        for i in 1:size(no_sgs_data.q, 1)
            lines!(axs[i], time_index, q_ref[i, 1:size(time_index,1)], label = "ref")
            lines!(axs[i], time_index, no_sgs_data.q[i, 1:size(time_index,1)], label = "no sgs")
            #lines!(axs[i], no_sgs_data2.q[i, :], label = "no sgs2", color = :red)
            if i == size(no_sgs_data.q, 1) axislegend(axs[i], position = :rt) end
        end
        display(g)
        save(fig_folder*"/q_no_sgs_dns512_les64_Re2000.0_tsim100.png", g)
    end

    # plot distrubution of QoIs
    let 
        g = Figure()
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(no_sgs_data.q, 1)-1]
        for i in 1:size(no_sgs_data.q, 1)
            density!(axs[i], q_ref[i, :], label = "ref", color = (:black, 0.3),
            strokecolor = :black, strokewidth = 3, strokearound = false)
            density!(axs[i], no_sgs_data.q[i, :], label = "no model", color = (:red, 0.3),
            strokecolor = :red, strokewidth = 3, strokearound = false)
            if i == size(no_sgs_data.q, 1) axislegend(axs[i], position = :rt) end
        end
        display(g)
        save(fig_folder*"/lt_distr_q_nomodel_dns512_les64_Re2000.0_tsim100.png", g)
    end

    ## plot final field
    heatmap(no_sgs_data.fields[end].u[1,:,:,1])
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = no_sgs_data.fields[end].u, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/output/vtk_files/LF_no_model", fieldnames = (:velocity, :Qfield))
#end

### SMAG  ###
#############
#begin
    #let 
        smag_vals = [0.05, 0.055, 0.06, 0.065, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.08, 0.085, 0.09, 0.095,0.1]
        smag_data = [load(
            @__DIR__()*"/output/new/smag/data_smag_$(c)_dns512_les64_Re2000.0_tsim100.0.jld2",
            "data_online").q for c in smag_vals];
        ks_dists = []
        for j in 1:size(smag_vals,1)
            ks = [ks_dist(q_ref[i,:], smag_data[j][i,:])[1] for i in 1:size(q_ref, 1)]
            push!(ks_dists, ks)
        end
        ks_dists = stack(ks_dists)
        g = Figure(size = (900,600))
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(smag_data[1], 1)-1]

        for i in 1:size(smag_data[1], 1)
            lines!(axs[i], smag_vals, ks_dists[i,:])
            #if i == size(smag_data[1], 1) axislegend(axs[i], position = :rt) end
        end
        display(g)
        save(fig_folder*"/KSdists_smag_dns512_les64_Re2000.0_tsim100.png", g)
        
        summed_ks = sum(ks_dists, dims = 1)[:]
        g = Figure()
        ax = Axis(g[1, 1], xlabel = "Smag constant", ylabel = "Summed KS distance")
        lines!(ax,smag_vals,summed_ks)
        display(g)
        save(fig_folder*"/summed_KSdists_smag_dns512_les64_Re2000.0_tsim100.png", g)
    #end
    ks_dists = [ks_dist(q_ref[i,:], smag_data[1][i,:])[1] for i in 1:size(q_ref, 1)]
    ks_dists2 = [ks_dist(q_ref[i,:], smag_data[2][i,:])[1] for i in 1:size(q_ref, 1)]
    sum(ks_dists)
    
    smag_vals = [0.07]
    smag_data = [load(
        @__DIR__()*"/output/new/smag/data_smag_$(c)_dns512_les64_Re2000.0_tsim100.0.jld2",
        "data_online").q for c in smag_vals];

    let 
        time_index = 0:2.5e-3:10
        g = Figure(size = (900,600))
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(smag_data[1], 1)-1]
        for i in 1:size(smag_data[1], 1)
            lines!(axs[i], time_index, q_ref[i, 1:size(time_index,1)], label = "ref", color = :black)
            for j in 1:length(smag_vals)
                lines!(axs[i],time_index, smag_data[j][i, 1:size(time_index,1)], label = "smag $(smag_vals[j])")
            end
            if i == size(smag_data[1], 1) axislegend(axs[i], position = :rt) end
        end
        display(g)
        save(fig_folder*"/q_smag_dns512_les64_Re2000.0_tsim10.png", g)
    end

    
    let 
        g = Figure()
        axs = [Axis(g[i ÷ 2, i%2], 
            title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
            for i in 0:size(smag_data[1], 1)-1]
        for i in 1:size(smag_data[1], 1)
            density!(axs[i], q_ref[i, :], label = "ref", color = (:black, 0.3),
            strokecolor = :black, strokewidth = 3, strokearound = true)
            for j in 1:length(smag_vals)
                density!(axs[i], smag_data[j][i, :], label = "smag $(smag_vals[j])", color = (:red, 0.3),
                strokecolor = :red, strokewidth = 3, strokearound = true)
            end
            if i == size(smag_data[1], 1) axislegend(axs[i], position = :rt) end
        end
        display(g)
        save(fig_folder*"/lt_distr_q_smag_dns512_les64_Re2000.0_tsim100.png", g)
    end
    smag_fields = [load(
        @__DIR__()*"/output/new/smag/data_smag_$(c)_dns512_les64_Re2000.0_tsim100.0.jld2",
        "data_online").fields for c in smag_vals];
    heatmap(smag_fields[1][end].u[1,:,:,1])
    n = 64
    axis_x = range(0.0, 1., n + 1)
    setup = Setup(;
            x = (axis_x, axis_x, axis_x),
            Re = Float32(2e3),);
    state = (;u = smag_fields[1][end].u, t=0., temp=0);
    save_vtk(state; setup, filename = @__DIR__()*"/output/vtk_files/LF_smag", fieldnames = (:velocity, :Qfield))
#end


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