using IncompressibleNavierStokes
using RikFlow
using CairoMakie
using JLD2
using LinearAlgebra
using Statistics
using DelimitedFiles
using FFTW

# Domain
xlims = 0, 4 * pi
ylims = 0, 2
zlims = 0, 4 / 3 * pi
# Grid
nx = 64 
ny = 64 
nz = 32 
kwargs = (;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re = 180.f0,
)
setup = Setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
);

function plot_xflow(u_fields, name=nothing)
    if isa(u_fields[1],NamedTuple)
        us = stack(map(x -> x.u, u_fields));
    else
        us = stack(u_fields);
    end
    # mean flow profile last 10 snapshots
    u_ave = mean(us[:,:,:,1,:], dims=[1,3,4])
    u_ave = reshape(u_ave, :)
    u_ave = (u_ave[1:end] + u_ave[end:-1:1])/2
    u_ave_TO = u_ave[2:33]

    # data = load(@__DIR__()*"/output/LF_nomodel_channel_to_64_64_32_tsim50.0.jld2", "fields");
    # u_fields = data[30:50];
    # us = stack(map(x -> x.u, u_fields));
    # # mean flow profile last 10 snapshots
    # u_ave = mean(us[:,:,:,1,:], dims=[1,3,4])
    # u_ave = reshape(u_ave, :)
    # u_ave = (u_ave[1:end] + u_ave[end:-1:1])/2
    # u_ave_NM = u_ave[2:33]

    yp = setup.grid.xu[1][2][2:Int(end//2)]*180

    
    #data = readdlm(@__DIR__()*"/output/LM_Channel_0180_mean_prof.dat", comments=true, comment_char='%')
    #cols = ["y/delta", "y^+", "U", "dU/dy", "W", "P"]
    data = readdlm(@__DIR__()*"/output/Chan180_FD2_all/Chan180_FD2_basic_u.txt", comments=true, comment_char='%')
    cols = ["y^+", "U", "rms(u)",  "<u'u'u'>",  "<u'u'u'u'>", "<u'u'v'>", "<u'w'>"]
    yp_ref = data[2:end, 1]
    u_ave_ref = data[2:end, 2]

    #log plot
    f = Figure()
    ax1 = Axis(f[1, 1], xscale = log10)
    scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
    #scatter!(ax1, yp, u_ave_NM, color=:green, label = "No model")
    scatter!(ax1, yp, u_ave_TO, color=:red, label = "TO")

    ylims!(ax1,0, 19)
    xlims!(ax1, 0.1, 180)
    axislegend(ax1, position = :lt)

    ax1 = Axis(f[1, 2])
    scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
    #scatter!(ax1, yp, u_ave_NM, color=:green, label = "No model")
    scatter!(ax1, yp, u_ave_TO, color=:red, label = "TO")
    ylims!(ax1,0, 19)
    xlims!(ax1, 0.1, 180)

    display(f)
    if !isnothing(name)
        save(@__DIR__()*"/output/figs/$(name)_xflow.pdf", f)
    end
end


# plot trajectories dt 0.005 6QOI
#data = load(@__DIR__()*"/output/online_mirror/LinReg1/LF_online_channel_to_64_64_32_tsim50.0_repl_1.jld2", "data_train");
data = load(@__DIR__()*"/output/LF_6qoi_mirror_track_channel_to_64_64_32_dt0.005_tsim10.0.jld2", "data_train");
q_LF = stack(load(@__DIR__()*"/output/LF_nomodel_mirror_channel_to_64_64_32_tsim50.0.jld2", "qoihist"))
data_ref = load(@__DIR__()*"/output/HF_channel_6qoi_mirror_1framerate_256_256_128_to_64_64_32_tsim10.0.jld2", "f");
q_ref = stack(data_ref.data[1].qoi_hist)[:, 1:5:end]
qois = [["Z",0,3],["E", 0, 3],["Z",4,12],["E", 4, 12],
        ["Z",13,17],["E", 13, 17]];
time_index = 0:0.005:9
g = Figure();
        axs = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
            for i in 0:5];
        for i in 1:6
            lines!(axs[i], time_index, q_ref[i, 1:size(time_index,1)], label = "ref", color = :black)
            #lines!(axs[i], time_index, q_LF[i, 1:size(time_index,1)], label = "nomodel")
            lines!(axs[i],time_index[1:end-1], data.q_star[i, 1:size(time_index,1)-1], label = "q_star")
            lines!(axs[i],time_index, data.q[i, 1:size(time_index,1)], label = "track")
            #lines!(axs[i],time_index, data.q_star[i, 1:size(time_index,1)] .+ data.dQ[i, 1:size(time_index,1)], label = "track")

            if i == 4 axislegend(axs[i], position = :rt) end
            
            if i in [5,6]
                axs[i].xlabel=L"t"
            end
            for i in [1, 2]
                hidexdecorations!(axs[i], ticks = false, grid = false)
            end
        end
        display(g)
        save(@__DIR__()*"/output/figs/track_mirror.png", g)

    u_fields = data.fields[9:11];
    plot_xflow(u_fields, "6qoi_dt0.005")

    # plot trajectories dt 0.01 6QOI
#data = load(@__DIR__()*"/output/online_mirror/LinReg1/LF_online_channel_to_64_64_32_tsim50.0_repl_1.jld2", "data_train");
data = load(@__DIR__()*"/output/LF_6qoi_mirror_track_channel_to_64_64_32_dt0.01_tsim10.0.jld2", "data_train");
q_LF = stack(load(@__DIR__()*"/output/LF_nomodel_mirror_channel_to_64_64_32_tsim50.0.jld2", "qoihist"))
data_ref = load(@__DIR__()*"/output/HF_channel_6qoi_mirror_1framerate_256_256_128_to_64_64_32_tsim10.0.jld2", "f");
q_ref = stack(data_ref.data[1].qoi_hist)[:, 1:10:end]
qois = [["Z",0,3],["E", 0, 3],["Z",4,12],["E", 4, 12],
        ["Z",13,17],["E", 13, 17]];
time_index = 0:0.01:9
g = Figure();
        axs = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
            for i in 0:5];
        for i in 1:6
            lines!(axs[i], time_index, q_ref[i, 1:size(time_index,1)], label = "ref", color = :black)
            #lines!(axs[i], time_index, q_LF[i, 1:size(time_index,1)], label = "nomodel")
            lines!(axs[i],time_index[1:end-1], data.q_star[i, 1:size(time_index,1)-1], label = "q_star")
            lines!(axs[i],time_index, data.q[i, 1:size(time_index,1)], label = "track")
            #lines!(axs[i],time_index, data.q_star[i, 1:size(time_index,1)] .+ data.dQ[i, 1:size(time_index,1)], label = "track")

            if i == 4 axislegend(axs[i], position = :rt) end
            
            if i in [5,6]
                axs[i].xlabel=L"t"
            end
            for i in [1, 2]
                hidexdecorations!(axs[i], ticks = false, grid = false)
            end
        end
        display(g)
        save(@__DIR__()*"/output/figs/track_mirror.png", g)

    u_fields = data.fields[9:11];
    plot_xflow(u_fields, "6qoi_dt0.01")

# plot trajectories dt 0.001
data = load(@__DIR__()*"/output/LF_mirror_track_1frame_channel_to_64_64_32_tsim4.0.jld2", "data_train");
#q_LF = stack(load(@__DIR__()*"/output/LF_nomodel_mirror_channel_to_64_64_32_tsim50.0.jld2", "qoihist"))
data_ref = load(@__DIR__()*"/output/HF_channel_mirror_1framerate_256_256_128_to_64_64_32_tsim10.0.jld2", "f");
q_ref = stack(data_ref.data[1].qoi_hist)
qois = [["Z",0,6],["E", 0, 6],["Z",7,16],["E", 7, 16]];
time_index = 0:0.001:4
g = Figure();
        axs = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
            for i in 0:3];
        for i in 1:4
            lines!(axs[i], time_index, q_ref[i, 1:size(time_index,1)], label = "ref", color = :black)
            #lines!(axs[i], time_index, q_LF[i, 1:size(time_index,1)], label = "nomodel")
            lines!(axs[i],time_index[1:end-1], data.q_star[i, 1:size(time_index,1)-1], label = "q_star")
            lines!(axs[i],time_index, data.q[i, 1:size(time_index,1)], label = "track")
            #lines!(axs[i],time_index, data.q_star[i, 1:size(time_index,1)] .+ data.dQ[i, 1:size(time_index,1)], label = "track")

            if i == 4 axislegend(axs[i], position = :rt) end
            
            if i in [5,6]
                axs[i].xlabel=L"t"
            end
            for i in [1, 2]
                hidexdecorations!(axs[i], ticks = false, grid = false)
            end
        end
        display(g)

u_fields = data.fields[20:41];
u_ref_fields = data_ref.data[1].u[5:11];
plot_xflow(u_ref_fields)
plot_xflow(u_fields)

# plot trajectories dt 0.001
data = load(@__DIR__()*"/output/LF_mirror_track_channel_to_64_64_32_dt0.005_tsim4.0.jld2", "data_train");
#q_LF = stack(load(@__DIR__()*"/output/LF_nomodel_mirror_channel_to_64_64_32_tsim50.0.jld2", "qoihist"))
data_ref = load(@__DIR__()*"/output/HF_channel_mirror_1framerate_256_256_128_to_64_64_32_tsim10.0.jld2", "f");
q_ref = stack(data_ref.data[1].qoi_hist)[:,1:5:end]
qois = [["Z",0,6],["E", 0, 6],["Z",7,16],["E", 7, 16]];
time_index = 0:0.005:4
g = Figure();
        axs = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
            for i in 0:3];
        for i in 1:4
            lines!(axs[i], time_index, q_ref[i, 1:size(time_index,1)], label = "ref", color = :black)
            #lines!(axs[i], time_index, q_LF[i, 1:size(time_index,1)], label = "nomodel")
            lines!(axs[i],time_index[1:end-1], data.q_star[i, 1:size(time_index,1)-1], label = "q_star")
            lines!(axs[i],time_index, data.q[i, 1:size(time_index,1)], label = "track")
            #lines!(axs[i],time_index, data.q_star[i, 1:size(time_index,1)] .+ data.dQ[i, 1:size(time_index,1)], label = "track")

            if i == 4 axislegend(axs[i], position = :rt) end
            
            if i in [5,6]
                axs[i].xlabel=L"t"
            end
            for i in [1, 2]
                hidexdecorations!(axs[i], ticks = false, grid = false)
            end
        end
        display(g)

u_fields = data.fields[2:end];
plot_xflow(u_fields)


# plot trajectories dt 0.01
#data = load(@__DIR__()*"/output/online_mirror/LinReg1/LF_online_channel_to_64_64_32_tsim50.0_repl_1.jld2", "data_train");
data = load(@__DIR__()*"/output/LF_mirror_track_channel_to_64_64_32_tsim10.0.jld2", "data_train");
q_LF = stack(load(@__DIR__()*"/output/LF_nomodel_mirror_channel_to_64_64_32_tsim50.0.jld2", "qoihist"))
data_ref = load(@__DIR__()*"/output/HF_channel_mirror_256_256_128_to_64_64_32_tsim10.0.jld2", "f");
q_ref = stack(data_ref.data[1].qoi_hist)
qois = [["Z",0,6],["E", 0, 6],["Z",7,16],["E", 7, 16]];
time_index = 0:0.01:4
g = Figure();
        axs = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
            for i in 0:3];
        for i in 1:4
            lines!(axs[i], time_index, q_ref[i, 1:size(time_index,1)], label = "ref", color = :black)
            #lines!(axs[i], time_index, q_LF[i, 1:size(time_index,1)], label = "nomodel")
            lines!(axs[i],time_index[1:end-1], data.q_star[i, 1:size(time_index,1)-1], label = "q_star")
            lines!(axs[i],time_index, data.q[i, 1:size(time_index,1)], label = "track")
            #lines!(axs[i],time_index, data.q_star[i, 1:size(time_index,1)] .+ data.dQ[i, 1:size(time_index,1)], label = "track")

            if i == 4 axislegend(axs[i], position = :rt) end
            
            if i in [5,6]
                axs[i].xlabel=L"t"
            end
            for i in [1, 2]
                hidexdecorations!(axs[i], ticks = false, grid = false)
            end
        end
        display(g)
        save(@__DIR__()*"/output/figs/track_mirror.png", g)

    u_fields = data.fields[9:end];
    plot_xflow(u_fields, "xflow_4qoi_dt0.01")

# plot trajectories dt 0.01 -- non mirrored
data = load(@__DIR__()*"/output/LF_track_channel_to_64_64_32_tsim10.0.jld2", "data_train");
data_ref = load(@__DIR__()*"/output/checkpoints/checkpoint_n50000.jld2", "results");
q_ref = stack(data_ref.data[1].qoi_hist)
qois = [["Z",0,6],["E", 0, 6],["Z",7,16],["E", 7, 16]];
time_index = 0:0.01:9
g = Figure();
        axs = [Axis(g[i ÷ 2, i%2], 
        title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
            for i in 0:3];
        for i in 1:4
            lines!(axs[i], time_index, q_ref[i, 1:size(time_index,1)], label = "ref", color = :black)
            #lines!(axs[i], time_index, q_LF[i, 1:size(time_index,1)], label = "nomodel")
            lines!(axs[i],time_index, data.q_star[i, 1:size(time_index,1)], label = "q_star")
            lines!(axs[i],time_index, data.q[i, 1:size(time_index,1)], label = "track")
            
            if i == 4 axislegend(axs[i], position = :rt) end
            
            if i in [5,6]
                axs[i].xlabel=L"t"
            end
            for i in [1, 2, 3, 4]
                hidexdecorations!(axs[i], ticks = false, grid = false)
            end
        end
        display(g)
save(@__DIR__()*"/output/figs/track.png", g)
u_fields = data.fields[5:10];
plot_xflow(u_fields)

##### PLOT energy / enstrophy density

u_start = load(@__DIR__()*"/output/HF_channel_mirror_1framerate_256_256_128_to_64_64_32_tsim10.0.jld2", "f").data[1].u[1];
heatmap(u_start[:,:,20,3])

qois = [["Z",0,3],["E", 0, 3],["Z",4,12],["E", 4, 12],
        ["Z",13,17],["E", 13, 17]];
to_setup = 
        RikFlow.TO_Setup(; qois, 
        to_mode = :TRACK_REF, 
        ArrayType=Array, 
        setup,
        nstep=10,
        mirror_y = true,);
u_hat = RikFlow.get_u_hat(u_start, setup, to_setup);
w_hat = RikFlow.get_w_hat_from_u_hat(u_hat, to_setup);
qd = RikFlow.compute_filtered_qoi_fields(u_hat, w_hat, to_setup, setup);


let
g = Figure(size = (800, 700));
axs = [Axis(g[i ÷ 2, i%2], 
        title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
    for i in 0:size(qois, 1)-1]

for i in 1:size(qois, 1)
    heatmap!(axs[i], sum(abs2,real(ifft(qd[i],[1,2,3])),dims = 4)[:,1:Int(end/2),30])
end
display(g)
end



heatmap(sum(abs2,real(ifft(qd[4],[1,2,3])),dims = 4)[:,:,20])

