using CairoMakie
using IncompressibleNavierStokes
using RikFlow
using FFTW
using JLD2
using Statistics

figs_folder = @__DIR__()*"/figures"
ispath(figs_folder) || mkdir(figs_folder)

let # HF initial turbulence field
    # Domain
    xlims = 0, 4 * pi
    ylims = 0, 2
    zlims = 0, 4 / 3 * pi
    # Grid
    nx = 512 
    ny = 512 
    nz = 256 
    setup = rf_setup(;
        x = (
            range(xlims..., nx + 1),
            range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
            range(zlims..., nz + 1)
        ),
        boundary_conditions = (; u = (
            (PeriodicBC(), PeriodicBC()),
            (DirichletBC(), DirichletBC()),
            (PeriodicBC(), PeriodicBC()),
        )),
        Re = 180.0,
    );

    u_start = load(@__DIR__()*"/output/paper_data_channel/HF/u_start_T15_512_512_256.jld2", "u_start");
    y_ax = setup.xu[1][2]
    x_ax = setup.xu[1][1]

    f = Figure(size = (900, 200));
    ax1 = Axis(f[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "y")
    heatmap!(ax1,x_ax[1:end-2], y_ax, (u_start[1:end-2,:,1,1]+ u_start[1:end-2,:,2,1])/2)
    display(f)
    name = figs_folder*"/u_start.png"
    save(name, f)
    run(`magick $name -trim $name`)
end

let #plot coarse initial turbulence field   
    # Domain
    xlims = 0, 4 * pi
    ylims = 0, 2
    zlims = 0, 4 / 3 * pi
    # Grid
    nx = 64 
    ny = 64 
    nz = 32 
    setup = rf_setup(;
        x = (
            range(xlims..., nx + 1),
            range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
            range(zlims..., nz + 1)
        ),
        boundary_conditions = (; u = (
            (PeriodicBC(), PeriodicBC()),
            (DirichletBC(), DirichletBC()),
            (PeriodicBC(), PeriodicBC()),
        )),
        Re = 180.0,
    );

    ustart = Array(load(@__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2")["f"].data[1].u[1]);
    y_ax = setup.xu[1][2]
    x_ax = setup.xu[1][1]

    f = Figure(size = (900, 200));
    ax1 = Axis(f[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "y")
    heatmap!(ax1,x_ax[1:end-2], y_ax, (ustart[1:end-2,:,1,1]+ ustart[1:end-2,:,2,1])/2)
    display(f)
    name = figs_folder*"/u_start_coarse.png"
    save(name, f)
    run(`magick $name -trim $name`)
end

# Plot QoI trajectories for the TO LRS simulations
for lr in ["LinReg11", "LinReg13"]
    qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10], ["Z",11,17],["E", 11, 17]];
    n_replicas = 5
    time_index = 0:0.005:100
    data = [load(@__DIR__()*"/output/paper_data_channel/TO_LRS/$(lr)/LF_online_channel_to_64_64_32_tsim100.0_repl_$(i).jld2", "data")
            for i in 1:n_replicas];
    q_rep = map(x -> x.q ./2, data)
    hf_data = load(@__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2");
    q_ref = stack(hf_data["f"].data[1].qoi_hist)./2

    g = Figure(size = (700, 600))
    ref, model= nothing, nothing
    axs = [Axis(g[i ÷ 2, i%2], 
            title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(q_ref, 1)-1]

    for i in 1:size(q_ref, 1)
        for j in 1:n_replicas
                model=lines!(axs[i],time_index, q_rep[j][i,:], color = (:blue, 0.25)) 
        end
        xlim_right = min(maximum(size.(q_rep,2)), size(time_index,1))
        ref = lines!(axs[i], time_index[1:2001], q_ref[i,1:5:10001], color = :black)
    end
    axislegend(axs[6],[ref, model],["HF", "TO LRS"], position=:rc)
    axs[5].xlabel="t"
    axs[6].xlabel="t"
    display(g)
    if lr == "LinReg11"
        save(figs_folder*"/Channel_TO_LRS5_q_trajectories.pdf", g)
    end
    if lr == "LinReg13"
        save(figs_folder*"/Channel_TO_LRS10_q_trajectories.pdf", g)
    end
end

let
    hf_data = load(@__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2");
    q_ref = stack(hf_data["f"].data[1].qoi_hist)./2
    wale_data = load(@__DIR__()*"/output/paper_data_channel/WALE/LF_wale_channel_c0.53_tsim100.0.jld2","qoihist");
    q_wale = stack(wale_data)./2
    nomodel_data = load(@__DIR__()*"/output/paper_data_channel/nomodel/LF_nomodel_channel_tsim100.0.jld2","qoihist");
    q_no_model = stack(nomodel_data)./2
    smag_data = load(@__DIR__()*"/output/paper_data_channel/smag/LF_smag_channel_c0.13_tsim100.0.jld2","qoihist");
    q_smag = stack(smag_data)./2
    TO_5_r1 = load(@__DIR__()*"/output/paper_data_channel/TO_LRS/LinReg11/LF_online_channel_to_64_64_32_tsim100.0_repl_1.jld2", "data");
    q_TO_5_r1 = TO_5_r1.q ./2

    qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10], ["Z",11,17],["E", 11, 17]];
    time_index = 0:0.005:100

    g = Figure(size = (700, 650))
    smag, ref, no_model, wale, TO_5= nothing, nothing, nothing, nothing, nothing
    axs = [Axis(g[i ÷ 2, i%2], 
            title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(q_ref, 1)-1]

    for i in 1:size(q_ref, 1)
        no_model = lines!(axs[i], time_index, q_no_model[i,:], color = (:red), linestyle = :dash)
        wale=lines!(axs[i],time_index, q_wale[i,:], color = (:orange))
        smag=lines!(axs[i],time_index, q_smag[i,:], color = (:purple), linestyle = :dashdot)
        ref = lines!(axs[i], time_index[1:2001], q_ref[i,1:5:10001], color = :black)
        TO_5 = lines!(axs[i], time_index, q_TO_5_r1[i,:], color = (:blue, 0.4), linestyle = :dot)
    end
    axislegend(axs[6],[ref, no_model, wale, smag, TO_5], ["HF", "No model" ,"WALE", "Smag", "TO LRS \n h=5 repl. 1"], position=:rc)
    axs[5].xlabel="t"
    axs[6].xlabel="t"
    display(g)
    save(figs_folder*"/Channel_eddyvisc_q_trajectories.pdf", g)

end

#########################################
# Plot mean velocity profiles for the channel flow
#########################################
# Domain
xlims = 0, 4 * pi
ylims = 0, 2
zlims = 0, 4 / 3 * pi
# Grid
nx = 64 
ny = 64 
nz = 32 
setup = rf_setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    boundary_conditions = (; u = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    )),
    Re = 180,
);

function get_u_ave(u_fields, dim = 1)
    if isa(u_fields[1], Array)
        us = stack(u_fields)
    else
        us = stack(map(x -> x.u, u_fields));
    end
    u_ave = mean(us[1:end-2,:,1:end-2,dim,:], dims=[1,3,4])
    u_ave = reshape(u_ave, :)
    u_ave = (u_ave[1:end] + u_ave[end:-1:1])/2
    return u_ave[2:33]
end

data = [load(@__DIR__()*"/output/paper_data_channel/TO_LRS/LinReg11/LF_online_channel_to_64_64_32_tsim100.0_repl_$i.jld2", "data").fields[2:end] for i=1:1];
data2 = cat(data..., dims=1);
u_ave_TO5 = get_u_ave(data2);

data = [load(@__DIR__()*"/output/paper_data_channel/TO_LRS/LinReg13/LF_online_channel_to_64_64_32_tsim100.0_repl_$i.jld2", "data").fields[2:end] for i=1:5];
data2 = cat(data..., dims=1);
u_ave_TO10 = get_u_ave(data2);

maximum(u_ave_TO5-u_ave_TO10)

data = load(@__DIR__()*"/output/paper_data_channel/nomodel/LF_nomodel_channel_tsim100.0.jld2", "fields");
u_fields = data[2:101];
u_ave_NM = get_u_ave(u_fields);

data = load(@__DIR__()*"/output/paper_data_channel/WALE/LF_wale_channel_c0.53_tsim100.0.jld2", "fields");
u_fields = data[2:101];
u_ave_wale = get_u_ave(u_fields);
u_ave_wale_short = get_u_ave(data[2:11]);

data = load(@__DIR__()*"/output/paper_data_channel/smag/LF_smag_channel_c0.13_tsim100.0.jld2", "fields");
u_fields = data[2:101];
u_ave_smag = get_u_ave(u_fields);
u_ave_smag_short = get_u_ave(data[2:11]);

yp = setup.xu[1][2][2:Int(end//2)]*180

using DelimitedFiles
if !ispath(@__DIR__()*"/ref_data_vreman/Chan180_FD2_basic_u.txt")
    println("Downloading reference data for channel flow profiles...")
    using Downloads
    Downloads.download("https://www.vremanresearch.nl/Chan180_FD2_basic_u.txt", @__DIR__()*"/ref_data_vreman/Chan180_FD2_basic_u.txt")
end

data = readdlm(@__DIR__()*"/ref_data_vreman/Chan180_FD2_basic_u.txt", comments=true, comment_char='%')
cols = ["y^+", "U", "rms(u)",  "<u'u'u'>",  "<u'u'u'u'>", "<u'u'v'>", "<u'w'>"]
yp_ref = data[2:end, 1]
u_ave_ref = data[2:end, 2]

#long term online simulations
let
    f = Figure(size=(600,400));
    ax1 = Axis(f[1, 1], xscale = log10)
    scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
    scatter!(ax1, yp, u_ave_NM, color=:red, marker = :xcross, label = "No model")
    scatter!(ax1, yp, u_ave_wale, color=:orange, marker=:diamond, label = "WALE")
    scatter!(ax1, yp, u_ave_smag, color=:purple, marker=:star6, label = "Smag")
    scatter!(ax1, yp, u_ave_TO5, color=:green, marker=:star4, label = "TO LRS h=5")
    ylims!(ax1,0, 19)
    xlims!(ax1, 0.2, 180)
    axislegend(ax1, position = :lt)


    ax2 = Axis(f[1, 2])
    scatter!(ax2, yp_ref, u_ave_ref, color=:blue, label = "Ref")
    scatter!(ax2, yp, u_ave_NM, color=:red, marker=:xcross, label = "No model")
    scatter!(ax2, yp, u_ave_wale, color=:orange, marker=:diamond, label = "WALE")
    scatter!(ax2, yp, u_ave_smag, color=:purple, marker=:star6, label = "Smag")
    scatter!(ax2, yp, u_ave_TO5, color=:green, marker=:star4, label = "TO LRS h=5")
    ylims!(ax2,0, 19)
    xlims!(ax2, 0.0, 180)
    ax1.ylabel=L"\text{mean } v_x"
    ax1.xlabel=L"y^+"
    ax2.xlabel=L"y^+"
    display(f)
    save(figs_folder*"/Channel_flowprofiles_online.pdf", f)
end


# plot velocity profiles for optimized viscosity models over first 10 time units
hf_file = @__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2"
data = load(hf_file, "f").data[1].u[:];
u_fields = data[2:11];
u_ave_HF = get_u_ave(u_fields);

let
f = Figure(size=(600,400));
ax1 = Axis(f[1, 1], xscale = log10)
scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
scatter!(ax1, yp, u_ave_HF, color=:green, marker=:xcross, label = "HF")
scatter!(ax1, yp, u_ave_wale_short, color=:orange, marker=:diamond, label = "WALE")
scatter!(ax1, yp, u_ave_smag_short, color=:purple, marker=:star6, label = "Smag")

ylims!(ax1,0, 19)
xlims!(ax1, 0.2, 180)
axislegend(ax1, position = :lt)


ax2 = Axis(f[1, 2])
scatter!(ax2, yp_ref, u_ave_ref, color=:blue, label = "Ref")
scatter!(ax2, yp, u_ave_HF, color=:green, marker=:xcross, label = "HF")
scatter!(ax2, yp, u_ave_wale_short, color=:orange, marker=:diamond, label = "WALE")
scatter!(ax2, yp, u_ave_smag_short, color=:purple, marker=:star6, label = "Smag")
ylims!(ax2,0, 19)
xlims!(ax2, 0.0, 180)
ax1.ylabel=L"\text{mean } v_x"
ax1.xlabel=L"y^+"
ax2.xlabel=L"y^+"
display(f)
save(figs_folder*"/Channel_flowprofiles_visc_tuned.pdf", f)
end




## Plot filtered initial field
# Domain
    xlims = 0, 4 * pi
    ylims = 0, 2
    zlims = 0, 4 / 3 * pi
    # Grid
    nx = 64 
    ny = 64 
    nz = 32 
    setup = rf_setup(;
        x = (
            range(xlims..., nx + 1),
            range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
            range(zlims..., nz + 1)
        ),
        boundary_conditions = (; u = (
            (PeriodicBC(), PeriodicBC()),
            (DirichletBC(), DirichletBC()),
            (PeriodicBC(), PeriodicBC()),
        )),
        Re = 180.0,
    );

    ustart = Array(load(@__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2")["f"].data[1].u[1]);
    y_ax = setup.xu[1][2]
    x_ax = setup.xu[1][1]


qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10],
        ["Z",11,17],["E", 11, 17]];
to_setup = 
        RikFlow.TO_Setup(; qois, 
        to_mode = :TRACK_REF, 
        ArrayType=Array, 
        setup,
        nstep=10,
        mirror_y = true,);
u_hat = RikFlow.get_u_hat(ustart, setup, to_setup);
w_hat = RikFlow.get_w_hat_from_u_hat(u_hat, to_setup);
qd = RikFlow.compute_filtered_qoi_fields(u_hat, w_hat, to_setup, setup);


for i in 1:size(qois, 1)
    g = Figure(size = (600, 200));
    if i in [5,6]
        axs = Axis(g[1,1][1,1],
        ylabel = "y", xlabel = "x",
        aspect = DataAspect(), )
    else
        axs = Axis(g[1,1][1,1],
            ylabel = "y",
            aspect = DataAspect(), )
    end
    hm = heatmap!(axs,x_ax[1:end-2], y_ax[2:end-1], sum(abs2,real(ifft(qd[i],[1,2,3])),dims = 4)[1:end,1:Int(end/2),5])
    display(g)
    name = figs_folder*"/u_filtered_R$(i).png"
    save(name, g)
    run(`magick $name -trim $name`)
end


