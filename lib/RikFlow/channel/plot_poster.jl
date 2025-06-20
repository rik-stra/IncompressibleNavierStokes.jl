using IncompressibleNavierStokes
using CairoMakie
using JLD2
using LinearAlgebra
using Statistics

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
    Re = 180,
)
setup = Setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
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

function get_rms_u(u_fields, dim = 1)
    if isa(u_fields[1], Array)
        us = stack(u_fields)
    else
        us = stack(map(x -> x.u, u_fields));
    end
    u_ave = mean(us[1:end-2,:,1:end-2,dim,:], dims=[1,3,4])
    u_prime = us[1:end-2,:,1:end-2,dim,:] .- u_ave
    u_prime_squared = u_prime.^2
    u_rms = mean(u_prime_squared, dims=[1,3,4])
    u_rms = reshape(u_rms, :)
    u_rms = (u_rms[1:end] + u_rms[end:-1:1])/2
    u_rms = sqrt.(u_rms)
    return u_rms[2:33]
end



#data = load(@__DIR__()*"/output/online_mirror/LinReg1/LF_online_channel_to_64_64_32_tsim50.0_repl_1.jld2", "data_train");
data = [load(@__DIR__()*"/output/online_TOpaper/LinReg11/LF_online_channel_to_64_64_32_tsim100.0_repl_$i.jld2", "data").fields[2:end] for i=1:1];
data2 = cat(data..., dims=1);
u_ave_TO5 = get_u_ave(data2);
u_rms_TO5 = get_rms_u(data2,1);
v_rms_TO5 = get_rms_u(data2,2);
w_rms_TO5 = get_rms_u(data2,3);

data = [load(@__DIR__()*"/output/online_TOpaper/LinReg13/LF_online_channel_to_64_64_32_tsim100.0_repl_$i.jld2", "data").fields[2:end] for i=1:5];
data2 = cat(data..., dims=1);
u_ave_TO10 = get_u_ave(data2);

maximum(u_ave_TO5-u_ave_TO10)

data = load(@__DIR__()*"/output/LF_nomodel_mirror_channel_to_tsim100.0.jld2", "fields");
u_fields = data[2:101];
u_ave_NM = get_u_ave(u_fields);
u_rms_NM = get_rms_u(u_fields,1);
v_rms_NM = get_rms_u(u_fields,2);
w_rms_NM = get_rms_u(u_fields,3);

data = load(@__DIR__()*"/output/WALE/LF_wale_mirror_channel_to_0.53_tsim100.0.jld2", "fields");
u_fields = data[2:101];
u_ave_wale = get_u_ave(u_fields);
u_rms_wale = get_rms_u(u_fields,1);
v_rms_wale = get_rms_u(u_fields,2);
w_rms_wale = get_rms_u(u_fields,3);
u_ave_wale_short = get_u_ave(data[2:11]);

data = load(@__DIR__()*"/output/smag/LF_smag_mirror_channel_to_0.13_tsim100.0.jld2", "fields");
u_fields = data[2:101];
u_ave_smag = get_u_ave(u_fields);
u_rms_smag = get_rms_u(u_fields,1);
v_rms_smag = get_rms_u(u_fields,2);
w_rms_smag = get_rms_u(u_fields,3);
u_ave_smag_short = get_u_ave(data[2:11]);

yp = setup.grid.xu[1][2][2:Int(end//2)]*180
#lines(yp, u_ave_NM)
#lines(yp, u_ave_TO)

using DelimitedFiles
#data = readdlm(@__DIR__()*"/output/LM_Channel_0180_mean_prof.dat", comments=true, comment_char='%')
#cols = ["y/delta", "y^+", "U", "dU/dy", "W", "P"]
data = readdlm(@__DIR__()*"/output/Chan180_FD2_all/Chan180_FD2_basic_u.txt", comments=true, comment_char='%')
cols = ["y^+", "U", "rms(u)",  "<u'u'u'>",  "<u'u'u'u'>", "<u'u'v'>", "<u'w'>"]
yp_ref = data[2:end, 1]
u_ave_ref = data[2:end, 2]
rms_u_ref = data[2:end, 3]





#log plot
let
    f = Figure(size=(300,200));
    ax1 = Axis(f[1, 1])
    scatter!(ax1, yp_ref, u_ave_ref, color=(:black,0.4), label = "Ref")
    scatter!(ax1, yp, u_ave_wale, color=:purple, label = "Wale")
    scatter!(ax1, yp, u_ave_smag, color=:orange, label = "Smag")
    scatter!(ax1, yp, u_ave_TO5, color=:teal, label = "TO")
    ylims!(ax1,0, 19)
    xlims!(ax1, 0.2, 180)
    #axislegend(ax1, position = :lt)

    ax1.ylabel=L"\text{mean } v_x"
    ax1.xlabel=L"y^+"
    display(f)
    save(@__DIR__()*"/output/figs_poster/Channel_flowprofiles_online.pdf", f)
end





    qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10], ["Z",11,17],["E", 11, 17]];
    n_replicas = 5
    time_index = 0:0.05:100

    data = [load(@__DIR__()*"/output/online_TOpaper/LinReg11/LF_online_channel_to_64_64_32_tsim100.0_repl_$(i).jld2", "data")
            for i in 1:n_replicas];
    q_rep = map(x -> x.q ./2, data)
    hf_data = load(@__DIR__()*"/output/HF/HF_channel_6qoinew_mirror_2framerate_512_512_256_to_64_64_32_tsim15.0.jld2");
    q_ref = stack(hf_data["f"].data[1].qoi_hist)./2
    wale_data = load(@__DIR__()*"/output/WALE/LF_wale_mirror_channel_to_0.53_tsim100.0.jld2","qoihist");
    q_wale = stack(wale_data)./2
    nomodel_data = load(@__DIR__()*"/output/LF_nomodel_mirror_channel_to_tsim100.0.jld2","qoihist");
    q_no_model = stack(nomodel_data)./2
    smag_data = load(@__DIR__()*"/output/smag/LF_smag_mirror_channel_to_0.13_tsim100.0.jld2","qoihist");
    q_smag = stack(smag_data)./2

let
    g = Figure(size = (600, 200))
    ref, model= nothing, nothing
    ga = g[1, 1] = GridLayout()
    axs = [Axis(ga[i,1])
        for i in 0:size(q_ref, 1)-1]
    colgap!(ga, 0)
    rowgap!(ga, 0)  
    for i in 1:size(q_ref, 1)
        for j in 1:n_replicas
                model=lines!(axs[i],time_index, q_rep[j][i,1:10:end], color = (:teal, 0.25)) 
        end
        
        ref = lines!(axs[i], time_index[1:201], q_ref[i,1:50:10001], color = :black)
        wale=lines!(axs[i],time_index, q_wale[i,1:10:end], color = (:purple))
        smag=lines!(axs[i],time_index, q_smag[i,1:10:end], color = (:orange))
         
        hidedecorations!(axs[i])
        if i == 6
            axs[i].xlabel="t"
            makevisible = true
            axs[i].xgridvisible = makevisible
            axs[i].xticksvisible= makevisible
            axs[i].xticklabelsvisible = makevisible
            axs[i].xlabelvisible = makevisible
        end
        hidespines!(axs[i])
        xlims!(axs[i], 0, 100)
        vlines!(axs[i], [0, 10, 100], color=:black, linewidth=1)
        #ref = lines!(axs[i], time_index[1:3001], q_ref[i,1:5:15001], color = :black)
        
    end
    #Legend(g[0,2], [ref, model], ["HF", "TO LRS"], fontsize = 12)
    #axislegend(axs[6],[ref, model],["HF", "TO LRS"], position=:rc)
    
    display(g)
    save(@__DIR__()*"/output/figs_poster/Channel_TO_LRS5_q_trajectories.pdf", g)
end
    
   



