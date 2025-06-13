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
    f = Figure(size=(600,400));
    ax1 = Axis(f[1, 1], xscale = log10)
    scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
    scatter!(ax1, yp, u_ave_NM, color=:red, label = "No model")
    scatter!(ax1, yp, u_ave_wale, color=:orange, label = "Wale")
    scatter!(ax1, yp, u_ave_smag, color=:purple, label = "Smag")
    scatter!(ax1, yp, u_ave_TO5, color=:green, label = "TO LRS h=5")
    ylims!(ax1,0, 19)
    xlims!(ax1, 0.2, 180)
    axislegend(ax1, position = :lt)


    ax2 = Axis(f[1, 2])
    scatter!(ax2, yp_ref, u_ave_ref, color=:blue, label = "Ref")
    scatter!(ax2, yp, u_ave_NM, color=:red, label = "No model")
    scatter!(ax2, yp, u_ave_wale, color=:orange, label = "Wale")
    scatter!(ax2, yp, u_ave_smag, color=:purple, label = "Smag")
    scatter!(ax2, yp, u_ave_TO5, color=:green, label = "TO LRS h=5")
    ylims!(ax2,0, 19)
    xlims!(ax2, 0.0, 180)
    ax1.ylabel=L"\text{mean } v_x"
    ax1.xlabel=L"y^+"
    ax2.xlabel=L"y^+"
    display(f)
    save(@__DIR__()*"/output/figs/Channel_flowprofiles_online.pdf", f)
end

hf_file = @__DIR__()*"/output/HF/HF_channel_6qoinew_mirror_2framerate_512_512_256_to_64_64_32_tsim15.0.jld2"
data = load(hf_file, "f").data[1].u[:];
u_fields = data[2:11];
u_ave_HF = get_u_ave(u_fields);
u_rms_HF = get_rms_u(u_fields,1);
v_rms_HF = get_rms_u(u_fields,2);
w_rms_HF = get_rms_u(u_fields,3);

let
f = Figure(size=(600,400));
ax1 = Axis(f[1, 1], xscale = log10)
scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
scatter!(ax1, yp, u_ave_HF, color=:green, label = "HF")
scatter!(ax1, yp, u_ave_wale_short, color=:orange, label = "Wale")
scatter!(ax1, yp, u_ave_smag_short, color=:purple, label = "Smag")

ylims!(ax1,0, 19)
xlims!(ax1, 0.2, 180)
axislegend(ax1, position = :lt)


ax2 = Axis(f[1, 2])
scatter!(ax2, yp_ref, u_ave_ref, color=:blue, label = "Ref")
#scatter!(ax2, yp, u_ave_NM, color=:green, label = "No model")
scatter!(ax2, yp, u_ave_HF, color=:green, label = "HF")
scatter!(ax2, yp, u_ave_wale_short, color=:orange, label = "Wale")
scatter!(ax2, yp, u_ave_smag_short, color=:purple, label = "Smag")
ylims!(ax2,0, 19)
xlims!(ax2, 0.0, 180)
ax1.ylabel=L"\text{mean } v_x"
ax1.xlabel=L"y^+"
ax2.xlabel=L"y^+"
display(f)
save(@__DIR__()*"/output/figs/Channel_flowprofiles_visc_tuned.pdf", f)
end

data = readdlm(@__DIR__()*"/output/Chan180_FD2_all/Chan180_FD2_basic_v.txt", comments=true, comment_char='%')
rms_v_ref = data[2:end, 3]
data = readdlm(@__DIR__()*"/output/Chan180_FD2_all/Chan180_FD2_basic_w.txt", comments=true, comment_char='%')
rms_w_ref = data[2:end, 3];


let # plot rms u
    f = Figure(size=(600,400));
    ax1 = Axis(f[1, 1])
    scatter!(ax1, yp_ref, rms_u_ref, color=:blue, label = "Ref")
    scatter!(ax1, yp, u_rms_NM, color=:green, label = "No model")
    scatter!(ax1, yp, u_rms_wale, color=:orange, label = "Wale")
    scatter!(ax1, yp, u_rms_smag, color=:purple, label = "Smag")
    scatter!(ax1, yp, u_rms_TO5, color=:red, label = "TO LRS h=5")
    scatter!(ax1, yp, u_rms_HF, color=:black, label = "HF")
    # ylims!(ax1,0, 19)
    xlims!(ax1, 0.2, 180)
    axislegend(ax1, position = :rb)

    ax1.ylabel=L"\text{rms } v_x"
    ax1.xlabel=L"y^+"
    display(f)
end
let # plot rms v
    f = Figure(size=(600,400));
    ax1 = Axis(f[1, 1])
    scatter!(ax1, yp_ref, rms_v_ref, color=:blue, label = "Ref")
    scatter!(ax1, yp, v_rms_NM, color=:green, label = "No model")
    scatter!(ax1, yp, v_rms_wale, color=:orange, label = "Wale")
    scatter!(ax1, yp, v_rms_smag, color=:purple, label = "Smag")
    scatter!(ax1, yp, v_rms_TO5, color=:red, label = "TO LRS h=5")
    scatter!(ax1, yp, v_rms_HF, color=:black, label = "HF")
    # ylims!(ax1,0, 19)
    xlims!(ax1, 0.2, 180)
    axislegend(ax1, position = :rb)

    ax1.ylabel=L"\text{rms } v_x"
    ax1.xlabel=L"y^+"
    display(f)
end
let # plot rms w
    f = Figure(size=(600,400));
    ax1 = Axis(f[1, 1])
    scatter!(ax1, yp_ref, rms_w_ref, color=:blue, label = "Ref")
    scatter!(ax1, yp, w_rms_NM, color=:green, label = "No model")
    scatter!(ax1, yp, w_rms_wale, color=:orange, label = "Wale")
    scatter!(ax1, yp, w_rms_smag, color=:purple, label = "Smag")
    scatter!(ax1, yp, w_rms_TO5, color=:red, label = "TO LRS h=5")
    scatter!(ax1, yp, w_rms_HF, color=:black, label = "HF")
    # ylims!(ax1,0, 19)
    xlims!(ax1, 0.2, 180)
    axislegend(ax1, position = :rb)

    ax1.ylabel=L"\text{rms } v_x"
    ax1.xlabel=L"y^+"
    display(f)
end



for lr in ["LinReg11", "LinReg13"]
    

    qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10], ["Z",11,17],["E", 11, 17]];
    n_replicas = 5
    time_index = 0:0.005:100

    data = [load(@__DIR__()*"/output/online_TOpaper/$(lr)/LF_online_channel_to_64_64_32_tsim100.0_repl_$(i).jld2", "data")
            for i in 1:n_replicas];
    q_rep = map(x -> x.q ./2, data)
    hf_data = load(@__DIR__()*"/output/HF/HF_channel_6qoinew_mirror_2framerate_512_512_256_to_64_64_32_tsim15.0.jld2");
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
        #ref = lines!(axs[i], time_index[1:3001], q_ref[i,1:5:15001], color = :black)
        #ylims!(axs[i],(0, maximum(q_ref[i,:])*2)) 
    end
    #Legend(g[0,2], [ref, model], ["HF", "TO LRS"], fontsize = 12)
    axislegend(axs[6],[ref, model],["HF", "TO LRS"], position=:rc)
    axs[5].xlabel="t"
    axs[6].xlabel="t"
    display(g)
    if lr == "LinReg11"
        save(@__DIR__()*"/output/figs/Channel_TO_LRS5_q_trajectories.pdf", g)
    end
    if lr == "LinReg13"
        save(@__DIR__()*"/output/figs/Channel_TO_LRS10_q_trajectories.pdf", g)
    end

end



# data = [load(@__DIR__()*"/output/online_TOnew/LinReg6/LF_online_channel_to_64_64_32_tsim100.0_repl_$(i).jld2", "data")
#         for i in 1:n_replicas]
# q_rep = map(x -> x.q, data)
let
    hf_data = load(@__DIR__()*"/output/HF/HF_channel_6qoinew_mirror_2framerate_512_512_256_to_64_64_32_tsim15.0.jld2");
    q_ref = stack(hf_data["f"].data[1].qoi_hist)./2
    wale_data = load(@__DIR__()*"/output/WALE/LF_wale_mirror_channel_to_0.53_tsim100.0.jld2","qoihist");
    q_wale = stack(wale_data)./2
    nomodel_data = load(@__DIR__()*"/output/LF_nomodel_mirror_channel_to_tsim100.0.jld2","qoihist");
    q_no_model = stack(nomodel_data)./2
    smag_data = load(@__DIR__()*"/output/smag/LF_smag_mirror_channel_to_0.13_tsim100.0.jld2","qoihist");
    q_smag = stack(smag_data)./2



    qois = [["Z",0,3],["E", 0, 3],["Z",4,10],["E", 4, 10], ["Z",11,17],["E", 11, 17]];
    time_index = 0:0.005:100


    
    g = Figure(size = (700, 600))
    smag, ref, no_model, wale= nothing, nothing, nothing, nothing
    axs = [Axis(g[i ÷ 2, i%2], 
            title = L"%$(qois[i+1][1])_{[%$(qois[i+1][2]), %$(qois[i+1][3])]}")
        for i in 0:size(q_ref, 1)-1]

    for i in 1:size(q_ref, 1)
        no_model = lines!(axs[i], time_index, q_no_model[i,:], color = (:red))
        wale=lines!(axs[i],time_index, q_wale[i,:], color = (:orange))
        smag=lines!(axs[i],time_index, q_smag[i,:], color = (:purple))
        ref = lines!(axs[i], time_index[1:2001], q_ref[i,1:5:10001], color = :black)
        #xlim_right = min(maximum(size.(q_rep,2)), size(time_index,1))
        
        #no_model = lines!(axs[i], time_index, q_NM[i,:], color = (:red, 0.6))
        #ylims!(axs[i],(0, maximum(q_ref[i,:])*2)) 
    end
    #Label(g[-1, :], text = L"$\sigma_\epsilon =$ %$(linreg_params.tracking_noise[1]), $\eta =$ %$(linreg_params.model_noise_str[1]), hist $=$ %$(linreg_params.hist_len[1]), $\lambda =$ %$(linreg_params.lambda[1])", fontsize = 20)
    #Legend(g[0,2], [ref, no_model, wale, smag], ["HF", "No model" ,"WALE", "Smag"], fontsize = 12)
    axislegend(axs[6],[ref, no_model, wale, smag], ["HF", "No model" ,"WALE", "Smag"], position=:rc)
    axs[5].xlabel="t"
    axs[6].xlabel="t"
    display(g)
    save(@__DIR__()*"/output/figs/Channel_eddyvisc_q_trajectories.pdf", g)

end