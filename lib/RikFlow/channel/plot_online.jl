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

#data = load(@__DIR__()*"/output/online_mirror/LinReg1/LF_online_channel_to_64_64_32_tsim50.0_repl_1.jld2", "data_train");
data = load(@__DIR__()*"/output/online_mirror_6qoi/LinReg2/LF_online_channel_to_64_64_32_tsim50.0_repl_1.jld2", "data_train");
u_fields = data.fields[30:50];
us = stack(map(x -> x.u, u_fields));
# mean flow profile last 10 snapshots
u_ave = mean(us[:,:,:,1,:], dims=[1,3,4])
u_ave = reshape(u_ave, :)
u_ave = (u_ave[1:end] + u_ave[end:-1:1])/2
u_ave_TO = u_ave[2:33]

data = load(@__DIR__()*"/output/LF_nomodel_channel_to_64_64_32_tsim50.0.jld2", "fields");
u_fields = data[30:50];
us = stack(map(x -> x.u, u_fields));
# mean flow profile last 10 snapshots
u_ave = mean(us[:,:,:,1,:], dims=[1,3,4])
u_ave = reshape(u_ave, :)
u_ave = (u_ave[1:end] + u_ave[end:-1:1])/2
u_ave_NM = u_ave[2:33]

data = load(@__DIR__()*"/output/LF_wale_mirror_channel_to_64_64_32_tsim50.0.jld2", "fields");
u_fields = data[30:50];
us = stack(map(x -> x.u, u_fields));
# mean flow profile last 10 snapshots
u_ave = mean(us[:,:,:,1,:], dims=[1,3,4])
u_ave = reshape(u_ave, :)
u_ave = (u_ave[1:end] + u_ave[end:-1:1])/2
u_ave_wale = u_ave[2:33]

data = load(@__DIR__()*"/output/LF_smag_mirror_channel_to_64_64_32_tsim50.0.jld2", "fields");
u_fields = data[30:50];
us = stack(map(x -> x.u, u_fields));
# mean flow profile last 10 snapshots
u_ave = mean(us[:,:,:,1,:], dims=[1,3,4])
u_ave = reshape(u_ave, :)
u_ave = (u_ave[1:end] + u_ave[end:-1:1])/2
u_ave_smag = u_ave[2:33]

yp = setup.grid.xu[1][2][2:Int(end//2)]*180
lines(yp, u_ave_NM)
lines(yp, u_ave_TO)

using DelimitedFiles
#data = readdlm(@__DIR__()*"/output/LM_Channel_0180_mean_prof.dat", comments=true, comment_char='%')
#cols = ["y/delta", "y^+", "U", "dU/dy", "W", "P"]
data = readdlm(@__DIR__()*"/output/Chan180_FD2_all/Chan180_FD2_basic_u.txt", comments=true, comment_char='%')
cols = ["y^+", "U", "rms(u)",  "<u'u'u'>",  "<u'u'u'u'>", "<u'u'v'>", "<u'w'>"]
yp_ref = data[2:end, 1]
u_ave_ref = data[2:end, 2]

#log plot
f = Figure(size=(800,500))
ax1 = Axis(f[1, 1], xscale = log10)
scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
scatter!(ax1, yp, u_ave_NM, color=:green, label = "No model")
scatter!(ax1, yp, u_ave_TO, color=:red, label = "TO")
scatter!(ax1, yp, u_ave_wale, color=:orange, label = "Wale")
scatter!(ax1, yp, u_ave_smag, color=:purple, label = "Smag")

ylims!(ax1,0, 19)
xlims!(ax1, 0.1, 180)
axislegend(ax1, position = :lt)


ax1 = Axis(f[1, 2])
scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
scatter!(ax1, yp, u_ave_NM, color=:green, label = "No model")
scatter!(ax1, yp, u_ave_TO, color=:red, label = "TO")
scatter!(ax1, yp, u_ave_wale, color=:orange, label = "Wale")
scatter!(ax1, yp, u_ave_smag, color=:purple, label = "Smag")
ylims!(ax1,0, 19)
xlims!(ax1, 0.1, 180)

display(f)


qois = [["Z",0,6],["E", 0, 6],["Z",7,16],["E", 7, 16]];
n_replicas = 5
time_index = 0:0.01:50

data = [load(@__DIR__()*"/output/online/LinReg1/LF_online_channel_to_64_64_32_tsim50.0_repl_$(i).jld2", "data_train")
        for i in 1:n_replicas]
q_rep = map(x -> x.q, data)
hf_data = load(@__DIR__()*"/output/checkpoints/checkpoint_n50000.jld2");
q_ref = stack(hf_data["results"].data[1].qoi_hist)
nomodel_data = load(@__DIR__()*"/output/LF_nomodel_channel_to_64_64_32_tsim50.0.jld2","qoihist");
q_NM = stack(nomodel_data)

let
g = Figure(size = (800, 700))
best, ref, no_model, model= nothing, nothing, nothing, nothing
axs = [Axis(g[i ÷ 2, i%2], 
        title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
    for i in 0:size(q_ref, 1)-1]

for i in 1:size(q_ref, 1)
    for j in 1:n_replicas
            
            model=lines!(axs[i],time_index, q_rep[j][i,:], color = (:blue, 0.3))
        
    end
    xlim_right = min(maximum(size.(q_rep,2)), size(time_index,1))
    ref = lines!(axs[i], time_index, q_ref[i,:], color = :black)
    no_model = lines!(axs[i], time_index, q_NM[i,:], color = (:red, 0.6))
    #ylims!(axs[i],(0, maximum(q_ref[i,:])*2)) 
end
#Label(g[-1, :], text = L"$\sigma_\epsilon =$ %$(linreg_params.tracking_noise[1]), $\eta =$ %$(linreg_params.model_noise_str[1]), hist $=$ %$(linreg_params.hist_len[1]), $\lambda =$ %$(linreg_params.lambda[1])", fontsize = 20)
Legend(g[0,2], [ref, no_model, model], ["Reference", "No model", "Online"], fontsize = 12)
display(g)
end