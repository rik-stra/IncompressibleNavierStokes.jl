using IncompressibleNavierStokes
using CairoMakie
using JLD2
using LinearAlgebra
using Statistics
using CUDA

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
    Re = 180.0,
)
setup = Setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    kwargs...,
);
y_ax = setup.grid.xu[1][2]
x_ax = setup.grid.xu[1][1]

#u = Array(load(@__DIR__()*"/output/HF_channel_6qoi_mirror_2framerate_new_128_128_64_to_64_64_32_tsim20.0.jld2")["f"].data[1].u[:]);
u = Array(load(@__DIR__()*"/output/HF/HF_channel_6qoinew_mirror_2framerate_512_512_256_to_64_64_32_tsim15.0.jld2")["f"].data[1].u[:]);
#u = Array(load(@__DIR__()*"/output/checkpoint_n10000.jld2")["results"].data[1].u[:]);

u_ave_center = mean(stack(u[2:11])[1:end-2, Int(end/2):Int(end/2)+1, 1:end-2, 1, :])
u_ave = mean(stack(u[2:11])[1:end-2, 2:end-1, 1:end-2, 1, :])


for t in 1:size(u,1)
    f = Figure(size = (900, 200));
    ax1 = Axis(f[1, 1], aspect = DataAspect())
    heatmap!(ax1,x_ax, y_ax,u[t][:,:,1,1])
    display(f)
end
#save(@__DIR__()*"/output/figs/u_start_coarse.png", f)

u_fields = u[1:10];
us = stack(u_fields);
# mean flow profile last 10 snapshots
u_ave = mean(us[:,:,:,1,:], dims=[1,3,4])
u_ave = reshape(u_ave, :)
u_ave = (u_ave[1:end] + u_ave[end:-1:1])/2
u_ave = u_ave[2:33]

yp = setup.grid.xu[1][2][2:Int(end//2)]*180

using DelimitedFiles
#data = readdlm(@__DIR__()*"/output/LM_Channel_0180_mean_prof.dat", comments=true, comment_char='%')
#cols = ["y/delta", "y^+", "U", "dU/dy", "W", "P"]
data = readdlm(@__DIR__()*"/output/Chan180_FD2_all/Chan180_FD2_basic_u.txt", comments=true, comment_char='%')
cols = ["y^+", "U", "rms(u)",  "<u'u'u'>",  "<u'u'u'u'>", "<u'u'v'>", "<u'w'>"]
yp_ref = data[2:end, 1]
u_ave_ref = data[2:end, 2]

#log plot
f = Figure(size=(800,500));
ax1 = Axis(f[1, 1], xscale = log10)
scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
scatter!(ax1, yp, u_ave, color=:red, label = "filtered HF")

ylims!(ax1,0, 19)
xlims!(ax1, 0.1, 180)
axislegend(ax1, position = :lt)


ax1 = Axis(f[1, 2])
scatter!(ax1, yp_ref, u_ave_ref, color=:blue, label = "Ref")
scatter!(ax1, yp, u_ave, color=:red, label = "filtered HF")
ylims!(ax1,0, 19)
xlims!(ax1, 0.1, 180)

display(f)



let
q = stack(load(@__DIR__()*"/output/HF/HF_channel_6qoinew_mirror_2framerate_512_512_256_to_64_64_32_tsim15.0.jld2")["f"].data[1].qoi_hist)
qois = [["Z",0,3],["E", 0, 3],["Z",4,12],["E", 4, 12],
        ["Z",13,17],["E", 13, 17]];
time_index = 0:0.001:15
#let
g = Figure(size = (800, 700));

axs = [Axis(g[i ÷ 2, i%2], 
        title = "$(qois[i+1][1])_[$(qois[i+1][2]), $(qois[i+1][3])]")
    for i in 0:size(q, 1)-1]

for i in 1:size(q, 1)
    
            
    lines!(axs[i],time_index, q[i,:])

    #ylims!(axs[i],(0, maximum(q_ref[i,:])*2)) 
end
#Label(g[-1, :], text = L"$\sigma_\epsilon =$ %$(linreg_params.tracking_noise[1]), $\eta =$ %$(linreg_params.model_noise_str[1]), hist $=$ %$(linreg_params.hist_len[1]), $\lambda =$ %$(linreg_params.lambda[1])", fontsize = 20)

display(g)
end

