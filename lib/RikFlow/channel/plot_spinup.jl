using IncompressibleNavierStokes
using RikFlow
using CairoMakie
using JLD2
using LinearAlgebra
using Statistics
using CUDA
using FFTW

# Domain
xlims = 0, 4 * pi
ylims = 0, 2
zlims = 0, 4 / 3 * pi
# Grid
nx = 512 
ny = 512 
nz = 256 
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

u_start = load(@__DIR__()*"/output/HF/u_start_T15_512_512_256.jld2", "u_start");
u_ave = mean(u_start[:,:,:,1], dims=3)
y_ax = setup.grid.xu[1][2]
x_ax = setup.grid.xu[1][1]

let
f = Figure(size = (900, 200));
ax1 = Axis(f[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "y")
heatmap!(ax1,x_ax[1:end-2], y_ax, (u_start[1:end-2,:,1,1]+ u_start[1:end-2,:,2,1])/2)
#contourf!(ax1,x_ax[1:end-2], y_ax, (u_start[1:end-2,:,1,1]+ u_start[1:end-2,:,2,1])/2, levels=20)
display(f)
name = @__DIR__()*"/output/figs/u_start.png"
save(name, f)
run(`magick $name -trim $name`)
end

# plot spectrum
# scales = get_scale_numbers(u_start, setup)
# state = (;u = u_start, t=0., temp=0);
# fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [1,1], slopeoffset = 50, plot_wavelength = false)
# display(fig)
# v = [scales.λ, scales.η, 1/n]
# v_labels = ["λ", "η", "Δx"]
# for i in 1:3
#     text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1e-12*1.2), align = (:left, :bottom), color = :black)
# end
# display(fig)
# save(fig_folder*"/energy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.png", fig)

# plot coarse spectrum
ustart = Array(load(@__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2")["f"].data[1].u[1]);
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

u_ave = mean(ustart[:,:,:,1], dims=3)
y_ax = setup.grid.xu[1][2]
x_ax = setup.grid.xu[1][1]

let
f = Figure(size = (900, 200));
ax1 = Axis(f[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "y")
heatmap!(ax1,x_ax[1:end-2], y_ax, (ustart[1:end-2,:,1,1] + ustart[1:end-2,:,1,1])/2)
#contourf!(ax1,x_ax[1:end-2], y_ax, (ustart[1:end-2,:,1,1] + ustart[1:end-2,:,1,1])/2, levels=20)
display(f)
name = @__DIR__()*"/output/figs/u_start_coarse.png"
save(name, f)
run(`magick $name -trim $name`)
end

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


let
g = Figure(size = (1000, 500));
axs = [Axis(g[i ÷ 2, i%2][1,1],
        #xlabel = "x", ylabel = "y",
        aspect = DataAspect(), )
        #title = L"||R_{[%$(qois[i+1][2]), %$(qois[i+1][3])]} \omega ||")
    for i in 0:size(qois, 1)-1]

for i in 1:size(qois, 1)
    hm = heatmap!(axs[i],x_ax[1:end-2], y_ax[2:end-1], sum(abs2,real(ifft(qd[i],[1,2,3])),dims = 4)[1:end,1:Int(end/2),5])
    #hm = heatmap!(axs[i], real(ifft(qd[i],[1,2,3]))[:,4,:,1])
    #Colorbar(g[(i-1) ÷ 2, (i-1)%2][1,2],hm)
end
display(g)
end

let
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
    #hm = heatmap!(axs[i], real(ifft(qd[i],[1,2,3]))[:,4,:,1])
    #Colorbar(g[1,1][1,2],hm)
    display(g)
    name = @__DIR__()*"/output/figs/u_filtered_R$(i).png"
    save(name, g)
    run(`magick $name -trim $name`)
end

end