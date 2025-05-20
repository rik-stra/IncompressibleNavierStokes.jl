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

u_start = load(@__DIR__()*"/output/u_start_512_512_256_tspin10.0.jld2", "u_start");
u_ave = mean(u_start[:,:,:,1], dims=3)
y_ax = setup.grid.xu[1][2]
x_ax = setup.grid.xu[1][1]

f = Figure(size = (900, 200))
ax1 = Axis(f[1, 1], aspect = DataAspect())
heatmap!(ax1,x_ax, y_ax,u_start[:,:,20,1])
display(f)
save(@__DIR__()*"/output/figs/u_start.png", f)

# plot spectrum
scales = get_scale_numbers(u_start, setup)
state = (;u = u_start, t=0., temp=0);
fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [1,1], slopeoffset = 50, plot_wavelength = false)
display(fig)
v = [scales.λ, scales.η, 1/n]
v_labels = ["λ", "η", "Δx"]
for i in 1:3
    text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1e-12*1.2), align = (:left, :bottom), color = :black)
end
display(fig)
save(fig_folder*"/energy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.png", fig)

# plot coarse spectrum
ustart = Array(load(@__DIR__()*"/output/checkpoints/checkpoint_n50000.jld2")["results"].data[1].u[1]);
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

f = Figure(size = (900, 200))
ax1 = Axis(f[1, 1], aspect = DataAspect())
heatmap!(ax1,x_ax, y_ax,ustart[:,:,20,1])
display(f)
save(@__DIR__()*"/output/figs/u_start_coarse.png", f)

# compute energy/enstrophy
ArrayType = CuArray
qois = [["Z",0,100],["E", 0, 100]];
TO_setup = RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = setup,
    mirror_y = true,);

u_hat,u = RikFlow.get_u_hat(ustart, setup, TO_setup);
w_hat = RikFlow.get_w_hat_from_u_hat(ArrayType(u_hat), TO_setup);
E = RikFlow.compute_QoI(ArrayType(u_hat), w_hat, TO_setup, setup)
w = real(ifft(w_hat, [1,2,3]));
z = sum(w.*w, dims=4)
heatmap(Array(z[:,1:Int(end//2),5,1]))

TO_setup = RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = setup,
    mirror_y = false,);

u_hat,u = RikFlow.get_u_hat(ustart, setup, TO_setup);
w_hat = RikFlow.get_w_hat_from_u_hat(ArrayType(u_hat), TO_setup);
E = RikFlow.compute_QoI(ArrayType(u_hat), w_hat, TO_setup, setup)
w = real(ifft(w_hat, [1,2,3]));
z = sum(w.*w, dims=4)
heatmap(Array(z[:,:,5,1]))

# plot spectrum
#scales = get_scale_numbers(u_start, setup)
state = (;u = ustart, t=0., temp=0);
fig = energy_spectrum_plot(state; setup, npoint = 100, sloperange = [1,1], slopeoffset = 50, plot_wavelength = false)
display(fig)
v = [scales.λ, scales.η, 1/n]
v_labels = ["λ", "η", "Δx"]
for i in 1:3
    text!(fig[1,1], v_labels[i], position = (v[i]*0.96,1e-12*1.2), align = (:left, :bottom), color = :black)
end
display(fig)
save(fig_folder*"/energy_spectrum_afterspinup_512_Re2000.0_freeze_10_tsim4.png", fig)












# mean flow profile
u_ave = mean(u_start[:,:,:,1], dims=[1,3])
u_ave = reshape(u_ave, :)
u_ave = (u_ave[1:128] + u_ave[129:256][end:-1:1])/2

yp = setup.grid.xu[1][2][2:Int(end//2)]*180
f = hlines([18.42, 18.25], color=:red) # centerline values from Vreman
lines!(yp, u_ave)

display(f)

using DelimitedFiles
data_MKM = readdlm(@__DIR__()*"/output/LM_Channel_0180_mean_prof.dat", comments=true, comment_char='%')
cols = ["y/delta", "y^+", "U", "dU/dy", "W", "P"]
yp_ref_MKM = data_MKM[2:end, 2]
u_ave_ref_MKM = data_MKM[2:end, 3]

data_Vre = readdlm(@__DIR__()*"/output/Chan180_FD2_all/Chan180_FD2_basic_u.txt", comments=true, comment_char='%')
cols = ["y^+", "U", "rms(u)",  "<u'u'u'>",  "<u'u'u'u'>", "<u'u'v'>", "<u'w'>"]
yp_ref_Vre = data_Vre[2:end, 1]
u_ave_ref_Vre = data_Vre[2:end, 2]

#log plot
f = Figure()
ax1 = Axis(f[1, 1], xscale = log10)
lines!(ax1, yp_ref_MKM, u_ave_ref_MKM, color=:blue, linewidth=2)
lines!(ax1, yp_ref_Vre, u_ave_ref_Vre, color=:green, linewidth=2)
lines!(ax1, yp, u_ave, color=:red)
ylims!(ax1,0, 19)
xlims!(ax1, 0.1, 180)

ax1 = Axis(f[1, 2])
lines!(ax1, yp_ref_MKM, u_ave_ref_MKM, color=:blue, linewidth=2)
lines!(ax1, yp_ref_Vre, u_ave_ref_Vre, color=:green, linewidth=2)
lines!(ax1, yp, u_ave, color=:red)
ylims!(ax1,0, 19)
xlims!(ax1, 0.1, 180)
display(f)

bulk_mean_velocity = mean(u_ave[1:128]*2)
Rem = bulk_mean_velocity*180

### plot HF ref ###
hf_data = load(@__DIR__()*"/output/checkpoints/checkpoint_n50000.jld2");
keys(hf_data)
hf_u = hf_data["u_cpu"];
heatmap(hf_u[:,:,1,1])

lf_u = hf_data["results"].data[1].u[end]
heatmap(lf_u[:,:,1,1])

q_ref = stack(hf_data["results"].data[1].qoi_hist)
#plot the time series in q_ref in 4 different axes
f = Figure()
ax1 = Axis(f[1, 1])
ax2 = Axis(f[1, 2])
ax3 = Axis(f[2, 1])
ax4 = Axis(f[2, 2])
lines!(ax1, q_ref[1,:], color=:blue)
lines!(ax1, q[1,:], color=:red)
lines!(ax2, q_ref[2,:], color=:blue)
lines!(ax2, q[2,:], color=:red)
lines!(ax3, q_ref[3,:], color=:blue)
lines!(ax3, q[3,:], color=:red)
lines!(ax4, q_ref[4,:], color=:blue)
lines!(ax4, q[4,:], color=:red)
display(f)

heatmap(outputs.fields[7].u[:,:,1,1])

keys(outputs.fields[end])