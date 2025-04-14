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
nx = 256 
ny = 256 
nz = 128 
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

u_start = load(@__DIR__()*"/output/u_start_256_256_128_tspin10.0.jld2", "u_start");
u_ave = mean(u_start[:,:,:,1], dims=3)
heatmap(u_start[:,:,20,1])

# mean flow profile
u_ave = mean(u_start[:,:,:,1], dims=[1,3])
u_ave = reshape(u_ave, :)
u_ave = (u_ave[1:128] + u_ave[129:256][end:-1:1])/2

yp = setup.grid.xu[1][2][2:Int(end//2)]*180
lines(yp, u_ave)

using DelimitedFiles
data = readdlm(@__DIR__()*"/output/LM_Channel_0180_mean_prof.dat", comments=true, comment_char='%')
cols = ["y/delta", "y^+", "U", "dU/dy", "W", "P"]
yp_ref = data[2:end, 2]
u_ave_ref = data[2:end, 3]

#log plot
f = Figure()
ax1 = Axis(f[1, 1], xscale = log10)
lines!(ax1, yp_ref, u_ave_ref, color=:blue, linewidth=2)
lines!(ax1, yp, u_ave, color=:red)
ylims!(ax1,0, 19)
xlims!(ax1, 0.1, 180)

ax1 = Axis(f[1, 2])
lines!(ax1, yp_ref, u_ave_ref, color=:blue, linewidth=2)
lines!(ax1, yp, u_ave, color=:red)
ylims!(ax1,0, 19)
xlims!(ax1, 0.1, 180)
display(f)

bulk_mean_velocity = mean(u_ave[1:128]*2)
Rem = bulk_mean_velocity*180

