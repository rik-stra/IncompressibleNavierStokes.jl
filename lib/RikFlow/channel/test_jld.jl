using IncompressibleNavierStokes
using RikFlow
using CairoMakie
using FFTW
using JLD2

x = 0:0.2:1.8
size(x)
fx = x.^5
lines(x,fx)
x_hat = fft(fx)
k = Array(fftfreq(length(x), 0.2))
k[6]=0
dx_hat = 2 .*pi .*k .* im .*x_hat
dx = ifft(dx_hat)
lines(x,real(ifft(x_hat)))
lines(x,real(dx))
lines(x,imag(dx))
maximum(imag(dx))

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

function icfunc(dim, x, y, z)
    #ux = (y*(y-2))*(sin(10*y)+sin(3*z))
    ux = sin(14*x)
    uy = sin(14*x)
    uz = sin(14*x)
    (dim == 1) * ux + (dim == 2) * uy + (dim == 3) * uz
end

psolver = psolver_transform(setup);
ustart = velocityfield(setup, icfunc, psolver=psolver, doproject=false);

ustart = load(@__DIR__()*"/output/HF/HF_channel_6qoi_mirror_2framerate_T15_T30_512_512_256_to_64_64_32.jld2", "u")[1];

heatmap(ustart[:,:,6,1])

ustart[3,:,4,1]

qois = [["Z",0,3],["E", 0, 3],["Z",4,12],["E", 4, 12],
        ["Z",13,17],["E", 13, 17]];
to_setup = 
        RikFlow.TO_Setup(; qois, 
        to_mode = :TRACK_REF, 
        ArrayType=Array, 
        setup,
        nstep=10,
        mirror_y = true,);
u_hat = RikFlow.get_u_hat(ustart, setup, to_setup);
u = ifft(u_hat, [1,2,3]);
maximum(imag(u))
heatmap(real(u[:,:,6,1]))
heatmap(imag(u[:,:,6,1]))

w_hat = RikFlow.get_w_hat_from_u_hat(u_hat, to_setup);
w = ifft(w_hat, [1,2,3]);
maximum(imag(w))
maximum(real(w))
heatmap(real(w[:,:,6,2]))
heatmap(imag(w[:,:,6,2]))

q = RikFlow.compute_QoI(u_hat, w_hat, to_setup, setup);

# 6-element Vector{Float64}:
#  388835.5750799031
#   27028.799579250128
#   90038.87852460732
#      51.75628997695312
#    2115.478027650952
#       0.14278180056930548