using IncompressibleNavierStokes
using RikFlow
using CairoMakie
using FFTW

# create square grid
n = 32
setup = Setup(;
    x = (
        range(0,2, n + 1),
        range(0,1, 64 + 1), # tanh_grid(ylims..., ny + 1),
        range(0,1, n + 1)
    ),
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re = 180.,
);

qois = [["Z",0,17],["E", 0, 17]];
ArrayType = Array
TO_setup = RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = setup,);

function icfunc(dim, x, y, z)
    ux = sinpi(x)
    uy = sinpi(2*x)
    uz = sinpi(3*z)
    (dim == 1) * ux + (dim == 2) * uy + (dim == 3) * uz
end

ustart = velocityfield(setup, icfunc);

u_hat = RikFlow.get_u_hat(ustart, setup);
w_hat = RikFlow.get_w_hat_from_u_hat(u_hat, TO_setup);
E_hat = RikFlow.compute_QoI(u_hat, w_hat, TO_setup, setup)

E_hat[2]
E = total_kinetic_energy(ustart, setup, interpolate_first = true)
E = total_kinetic_energy(ustart, setup, interpolate_first = false)
E_hat[1]
Z = IncompressibleNavierStokes.total_enstropy(ustart, setup)

# channel setup
# Domain
xlims = 0, 4 * pi
ylims = 0, 2.
zlims = 0, 4 / 3 * pi

# Grid
nx = 16 
ny = 64 
nz = 16

setup = Setup(;
    x = (
        range(xlims..., nx + 1),
        range(ylims..., ny + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz + 1)
    ),
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re = 180.,
);


Re_tau = 180.
Re_m = 2800.
Re_ratio = Re_m / Re_tau

ustartfunc = let
    Lx = xlims[2] - xlims[1]
    Ly = ylims[2] - ylims[1]
    Lz = zlims[2] - zlims[1]
    C = 9 / 8 * Re_ratio
    E = 1 / 10 * Re_ratio # 10% of average mean velocity
    function icfunc(dim, x, y, z)
        ux =
            C * (1 - (y - Ly / 2)^8) +
            E * Lx / 2 * sinpi(y) * cospi(4 * x / Lx) * sinpi(2 * z / Lz)
        uy = -E * (1 - cospi(y)) * sinpi(4 * x / Lx) * sinpi(2 * z / Lz)
        uz = -E * Lz / 2 * sinpi(4 * x / Lx) * sinpi(y) * cospi(2 * z / Lz)
        (dim == 1) * ux + (dim == 2) * uy + (dim == 3) * uz
    end
end

ustart = velocityfield(setup, ustartfunc);
qois = [["Z",0,100],["E", 0, 17]];
ArrayType = Array
TO_setup = RikFlow.TO_Setup(; qois, 
    to_mode = :CREATE_REF, 
    ArrayType, 
    setup = setup,);

u_hat = RikFlow.get_u_hat(ustart, setup);
w_hat = RikFlow.get_w_hat_from_u_hat(u_hat, TO_setup);
E_hat = RikFlow.compute_QoI(u_hat, w_hat, TO_setup, setup)

w_tilde = real(ifft(w_hat, [1,2,3]));
w = IncompressibleNavierStokes.vorticity(ustart, setup);
size(w[2:33,2:33,2:17,:]), size(w_tilde)
heatmap(sum(w.*w,dims=4)[:,:,10])
heatmap(sum(w_tilde .* w_tilde, dims=4)[:,:,11])

sum(w.*w,dims=4)[4,:,10]
sum(w_tilde .* w_tilde, dims=4)[4,:,11]

u_tilde = zeros(size(ustart));
u_out = real(ifft(u_hat, [1,2,3]));
for a in 1:3
    u_tilde[RikFlow.select_physical_fourier_points(a, setup),a] = u_out[:,:,:,a]
end
#IncompressibleNavierStokes.apply_bc_u!(u_tilde, 0, setup);

E_hat[2]
E = total_kinetic_energy(ustart, setup, interpolate_first = true)
E = total_kinetic_energy(ustart, setup, interpolate_first = false)
E_hat[1]
Z = IncompressibleNavierStokes.total_enstropy(u_tilde, setup)
Z_field = IncompressibleNavierStokes.total_enstropy2(u_tilde, setup);
Z = IncompressibleNavierStokes.total_enstropy(ustart, setup)

diff = maximum(u_tilde-ustart)

heatmap(Z_field[:,:,10])
heatmap(ustart[:,:,10,1])
heatmap(u_tilde[:,:,10,1])
heatmap(real(u_hat[:,:,10,1]))
heatmap(u_out[:,:,10,1])

u_tilde = real(ifft(fft(ustart,[1,2,3]),[1,2,3]));
