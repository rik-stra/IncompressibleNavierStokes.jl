## This script runs multiple short WALE and Smagorinsky LES simulations to find the optimal closure model constant for the channel flow at Re = 180.

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using CUDA
using RikFlow
using JLD2
using Statistics


hf_file = @__DIR__()*"/output/paper_data_channel/HF/HF_channel_512_512_256_to_64_64_32_tsim15.0.jld2"
# Precision
T = Float64
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 1f   # we optimize over 10-time unit simulations
Δt = 0.005f

nx_les = 64
ny_les = 64
nz_les = 32
ArrayType = CuArray
kwargs = (;
    boundary_conditions = (
        (PeriodicBC(), PeriodicBC()),
        (DirichletBC(), DirichletBC()),
        (PeriodicBC(), PeriodicBC()),
    ),
    Re = 180f,
    bodyforce = (dim, x, y, z, t) -> 1 * (dim == 1),
    issteadybodyforce = true,
    backend = CUDABackend(),
    ArrayType = ArrayType,
)

setup = Setup(;
    x = (
        range(xlims..., nx_les + 1),
        range(ylims..., ny_les + 1), # tanh_grid(ylims..., ny + 1),
        range(zlims..., nz_les + 1)
    ),
    kwargs...,
);


@info "Grid size LF: $(nx_les) x $(ny_les) x $(nz_les)"
#amgx_objects = amgx_setup();
psolver = psolver_transform(setup);
u_aves_smag = []
c_s = 0.125:0.005:0.145 # Smagorinsky constant range
for c in c_s
    @info "c: $c"
    ustart = ArrayType(load(hf_file)["f"].data[1].u[1]);

    @info "Solving LES"
    # Solve DNS and store filtered quantities
    (; u, t), outputs = solve_unsteady(;
        # setup,
        setup = (; setup..., closure_model = IncompressibleNavierStokes.smagorinsky_closure),
        θ = T(c), 
        ustart,
        tlims = (0f, tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 100),
            fields = fieldsaver(; setup, nupdate = 200),  # by calling this BEFORE qoisaver, we also save the field at t=0!
        ),
        psolver,
    );

    u_fields = outputs.fields[2:end];
    us = stack(map(x -> x.u, u_fields));
    u_ave = mean(us[1:end-2, Int(end/2):Int(end/2)+1, 1:end-2, 1, :])
    @info "u_ave: $u_ave"
    push!(u_aves_smag, u_ave)

end
out_dir = @__DIR__()*"/output/smag"
ispath(out_dir) || mkpath(out_dir)
filename = out_dir*"optimize_smag.jld2"
jldsave(filename; c_s , u_aves_smag)

# Now we do the same for the WALE closure model
c_w = 0.49:0.005:0.55
u_aves_WALE = []

for c in c_w
    @info "c: $c"

    ustart = ArrayType(load(hf_file)["f"].data[1].u[1]);

    @info "Solving LES"
    # Solve DNS and store filtered quantities
    (; u, t), outputs = solve_unsteady(;
        # setup,
        setup = (; setup..., closure_model = IncompressibleNavierStokes.wale_closure),
        θ = T(c), 
        ustart,
        tlims = (0f, tsim),
        Δt,
        processors = (;
            log = timelogger(; nupdate = 100),
            fields = fieldsaver(; setup, nupdate = 200),  # by calling this BEFORE qoisaver, we also save the field at t=0!
        ),
        psolver,
    );

    u_fields = outputs.fields[2:end];
    us = stack(map(x -> x.u, u_fields));
    u_ave = mean(us[1:end-2, Int(end/2):Int(end/2)+1, 1:end-2, 1, :])
    @info "u_ave: $u_ave"
    push!(u_aves_WALE, u_ave)

end
out_dir = @__DIR__()*"/output/WALE"
ispath(out_dir) || mkpath(out_dir)
filename = out_dir*"optimize_WALE_centerv.jld2"
jldsave(filename; c_w, u_aves_WALE)
