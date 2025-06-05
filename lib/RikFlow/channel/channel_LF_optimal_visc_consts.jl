## Turbulent channel flow

if false
    include("../../../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
#using CairoMakie
using CUDA
using RikFlow
using JLD2
using Statistics



# Precision
T = Float64
f = one(T)

# Domain
xlims = 0f, 4f * pi
ylims = 0f, 2f
zlims = 0f, 4f / 3f * pi

tsim = 10f
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

target_u_ave = 15.647



c_s = 0.1:0.005:0.2
u_aves_smag = []

for c in c_s
    @info "c: $c"

    hf_file = @__DIR__()*"/output/HF/HF_channel_6qoinew_mirror_2framerate_512_512_256_to_64_64_32_tsim15.0.jld2"
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
    u_ave = mean(us[1:end-2, 2:end-1, 1:end-2, 1, :])

    push!(u_aves_smag, u_ave)

end

filename = @__DIR__()*"/output/smag/optimize_smag.jld2"
jldsave(filename; c_s , u_aves_smag)

c_w = 0.5:0.005:0.54
u_aves_WALE = []

for c in c_w
    @info "c: $c"

    hf_file = @__DIR__()*"/output/HF/HF_channel_6qoinew_mirror_2framerate_512_512_256_to_64_64_32_tsim15.0.jld2"
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
    u_ave = mean(us[1:end-2, 2:end-1, 1:end-2, 1, :])

    push!(u_aves_WALE, u_ave)

end

filename = @__DIR__()*"/output/WALE/optimize_WALE.jld2"
jldsave(filename; c_w, u_aves_WALE)


# u_ave target   15.647
# theta = 0.5 -> u_ave 15.622711
# theta = 0.6 -> u_ave 15.774

