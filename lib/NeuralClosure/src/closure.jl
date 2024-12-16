"""
Wrap closure model and parameters so that it can be used in the solver.
"""
function wrappedclosure(m, setup)
    (; Iu) = setup.grid
    inside = Iu[1]
    @assert all(==(inside), Iu) "Only periodic grids are supported"
    function neuralclosure(u, θ)
        s = size(u)
        # u = view(u, inside, :)
        u = u[inside, :]
        u = reshape(u, size(u)..., 1) # Add sample dim
        mu = m(u, θ)
        mu = pad_circular(mu, 1)
        mu = reshape(mu, s) # Remove sample dim
    end
end

"""
Create neural closure model from layers.
"""
function create_closure(layers...; rng)
    chain = Chain(layers...)

    # Create parameter vector (empty state)
    params, state = Lux.setup(rng, chain)
    θ = ComponentArray(params)

    # Compute closure term for given parameters
    closure(u, θ) = first(chain(u, θ, state))

    closure, θ
end

"""
Create tensor basis closure.
"""
function create_tensorclosure(layers...; setup, rng)
    D = setup.grid.dimension()
    cnn, θ = create_closure(layers...; rng)
    function closure(u, θ)
        B, V = tensorbasis(u, setup)
        V = stack(V)
        α = cnn(V, θ)
        τ = sum(k -> α[ntuple(Returns(:), D)..., k] .* B[k], 1:length(B))
    end
end

"""
Interpolate velocity components to volume centers.
"""
function collocate(u)
    sz..., D, _ = size(u)
    # for α = 1:D
    #     v = selectdim(u, D + 1, α)
    #     v = (v + circshift(v, ntuple(β -> α == β ? -1 : 0, D + 1))) / 2
    # end
    if D == 2
        # TODO: Check if this is more efficient as
        #   a convolution with the two channel kernels
        #   [1 1; 0 0] / 2
        #   and
        #   [0 1; 0 1] / 2
        # TODO: Maybe skip this step entirely and learn the
        #   collocation function subject to the skewness
        #   constraint (left-skewed kernel from staggered right face to center)
        a = selectdim(u, 3, 1)
        b = selectdim(u, 3, 2)
        a = (a .+ circshift(a, (1, 0, 0))) ./ 2
        b = (b .+ circshift(b, (0, 1, 0))) ./ 2
        a = reshape(a, sz..., 1, :)
        b = reshape(b, sz..., 1, :)
        cat(a, b; dims = 3)
    elseif D == 3
        a = selectdim(u, 4, 1)
        b = selectdim(u, 4, 2)
        c = selectdim(u, 4, 3)
        a = (a .+ circshift(a, (1, 0, 0, 0))) ./ 2
        b = (b .+ circshift(b, (0, 1, 0, 0))) ./ 2
        c = (c .+ circshift(c, (0, 0, 1, 0))) ./ 2
        a = reshape(a, sz..., 1, :)
        b = reshape(b, sz..., 1, :)
        c = reshape(c, sz..., 1, :)
        cat(a, b, c; dims = 4)
    end
end

"""
Interpolate closure force from volume centers to volume faces.
"""
function decollocate(u)
    sz..., D, _ = size(u)
    # for α = 1:D
    #     v = selectdim(u, D + 1, α)
    #     v = (v + circshift(v, ntuple(β -> α == β ? -1 : 0, D + 1))) / 2
    # end
    if D == 2
        a = selectdim(u, 3, 1)
        b = selectdim(u, 3, 2)
        a = (a .+ circshift(a, (-1, 0, 0))) ./ 2
        b = (b .+ circshift(b, (0, -1, 0))) ./ 2
        # a = circshift(a, (1, 0, 0)) .- a
        # b = circshift(b, (0, 1, 0)) .- b
        a = reshape(a, sz..., 1, :)
        b = reshape(b, sz..., 1, :)
        cat(a, b; dims = 3)
    elseif D == 3
        a = selectdim(u, 4, 1)
        b = selectdim(u, 4, 2)
        c = selectdim(u, 4, 3)
        a = (a .+ circshift(a, (-1, 0, 0, 0))) ./ 2
        b = (b .+ circshift(b, (0, -1, 0, 0))) ./ 2
        c = (c .+ circshift(c, (0, 0, -1, 0))) ./ 2
        # a = circshift(a, (1, 0, 0, 0)) .- a
        # b = circshift(b, (0, 1, 0, 0)) .- b
        # c = circshift(c, (0, 0, 1, 0)) .- c
        a = reshape(a, sz..., 1, :)
        b = reshape(b, sz..., 1, :)
        c = reshape(c, sz..., 1, :)
        cat(a, b, c; dims = 4)
    end
end
