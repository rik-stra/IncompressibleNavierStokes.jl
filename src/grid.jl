"""
Represent an `N`-dimensional space.
Returns `N` when called.

```julia-repl
julia> d = Dimension(3)
Dimension{3}()

julia> d()
3
```
"""
struct Dimension{N} end

Dimension(N) = Dimension{N}()

(::Dimension{N})() where {N} = N

"""
Get size of the largest grid element.
"""
function max_size(grid)
    (; Δ) = grid
    m = maximum.(Δ)
    sqrt(sum(m .^ 2))
end

"""
Create a nonuniform grid of `N + 1` points from `a` to `b` using a cosine
profile, i.e.

```math
x_i = a + \\frac{1}{2} \\left( 1 - \\cos \\left( \\pi \\frac{i}{n} \\right) \\right)
(b - a), \\quad i = 0, \\dots, N
```

See also [`stretched_grid`](@ref).
"""
function cosine_grid(a, b, N)
    T = typeof(a)
    i = T.(0:N)
    @. a + (b - a) * (1 - cospi(i / N)) / 2
end

"""
Create a nonuniform grid of `N + 1` points from `a` to `b` with a stretch
factor of `s`. If `s = 1`, return a uniform spacing from `a` to `b`. Otherwise,
return a vector ``x \\in \\mathbb{R}^{N + 1}`` such that ``x_n = a + \\sum_{i =
1}^n s^{i - 1} h`` for ``n = 0, \\dots , N``. Setting ``x_N = b`` then gives
``h = (b - a) \\frac{1 - s}{1 - s^N}``, resulting in

```math
x_n = a + (b - a) \\frac{1 - s^n}{1 - s^N}, \\quad n = 0, \\dots, N.
```

Note that `stretched_grid(a, b, N, s)[n]` corresponds to ``x_{n - 1}``.

See also [`cosine_grid`](@ref).
"""
function stretched_grid(a, b, N, s = 1)
    s > 0 || error("The stretch factor must be positive")
    if s ≈ 1
        LinRange(a, b, N + 1)
    else
        map(i -> a + (b - a) * (1 - s^i) / (1 - s^N), 0:N)
    end
end

"""
Create a nonuniform grid of `N + 1` points from `a` to `b`, as proposed
by Trias et al. [Trias2007](@cite).
"""
function tanh_grid(a, b, N, γ = typeof(a)(1))
    T = typeof(a)
    x = LinRange{T}(0, 1, N + 1)
    @. a + (b - a) * (1 + tanh(γ * (2 * x - 1)) / tanh(γ)) / 2
end

"""
Create nonuniform Cartesian box mesh `x[1]` × ... × `x[d]` with boundary
conditions `boundary_conditions`.
"""
function Grid(x, boundary_conditions; ArrayType = Array)
    # Kill all LinRanges etc.
    x = Array.(x)
    xlims = extrema.(x)

    D = length(x)
    dimension = Dimension(D)

    T = eltype(x[1])

    # Add offset positions for ghost volumes
    # For all BC, there is one ghost volume on each side,
    # but not all of the ``d + 1`` fields have a component inside this ghost
    # volume.
    for d = 1:D
        a, b = boundary_conditions[d]
        ghost_a!(a, x[d])
        ghost_b!(b, x[d])
    end

    # Number of finite volumes in each dimension, including ghost volumes
    N = length.(x) .- 1

    # Number of velocity DOFs in each dimension
    Nu = ntuple(D) do α
        ntuple(D) do β
            na = offset_u(boundary_conditions[β][1], α == β, false)
            nb = offset_u(boundary_conditions[β][2], α == β, true)
            N[β] - na - nb
        end
    end

    # Cartesian index ranges of velocity DOFs
    Iu = ntuple(D) do α
        Iuα = ntuple(D) do β
            na = offset_u(boundary_conditions[β][1], α == β, false)
            nb = offset_u(boundary_conditions[β][2], α == β, true)
            1+na:N[β]-nb
        end
        CartesianIndices(Iuα)
    end

    # Number of p DOFs in each dimension
    Np = ntuple(D) do α
        na = offset_p(boundary_conditions[α][1], false)
        nb = offset_p(boundary_conditions[α][2], true)
        N[α] - na - nb
    end

    # Cartesian index range of pressure DOFs
    Ip = CartesianIndices(ntuple(D) do α
        na = offset_p(boundary_conditions[α][1], false)
        nb = offset_p(boundary_conditions[α][2], true)
        1+na:N[α]-nb
    end)

    xp = ntuple(d -> (x[d][1:end-1] .+ x[d][2:end]) ./ 2, D)

    # Volume widths
    # Infinitely thin widths are set to `eps(T)` to avoid division by zero
    Δ = ntuple(D) do d
        Δ = diff(x[d])
        Δ[Δ.==0] .= eps(eltype(Δ))
        Δ
    end
    Δu = ntuple(D) do d
        Δu = push!(diff(xp[d]), Δ[d][end] / 2)
        Δu[Δu.==0] .= eps(eltype(Δu))
        Δu
    end

    # Reference volume sizes
    Ω = ones(T, N...)
    for d = 1:D
        Ω .*= reshape(Δ[d], ntuple(Returns(1), d - 1)..., :)
    end

    # # Velocity volume sizes
    # Ωu = ntuple(α -> ones(T, N), D)
    # for α = 1:D, β = 1:D
    #     Ωu[α] .*= reshape((α == β ? Δu : Δ)[β], ntuple(Returns(1), β - 1)..., :)
    # end

    # # Vorticity volume sizes
    # Ωω = ones(T, N)
    # for α = 1:D
    #     Ωω .*= reshape(Δu[α], ntuple(Returns(1), α - 1)..., :)
    # end

    # # Velocity volume mid-sections
    # Γu = ntuple(α -> ntuple(β -> ones(T, N), D), D)
    # for α = 1:D, β = 1:D, γ in ((1:β-1)..., (β+1:D)...)
    #     Γu[α][β] .*=
    #         reshape(γ == β ? 1 : γ == α ? Δu[γ] : Δ[γ], ntuple(Returns(1), γ - 1)..., :)
    # end

    # # Velocity points
    # Xu = ntuple(α -> ones(T, N))

    # Interpolation weights from α-face centers x_I to x_{I + δ(β) / 2}
    A = ntuple(
        α -> ntuple(
            β -> begin
                if α == β
                    # Interpolation from face center to volume center
                    Aαβ1 = fill(T(1 / 2), N[α])
                    Aαβ1[1] = 1
                    Aαβ2 = fill(T(1 / 2), N[α])
                    Aαβ2[end] = 1
                else
                    # Interpolation from α-face center to left (1) or right (2) α-face β-edge
                    # Aαβ1 = [(x[β][i] - xp[β][i-1]) / Δu[β][i-1] for i = 2:N[β]]
                    # Aαβ2 = 1 .- Aαβ1
                    Aαβ2 = [(x[β][i] - xp[β][i-1]) / Δu[β][i-1] for i = 2:N[β]]
                    Aαβ1 = 1 .- Aαβ2
                    pushfirst!(Aαβ1, 1)
                    push!(Aαβ2, 1)
                end
                (ArrayType(Aαβ1), ArrayType(Aαβ2))
            end,
            D,
        ),
        D,
    )

    # Grid quantities
    (;
        dimension,
        N,
        Nu,
        Np,
        Iu,
        Ip,
        xlims,
        x = ArrayType.(x),
        xp = ArrayType.(xp),
        Δ = ArrayType.(Δ),
        Δu = ArrayType.(Δu),
        Ω = ArrayType(Ω),
        # Ωu = ArrayType.(Ωu),
        # Ωω = ArrayType(Ωω),
        # Γu = ArrayType.(Γu),
        A,
    )
end
