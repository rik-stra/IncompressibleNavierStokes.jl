function strain_natural!(S, u, setup)
    (; grid, backend, workgroupsize) = setup
    (; Np, Ip, Δ, Δu) = grid
    I0 = getoffset(Ip)
    strain_natural_kernel!(backend, workgroupsize)(S, u, I0, Δ, Δu; ndrange = Np)
    S
end

@kernel function strain_natural_kernel!(S, u, I0::CartesianIndex{2}, Δ, Δu)
    I = @index(Global, Cartesian)
    I = I + I0
    ex, ey = unit_cartesian_indices(2)
    Δux, Δuy = Δu[1][I[1]], Δ[2][I[2]]
    Δvx, Δvy = Δ[1][I[1]], Δu[2][I[2]]
    ∂u∂x = (u[I, 1] - u[I-ex, 1]) / Δux
    ∂u∂y = (u[I+ey, 1] - u[I, 1]) / Δuy
    ∂v∂x = (u[I+ex, 2] - u[I, 2]) / Δvx
    ∂v∂y = (u[I, 2] - u[I-ey, 2]) / Δvy
    S.xx[I] = ∂u∂x
    S.yy[I] = ∂v∂y
    S.xy[I] = (∂u∂y + ∂v∂x) / 2
end

@kernel function strain_natural_kernel!(S, u, I0::CartesianIndex{3}, Δ, Δu)
    I = @index(Global, Cartesian)
    I = I + I0
    ex, ey, ez = unit_cartesian_indices(3)
    Δux, Δuy, Δuz = Δu[1][I[1]], Δ[2][I[2]], Δ[3][I[3]]
    Δvx, Δvy, Δvz = Δ[1][I[1]], Δu[2][I[2]], Δ[3][I[3]]
    Δwx, Δwy, Δwz = Δ[1][I[1]], Δ[2][I[2]], Δu[3][I[3]]
    ∂u∂x = (u[I, 1] - u[I-ex, 1]) / Δux
    ∂u∂y = (u[I+ey, 1] - u[I, 1]) / Δuy
    ∂u∂z = (u[I+ez, 1] - u[I, 1]) / Δuz
    ∂v∂x = (u[I+ex, 2] - u[I, 2]) / Δvx
    ∂v∂y = (u[I, 2] - u[I-ey, 2]) / Δvy
    ∂v∂z = (u[I+ez, 2] - u[I, 2]) / Δvz
    ∂w∂x = (u[I+ex, 3] - u[I, 3]) / Δwx
    ∂w∂y = (u[I+ey, 3] - u[I, 3]) / Δwy
    ∂w∂z = (u[I, 3] - u[I-ez, 3]) / Δwz
    S.xx[I] = ∂u∂x
    S.yy[I] = ∂v∂y
    S.zz[I] = ∂w∂z
    S.xy[I] = (∂u∂y + ∂v∂x) / 2
    S.xz[I] = (∂u∂z + ∂w∂x) / 2
    S.yz[I] = (∂v∂z + ∂w∂y) / 2
end

function smagorinsky_viscosity!(visc, S, θ, setup)
    (; grid, backend, workgroupsize) = setup
    (; Np, Ip, Δ) = grid
    I0 = getoffset(Ip)
    smagorinsky_viscosity_kernel!(backend, workgroupsize)(visc, S, I0, Δ, θ; ndrange = Np)
    visc
end

@kernel function smagorinsky_viscosity_kernel!(visc, S, I0::CartesianIndex{2}, Δ, θ)
    I = @index(Global, Cartesian)
    I = I + I0
    ex, ey = unit_cartesian_indices(2)
    d = gridsize(Δ, I)
    Sxx2 = S.xx[I]^2
    Syy2 = S.yy[I]^2
    Sxy2 = (S.xy[I]^2 + S.xy[I-ex]^2 + S.xy[I-ey]^2 + S.xy[I-ex-ey]^2) / 4
    visc[I] = θ^2 * d^2 * sqrt(2 * (Sxx2 + Syy2) + 4 * Sxy2)
end

@kernel function smagorinsky_viscosity_kernel!(visc, S, I0::CartesianIndex{3}, Δ, θ)
    I = @index(Global, Cartesian)
    I = I + I0
    ex, ey, ez = unit_cartesian_indices(3)
    d = gridsize(Δ, I)
    Sxx2 = S.xx[I]^2
    Syy2 = S.yy[I]^2
    Szz2 = S.zz[I]^2
    Sxy2 = (S.xy[I]^2 + S.xy[I-ex]^2 + S.xy[I-ey]^2 + S.xy[I-ex-ey]^2) / 4
    Sxz2 = (S.xz[I]^2 + S.xz[I-ex]^2 + S.xz[I-ez]^2 + S.xz[I-ex-ez]^2) / 4
    Syz2 = (S.yz[I]^2 + S.yz[I-ey]^2 + S.yz[I-ez]^2 + S.yz[I-ey-ez]^2) / 4
    visc[I] = θ^2 * d^2 * sqrt(2 * (Sxx2 + Syy2 + Szz2) + 4 * (Sxy2 + Sxz2 + Syz2))
end

function apply_eddy_viscosity!(σ, visc, setup)
    (; grid, backend, workgroupsize) = setup
    (; Np, Ip, Δ, Δu) = grid
    I0 = getoffset(Ip)
    apply_eddy_viscosity_kernel!(backend, workgroupsize)(σ, visc, I0, Δ, Δu; ndrange = Np)
    σ
end

@kernel function apply_eddy_viscosity_kernel!(σ, visc, I0::CartesianIndex{2}, Δ, Δu)
    I = @index(Global, Cartesian)
    I = I + I0
    ex, ey = unit_cartesian_indices(2)
    # TODO: Add interpolation weights here
    visc_xy = (visc[I] + visc[I+ex] + visc[I+ey] + visc[I+ex+ey]) / 4
    σ.xx[I] = 2 * visc[I] * σ.xx[I]
    σ.yy[I] = 2 * visc[I] * σ.yy[I]
    σ.xy[I] = 2 * visc_xy * σ.xy[I]
end

@kernel function apply_eddy_viscosity_kernel!(σ, visc, I0::CartesianIndex{3}, Δ, Δu)
    I = @index(Global, Cartesian)
    I = I + I0
    ex, ey, ez = unit_cartesian_indices(3)
    # TODO: Add interpolation weights here
    visc_xy = (visc[I] + visc[I+ex] + visc[I+ey] + visc[I+ex+ey]) / 4
    visc_xz = (visc[I] + visc[I+ex] + visc[I+ez] + visc[I+ex+ez]) / 4
    visc_yz = (visc[I] + visc[I+ey] + visc[I+ez] + visc[I+ey+ez]) / 4
    σ.xx[I] = 2 * visc[I] * σ.xx[I]
    σ.yy[I] = 2 * visc[I] * σ.yy[I]
    σ.zz[I] = 2 * visc[I] * σ.zz[I]
    σ.xy[I] = 2 * visc_xy * σ.xy[I]
    σ.xz[I] = 2 * visc_xz * σ.xz[I]
    σ.yz[I] = 2 * visc_yz * σ.yz[I]
end

function divoftensor_natural!(c, σ, setup)
    (; grid, backend, workgroupsize) = setup
    (; Np, Ip, Δ, Δu) = grid
    I0 = getoffset(Ip)
    divoftensor_natural_kernel!(backend, workgroupsize)(c, σ, I0, Δ, Δu; ndrange = Np)
    c
end

@kernel function divoftensor_natural_kernel!(c, σ, I0::CartesianIndex{2}, Δ, Δu)
    I = @index(Global, Cartesian)
    I = I + I0
    ex, ey = unit_cartesian_indices(2)
    Δpx, Δpy = Δ[1][I[1]], Δ[2][I[2]]
    Δux, Δuy = Δu[1][I[1]], Δu[2][I[2]]
    ∂σxx∂x = (σ.xx[I+ex] - σ.xx[I]) / Δux
    ∂σxy∂y = (σ.xy[I] - σ.xy[I-ey]) / Δpy
    ∂σyx∂x = (σ.xy[I] - σ.xy[I-ex]) / Δpx
    ∂σyy∂y = (σ.yy[I+ey] - σ.yy[I]) / Δuy
    c[I, 1] = ∂σxx∂x + ∂σxy∂y
    c[I, 2] = ∂σyx∂x + ∂σyy∂y
end

@kernel function divoftensor_natural_kernel!(c, σ, I0::CartesianIndex{3}, Δ, Δu)
    I = @index(Global, Cartesian)
    I = I + I0
    ex, ey, ez = unit_cartesian_indices(3)
    Δpx, Δpy, Δpz = Δ[1][I[1]], Δ[2][I[2]], Δ[3][I[3]]
    Δux, Δuy, Δuz = Δu[1][I[1]], Δu[2][I[2]], Δu[3][I[3]]
    ∂σxx∂x = (σ.xx[I+ex] - σ.xx[I]) / Δux
    ∂σxy∂y = (σ.xy[I] - σ.xy[I-ey]) / Δpy
    ∂σxz∂z = (σ.xz[I] - σ.xz[I-ez]) / Δpz
    ∂σyx∂x = (σ.xy[I] - σ.xy[I-ex]) / Δpx
    ∂σyy∂y = (σ.yy[I+ey] - σ.yy[I]) / Δuy
    ∂σyz∂z = (σ.yz[I] - σ.yz[I-ez]) / Δpz
    ∂σzx∂x = (σ.xz[I] - σ.xz[I-ex]) / Δpx
    ∂σzy∂y = (σ.yz[I] - σ.yz[I-ey]) / Δpy
    ∂σzz∂z = (σ.zz[I+ez] - σ.zz[I]) / Δuz
    c[I, 1] = ∂σxx∂x + ∂σxy∂y + ∂σxz∂z
    c[I, 2] = ∂σyx∂x + ∂σyy∂y + ∂σyz∂z
    c[I, 3] = ∂σzx∂x + ∂σzy∂y + ∂σzz∂z
end

function smagorinsky_closure_natural(setup)
    (; dimension, x, N) = setup.grid
    D = dimension()
    T = eltype(x[1])
    visc = scalarfield(setup)
    σ = if D == 2
        (; xx = scalarfield(setup), yy = scalarfield(setup), xy = scalarfield(setup))
    elseif D == 3
        (;
            xx = scalarfield(setup),
            yy = scalarfield(setup),
            zz = scalarfield(setup),
            xy = scalarfield(setup),
            xz = scalarfield(setup),
            yz = scalarfield(setup),
        )
    end
    c = vectorfield(setup)
    function closure(u, θ)
        strain_natural!(σ, u, setup)
        smagorinsky_viscosity!(visc, σ, θ, setup)
        apply_eddy_viscosity!(σ, visc, setup)
        divoftensor_natural!(c, σ, setup)
        c
    end
end
